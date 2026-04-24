"""Local backend: faster-whisper STT + Ollama LLM + Kokoro-ONNX or ElevenLabs TTS.

No cloud LLM or STT required. TTS can be local (Kokoro) or ElevenLabs.

Env vars (all optional):
  LOCAL_STT_MODEL      - Whisper model size        (default: medium)
  LOCAL_LLM_MODEL      - Ollama model name         (default: llama3.2:3b)
  LOCAL_OLLAMA_HOST    - Ollama base URL            (default: http://localhost:11434)
  LOCAL_TTS_PROVIDER   - "kokoro" or "elevenlabs"  (default: kokoro)
  LOCAL_TTS_VOICE      - Kokoro voice name          (default: af_heart)
  LOCAL_TTS_LANG       - Kokoro language code       (default: en-us)
  ELEVENLABS_API_KEY   - ElevenLabs API key
  ELEVENLABS_VOICE     - ElevenLabs voice name      (default: Rachel)
  ELEVENLABS_MODEL     - ElevenLabs model ID        (default: eleven_multilingual_v2)
"""

import json
import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample
from fastrtc import AdditionalOutputs, AsyncStreamHandler, audio_to_int16, wait_for_item

from reachy_mini_conversation_app.prompts import get_session_instructions
from reachy_mini_conversation_app.tools.core_tools import (
    ToolDependencies,
    dispatch_tool_call,
    get_tool_specs,
)
from reachy_mini_conversation_app.tools.background_tool_manager import BackgroundToolManager


logger = logging.getLogger(__name__)

LOCAL_INPUT_SAMPLE_RATE = 16_000
LOCAL_OUTPUT_SAMPLE_RATE = 24_000

# Energy-based VAD parameters
_ENERGY_THRESHOLD = 150      # RMS threshold (int16 scale) for speech detection
_SPEECH_START_FRAMES = 4     # consecutive loud frames before recording starts
_SPEECH_END_FRAMES = 30      # consecutive quiet frames after speech → end of turn (~600 ms)
_MAX_SPEECH_SECONDS = 30

# Default models (overridable via env)
_DEFAULT_WHISPER_MODEL = os.getenv("LOCAL_STT_MODEL", "medium")
_DEFAULT_OLLAMA_MODEL = os.getenv("LOCAL_LLM_MODEL", "llama3.2:3b")
_DEFAULT_OLLAMA_HOST = os.getenv("LOCAL_OLLAMA_HOST", "http://localhost:11434")
_DEFAULT_TTS_PROVIDER = os.getenv("LOCAL_TTS_PROVIDER", "kokoro")
_DEFAULT_TTS_VOICE = os.getenv("LOCAL_TTS_VOICE", "af_heart")
_DEFAULT_TTS_LANG = os.getenv("LOCAL_TTS_LANG", "en-us")
_DEFAULT_EL_KEY = os.getenv("ELEVENLABS_API_KEY", "")
_DEFAULT_EL_VOICE = os.getenv("ELEVENLABS_VOICE", "Rachel")
_DEFAULT_EL_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")

# Kokoro model on HuggingFace
_KOKORO_HF_REPO = "hexgrad/Kokoro-82M-v1.1-ONNX"
_KOKORO_MODEL_FILE = "kokoro-v1.1-zh.onnx"
_KOKORO_VOICES_FILE = "voices-v1.0.bin"

LOCAL_AVAILABLE_VOICES: List[str] = [
    "af_heart", "af_bella", "af_sarah", "af_sky",
    "am_adam", "am_michael",
    "bf_emma", "bf_isabella",
    "bm_george", "bm_lewis",
]
LOCAL_DEFAULT_VOICE = _DEFAULT_TTS_VOICE


def _to_ollama_tool_specs(specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert Realtime-API flat tool specs to Ollama/OpenAI-Chat nested format."""
    return [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s.get("description", ""),
                "parameters": s.get("parameters", {}),
            },
        }
        for s in specs
    ]


class LocalHandler(AsyncStreamHandler):
    """Local backend: faster-whisper STT + Ollama LLM + Kokoro-ONNX or ElevenLabs TTS."""

    def __init__(
        self,
        deps: ToolDependencies,
        gradio_mode: bool = False,
        instance_path: Optional[str] = None,
        startup_voice: Optional[str] = None,
        whisper_model: str = _DEFAULT_WHISPER_MODEL,
        ollama_model: str = _DEFAULT_OLLAMA_MODEL,
        ollama_host: str = _DEFAULT_OLLAMA_HOST,
        tts_provider: str = _DEFAULT_TTS_PROVIDER,
        tts_voice: str = _DEFAULT_TTS_VOICE,
        tts_lang: str = _DEFAULT_TTS_LANG,
        elevenlabs_key: str = _DEFAULT_EL_KEY,
        elevenlabs_voice: str = _DEFAULT_EL_VOICE,
        elevenlabs_model: str = _DEFAULT_EL_MODEL,
    ) -> None:
        super().__init__(
            expected_layout="mono",
            output_sample_rate=LOCAL_OUTPUT_SAMPLE_RATE,
            input_sample_rate=LOCAL_INPUT_SAMPLE_RATE,
        )
        self.deps = deps
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path
        self.whisper_model_name = whisper_model
        self.ollama_model = ollama_model
        self.ollama_host = ollama_host
        self._tts_provider = tts_provider
        self._voice = startup_voice or tts_voice
        self._tts_lang = tts_lang
        self._elevenlabs_key = elevenlabs_key
        self._elevenlabs_voice = elevenlabs_voice
        self._elevenlabs_model = elevenlabs_model

        self.output_queue: asyncio.Queue[
            Tuple[int, NDArray[np.int16]] | AdditionalOutputs
        ] = asyncio.Queue()
        self.tool_manager = BackgroundToolManager()
        self.last_activity_time: float = 0.0

        # VAD state
        self._pre_buffer: List[NDArray[np.int16]] = []
        self._audio_buffer: List[NDArray[np.int16]] = []
        self._loud_frames: int = 0
        self._quiet_frames: int = 0
        self._is_speaking: bool = False
        self._processing: bool = False

        # Conversation history
        self._messages: List[Dict[str, Any]] = []

        # Lazy-loaded model instances
        self._stt = None
        self._tts = None          # Kokoro instance
        self._el_client = None    # ElevenLabs client
        self._ollama = None

    # ------------------------------------------------------------------ #
    # AsyncStreamHandler interface                                         #
    # ------------------------------------------------------------------ #

    def copy(self) -> "LocalHandler":
        return LocalHandler(
            self.deps,
            self.gradio_mode,
            self.instance_path,
            startup_voice=self._voice,
            whisper_model=self.whisper_model_name,
            ollama_model=self.ollama_model,
            ollama_host=self.ollama_host,
            tts_provider=self._tts_provider,
            tts_voice=self._voice,
            tts_lang=self._tts_lang,
            elevenlabs_key=self._elevenlabs_key,
            elevenlabs_voice=self._elevenlabs_voice,
            elevenlabs_model=self._elevenlabs_model,
        )

    async def start_up(self) -> None:
        self.last_activity_time = asyncio.get_event_loop().time()
        loop = asyncio.get_event_loop()

        # STT — faster-whisper
        logger.info("Loading faster-whisper (%s)…", self.whisper_model_name)
        from faster_whisper import WhisperModel  # type: ignore[import]
        self._stt = await loop.run_in_executor(
            None,
            lambda: WhisperModel(self.whisper_model_name, device="cpu", compute_type="int8"),
        )
        logger.info("faster-whisper ready")

        # TTS
        if self._tts_provider == "elevenlabs":
            await self._init_elevenlabs()
        else:
            await self._init_kokoro(loop)

        # LLM — Ollama
        import ollama  # type: ignore[import]
        self._ollama = ollama.AsyncClient(host=self.ollama_host)
        logger.info("Ollama client ready (model=%s)", self.ollama_model)

        self.tool_manager.start_up(tool_callbacks=[])
        self._messages = [{"role": "system", "content": get_session_instructions()}]

    async def _init_kokoro(self, loop: asyncio.AbstractEventLoop) -> None:
        logger.info("Loading Kokoro-ONNX TTS…")
        from kokoro_onnx import Kokoro  # type: ignore[import]
        import huggingface_hub

        def _load() -> Any:
            mp = huggingface_hub.hf_hub_download(_KOKORO_HF_REPO, _KOKORO_MODEL_FILE)
            vp = huggingface_hub.hf_hub_download(_KOKORO_HF_REPO, _KOKORO_VOICES_FILE)
            return Kokoro(mp, vp)

        self._tts = await loop.run_in_executor(None, _load)
        logger.info("Kokoro-ONNX ready (voice=%s lang=%s)", self._voice, self._tts_lang)

    async def _init_elevenlabs(self) -> None:
        from elevenlabs.client import ElevenLabs  # type: ignore[import]
        if not self._elevenlabs_key:
            raise RuntimeError("ELEVENLABS_API_KEY is required for ElevenLabs TTS")
        self._el_client = ElevenLabs(api_key=self._elevenlabs_key)
        logger.info(
            "ElevenLabs TTS ready (voice=%s model=%s)",
            self._elevenlabs_voice, self._elevenlabs_model,
        )

    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        if self._stt is None:
            return

        input_rate, audio = frame

        if audio.ndim == 2:
            if audio.shape[1] > audio.shape[0]:
                audio = audio.T
            if audio.shape[1] > 1:
                audio = audio[:, 0]

        if input_rate != LOCAL_INPUT_SAMPLE_RATE:
            n_out = int(len(audio) * LOCAL_INPUT_SAMPLE_RATE / input_rate)
            audio = resample(audio, n_out)
        audio = audio_to_int16(audio)

        rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
        is_loud = rms > _ENERGY_THRESHOLD

        if is_loud:
            self._loud_frames += 1
            self._quiet_frames = 0
            if self._loud_frames >= _SPEECH_START_FRAMES and not self._is_speaking:
                self._is_speaking = True
                self._audio_buffer = list(self._pre_buffer)
                self._pre_buffer.clear()
        else:
            self._loud_frames = 0
            if self._is_speaking:
                self._quiet_frames += 1

        if self._is_speaking:
            self._audio_buffer.append(audio)
        else:
            self._pre_buffer.append(audio)
            if len(self._pre_buffer) > _SPEECH_START_FRAMES + 2:
                self._pre_buffer.pop(0)

        samples_per_frame = max(len(audio), 1)
        max_frames = int(_MAX_SPEECH_SECONDS * LOCAL_INPUT_SAMPLE_RATE / samples_per_frame)

        turn_ended = (
            self._is_speaking
            and self._quiet_frames >= _SPEECH_END_FRAMES
            and not self._processing
        )
        max_reached = (
            self._is_speaking
            and len(self._audio_buffer) > max_frames
            and not self._processing
        )

        if turn_ended or max_reached:
            audio_data = np.concatenate(self._audio_buffer)
            self._audio_buffer.clear()
            self._pre_buffer.clear()
            self._is_speaking = False
            self._quiet_frames = 0
            self._loud_frames = 0
            self._processing = True
            asyncio.create_task(self._process_turn(audio_data))

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        return await wait_for_item(self.output_queue)

    # ------------------------------------------------------------------ #
    # Pipeline                                                             #
    # ------------------------------------------------------------------ #

    async def _process_turn(self, audio: NDArray[np.int16]) -> None:
        try:
            text = (await self._transcribe(audio)).strip()
            if not text:
                return
            logger.info("User: %s", text)
            await self.output_queue.put(AdditionalOutputs({"role": "user", "content": text}))
            self._messages.append({"role": "user", "content": text})
            await self._llm_then_tts()
        finally:
            self._processing = False
            self.last_activity_time = asyncio.get_event_loop().time()

    async def _transcribe(self, audio: NDArray[np.int16]) -> str:
        loop = asyncio.get_event_loop()
        audio_f32 = audio.astype(np.float32) / 32768.0
        audio_sec = len(audio) / LOCAL_INPUT_SAMPLE_RATE
        t0 = time.perf_counter()

        def _run() -> str:
            segments, _ = self._stt.transcribe(audio_f32, beam_size=5)
            return " ".join(s.text.strip() for s in segments)

        result = await loop.run_in_executor(None, _run)
        try:
            from reachy_mini_conversation_app.usage_tracker import record_stt
            record_stt(f"faster-whisper/{self.whisper_model_name}", audio_sec,
                       int((time.perf_counter() - t0) * 1000))
        except Exception:
            pass
        return result

    async def _llm_then_tts(self) -> None:
        tool_specs = _to_ollama_tool_specs(get_tool_specs())
        full_text = ""
        sentence_buf = ""
        tool_calls_buf: List[Any] = []
        t0_llm = time.perf_counter()

        try:
            stream = await self._ollama.chat(
                model=self.ollama_model,
                messages=self._messages,
                tools=tool_specs or None,
                stream=True,
            )

            async for chunk in stream:
                msg = chunk.message
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tool_calls_buf.extend(msg.tool_calls)
                if msg.content:
                    full_text += msg.content
                    sentence_buf += msg.content
                    sentence_buf = await self._flush_sentences(sentence_buf, force=False)
                if chunk.done:
                    try:
                        from reachy_mini_conversation_app.usage_tracker import record_llm
                        record_llm(
                            self.ollama_model,
                            tokens_in=getattr(chunk, "prompt_eval_count", 0) or 0,
                            tokens_out=getattr(chunk, "eval_count", 0) or 0,
                            latency_ms=int((time.perf_counter() - t0_llm) * 1000),
                        )
                    except Exception:
                        pass
                    break

            if sentence_buf.strip():
                await self._flush_sentences(sentence_buf, force=True)

            if tool_calls_buf:
                self._messages.append({
                    "role": "assistant",
                    "content": full_text or "",
                    "tool_calls": [
                        {"function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in tool_calls_buf
                    ],
                })
                await self._execute_tools(tool_calls_buf)
                return

            if full_text:
                self._messages.append({"role": "assistant", "content": full_text})
                await self.output_queue.put(
                    AdditionalOutputs({"role": "assistant", "content": full_text})
                )

        except Exception:
            logger.exception("LLM error")

    async def _flush_sentences(self, buf: str, force: bool) -> str:
        while True:
            idx = -1
            for t in (".", "!", "?", "\n"):
                pos = buf.find(t)
                if pos != -1 and (idx == -1 or pos < idx):
                    idx = pos
            if idx == -1:
                break
            sentence = buf[: idx + 1].strip()
            buf = buf[idx + 1 :].lstrip()
            if sentence:
                await self._speak(sentence)
        if force and buf.strip():
            await self._speak(buf.strip())
            buf = ""
        return buf

    async def _speak(self, text: str) -> None:
        if self._tts_provider == "elevenlabs":
            await self._speak_elevenlabs(text)
        else:
            await self._speak_kokoro(text)

    async def _speak_kokoro(self, text: str) -> None:
        loop = asyncio.get_event_loop()
        t0 = time.perf_counter()

        def _run() -> Tuple[NDArray[np.float32], int]:
            return self._tts.create(text, voice=self._voice, speed=1.0, lang=self._tts_lang)

        try:
            samples, sr = await loop.run_in_executor(None, _run)
            audio_i16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
            if sr != LOCAL_OUTPUT_SAMPLE_RATE:
                n_out = int(len(audio_i16) * LOCAL_OUTPUT_SAMPLE_RATE / sr)
                audio_i16 = resample(audio_i16, n_out).astype(np.int16)
            audio_sec = len(audio_i16) / LOCAL_OUTPUT_SAMPLE_RATE
            try:
                from reachy_mini_conversation_app.usage_tracker import record_tts
                record_tts(f"kokoro/{self._voice}", len(text), audio_sec,
                           int((time.perf_counter() - t0) * 1000))
            except Exception:
                pass
            chunk = 4096
            for i in range(0, len(audio_i16), chunk):
                await self.output_queue.put((LOCAL_OUTPUT_SAMPLE_RATE, audio_i16[i : i + chunk]))
        except Exception:
            logger.exception("Kokoro TTS error for %r", text[:60])

    async def _speak_elevenlabs(self, text: str) -> None:
        loop = asyncio.get_event_loop()
        queue = self.output_queue
        t0 = time.perf_counter()
        total_samples = 0

        def _stream() -> None:
            nonlocal total_samples
            audio_gen = self._el_client.generate(
                text=text,
                voice=self._elevenlabs_voice,
                model=self._elevenlabs_model,
                output_format="pcm_24000",
                stream=True,
            )
            for chunk_bytes in audio_gen:
                if chunk_bytes:
                    audio_chunk = np.frombuffer(chunk_bytes, dtype=np.int16).copy()
                    total_samples += len(audio_chunk)
                    loop.call_soon_threadsafe(
                        queue.put_nowait, (LOCAL_OUTPUT_SAMPLE_RATE, audio_chunk)
                    )

        try:
            await loop.run_in_executor(None, _stream)
            try:
                from reachy_mini_conversation_app.usage_tracker import record_tts
                record_tts(f"elevenlabs/{self._elevenlabs_voice}", len(text),
                           total_samples / LOCAL_OUTPUT_SAMPLE_RATE,
                           int((time.perf_counter() - t0) * 1000))
            except Exception:
                pass
        except Exception:
            logger.exception("ElevenLabs TTS error for %r", text[:60])

    async def _execute_tools(self, tool_calls: List[Any]) -> None:
        for tc in tool_calls:
            name = tc.function.name
            args = tc.function.arguments
            args_json = json.dumps(args) if isinstance(args, dict) else (args or "{}")
            logger.info("Tool: %s(%s)", name, args_json[:80])
            await self.output_queue.put(
                AdditionalOutputs({"role": "assistant", "content": f"🛠️ Using tool {name}…"})
            )
            result = await dispatch_tool_call(name, args_json, self.deps)
            self._messages.append({"role": "tool", "content": json.dumps(result)})
        await self._llm_then_tts()

    # ------------------------------------------------------------------ #
    # Voice / personality                                                  #
    # ------------------------------------------------------------------ #

    async def apply_personality(self, profile: Optional[str]) -> str:
        from reachy_mini_conversation_app.config import set_custom_profile
        set_custom_profile(profile)
        self._messages = [{"role": "system", "content": get_session_instructions()}]
        return f"Applied personality: {profile or 'default'}"

    async def change_voice(self, voice: str) -> str:
        if self._tts_provider == "elevenlabs":
            self._elevenlabs_voice = voice
        else:
            self._voice = voice
        return f"Voice changed to {voice}"

    def get_current_voice(self) -> str:
        return self._elevenlabs_voice if self._tts_provider == "elevenlabs" else self._voice

    async def send_idle_signal(self, idle_duration: float) -> None:
        pass

    async def shutdown(self) -> None:
        await self.tool_manager.shutdown()
