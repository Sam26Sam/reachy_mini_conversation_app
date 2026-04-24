#!/usr/bin/env python3
"""Reachy Mini Dashboard — configuration + lancement depuis le navigateur.

Ce dashboard tourne en permanence sur http://localhost:7860
et gère toute la configuration ainsi que le démarrage de l'application.
"""

import os
import sys
import time
import subprocess
import threading
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv, set_key


# ── Paths ─────────────────────────────────────────────────────────────────────

DASHBOARD_PORT = 7860
APP_PORT = 7861
APP_DIR = Path(__file__).parent
ENV_PATH = APP_DIR / ".env"
LOG_PATH = Path("/tmp/reachy_app.log")

# ── Global app process ────────────────────────────────────────────────────────

_app_proc: subprocess.Popen | None = None
_app_lock = threading.Lock()

# ── Static catalogues ─────────────────────────────────────────────────────────

BACKEND_OPTIONS = [
    ("🟢  Local — gratuit, 100% local  (faster-whisper + Ollama + Kokoro)", "local"),
    ("💳  OpenAI Realtime — payant  (gpt-4o · STT + LLM + TTS intégrés)", "openai"),
    ("🟡  Gemini Live — gratuit limité  (gemini-3.1-flash-live-preview)", "gemini"),
]

STT_OPTIONS = [
    ("tiny  — ⚡ ultra-rapide · qualité basique", "tiny"),
    ("base  — ⚡ très rapide · qualité correcte", "base"),
    ("small  — 🔵 rapide · bonne qualité", "small"),
    ("medium  — ⭐ recommandé · meilleur compromis", "medium"),
    ("large-v3  — 🏆 meilleure qualité · plus lent", "large-v3"),
]

VLM_OPTIONS = [
    ("SmolVLM2 · 500M  — ⚡ ultra-léger  (~1 GB à télécharger)", "HuggingFaceTB/SmolVLM2-500M-Instruct"),
    ("SmolVLM2 · 2.2B  — ⭐ recommandé  (~4 GB à télécharger)", "HuggingFaceTB/SmolVLM2-2.2B-Instruct"),
]

# (nom affiché, tag ollama, taille, description)
OLLAMA_CATALOG: list[tuple[str, str, str, str]] = [
    # ── Ultra-légers (< 1 GB) ─────────────────────────────────────────────────
    ("Smollm2 · 360M",       "smollm2:360m",     "0.2 GB", "Le plus petit — pour Mac avec très peu de RAM"),
    ("Qwen 2.5 · 0.5B",      "qwen2.5:0.5b",     "0.4 GB", "Ultra-léger · multilingue FR/EN/ZH"),
    ("Gemma 3 · 1B",         "gemma3:1b",         "0.8 GB", "Léger · Google · multilingue"),
    ("Smollm2 · 1.7B",       "smollm2:1.7b",     "1.0 GB", "Léger et efficace"),
    ("Qwen 2.5 · 1.5B",      "qwen2.5:1.5b",     "1.0 GB", "Léger · multilingue FR/EN/ZH"),
    ("DeepSeek-R1 · 1.5B",   "deepseek-r1:1.5b", "1.1 GB", "Raisonnement (thinking) · léger"),
    ("Llama 3.2 · 1B",       "llama3.2:1b",      "1.3 GB", "Ultra-rapide · idéal pour débuter"),
    # ── Compacts (1–3 GB) ─────────────────────────────────────────────────────
    ("Phi 4 Mini · 3.8B",    "phi4-mini:3.8b",   "2.5 GB", "Compact · performant · Microsoft"),
    ("Llama 3.2 · 3B ⭐",    "llama3.2:3b",      "2.0 GB", "Recommandé · meilleur compromis vitesse/qualité"),
    ("Qwen 2.5 · 3B",        "qwen2.5:3b",       "1.9 GB", "Rapide · multilingue FR/EN/ZH"),
    ("Gemma 3 · 4B",         "gemma3:4b",        "3.3 GB", "Bonne qualité · Google · multilingue"),
    # ── Haute qualité (4–8 GB) ────────────────────────────────────────────────
    ("Qwen 2.5 · 7B",        "qwen2.5:7b",       "4.7 GB", "Haute qualité · multilingue FR/EN/ZH"),
    ("Llama 3.1 · 8B",       "llama3.1:8b",      "4.7 GB", "Haute qualité · conversations avancées"),
    ("Mistral · 7B",         "mistral:7b",        "4.1 GB", "Haute qualité · excellent en français"),
    ("DeepSeek-R1 · 7B",     "deepseek-r1:7b",   "4.7 GB", "Raisonnement (thinking) · haute qualité"),
    # ── Très haute qualité (> 8 GB) ───────────────────────────────────────────
    ("Qwen 2.5 · 14B",       "qwen2.5:14b",      "9.0 GB",  "Très haute qualité · multilingue"),
    ("Phi 4 · 14B",          "phi4:14b",         "9.1 GB",  "Très haute qualité · Microsoft"),
    ("Gemma 3 · 12B",        "gemma3:12b",       "8.1 GB",  "Très haute qualité · Google"),
    ("Mistral Small · 22B",  "mistral-small:22b","13.0 GB", "Qualité maximale locale"),
]

TTS_PROVIDER_OPTIONS = [
    ("🟢  Kokoro-ONNX — 100% local, gratuit", "kokoro"),
    ("🔵  ElevenLabs — qualité professionnelle, payant", "elevenlabs"),
]

KOKORO_VOICES = [
    ("af_heart  — American Female · chaleureuse  ⭐", "af_heart"),
    ("af_bella  — American Female · bella", "af_bella"),
    ("af_sarah  — American Female · sarah", "af_sarah"),
    ("af_sky  — American Female · sky", "af_sky"),
    ("am_adam  — American Male · adam", "am_adam"),
    ("am_michael  — American Male · michael", "am_michael"),
    ("bf_emma  — British Female · emma", "bf_emma"),
    ("bf_isabella  — British Female · isabella", "bf_isabella"),
    ("bm_george  — British Male · george", "bm_george"),
    ("bm_lewis  — British Male · lewis", "bm_lewis"),
]

TTS_LANGS = [
    ("English US  (en-us)", "en-us"),
    ("English GB  (en-gb)", "en-gb"),
    ("Français    (fr-fr) — expérimental", "fr-fr"),
    ("Spanish     (es)", "es"),
    ("Japanese    (ja)", "ja"),
]

EL_MODELS = [
    ("eleven_flash_v2_5  — ⚡ ultra-rapide · ~$0.11/1K chars", "eleven_flash_v2_5"),
    ("eleven_turbo_v2_5  — 🔵 rapide · haute qualité", "eleven_turbo_v2_5"),
    ("eleven_multilingual_v2  — ⭐ multilingue · recommandé", "eleven_multilingual_v2"),
    ("eleven_monolingual_v1  — English only · original", "eleven_monolingual_v1"),
]

OPENAI_VOICES = [
    ("cedar  — chaleureux · recommandé", "cedar"),
    ("alloy", "alloy"), ("ash", "ash"), ("ballad", "ballad"),
    ("coral", "coral"), ("echo", "echo"), ("marin", "marin"),
    ("sage", "sage"), ("shimmer", "shimmer"), ("verse", "verse"),
]

GEMINI_VOICES = [
    ("Kore  — recommandé", "Kore"),
    ("Aoede", "Aoede"), ("Charon", "Charon"), ("Fenrir", "Fenrir"),
    ("Leda", "Leda"), ("Orus", "Orus"), ("Puck", "Puck"), ("Zephyr", "Zephyr"),
]

# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_env() -> dict:
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH, override=True)
    return {
        "backend": os.getenv("BACKEND_PROVIDER", "local"),
        "stt": os.getenv("LOCAL_STT_MODEL", "medium"),
        "llm": os.getenv("LOCAL_LLM_MODEL", "llama3.2:3b"),
        "tts_provider": os.getenv("LOCAL_TTS_PROVIDER", "kokoro"),
        "tts_voice": os.getenv("LOCAL_TTS_VOICE", "af_heart"),
        "tts_lang": os.getenv("LOCAL_TTS_LANG", "en-us"),
        "el_key": os.getenv("ELEVENLABS_API_KEY", ""),
        "el_voice": os.getenv("ELEVENLABS_VOICE", "Rachel"),
        "el_model": os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2"),
        "openai_key": os.getenv("OPENAI_API_KEY", ""),
        "openai_voice": os.getenv("OPENAI_VOICE_PREFERENCE", "cedar"),
        "gemini_key": os.getenv("GEMINI_API_KEY", ""),
        "gemini_voice": os.getenv("GEMINI_VOICE_PREFERENCE", "Kore"),
        "use_vision": os.getenv("USE_LOCAL_VISION", "0") == "1",
        "vlm_model": os.getenv("LOCAL_VISION_MODEL", "HuggingFaceTB/SmolVLM2-2.2B-Instruct"),
    }


def _save(key: str, value: str) -> None:
    if not ENV_PATH.exists():
        ENV_PATH.write_text("")
    set_key(str(ENV_PATH), key, value)


def _get_installed_tags() -> set[str]:
    """Return the set of installed Ollama model tags."""
    try:
        r = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        tags = set()
        for line in r.stdout.strip().splitlines()[1:]:
            parts = line.split()
            if parts:
                tags.add(parts[0])
        return tags
    except Exception:
        return set()


def _get_llm_dropdown_options() -> tuple[list[tuple[str, str]], str]:
    """Return (choices, default_value) for the LLM dropdown."""
    installed = _get_installed_tags()
    cfg = _load_env()
    current_llm = cfg["llm"]

    options: list[tuple[str, str]] = []

    # Installed models first (catalog + any extra)
    catalog_tags = {tag for _, tag, _, _ in OLLAMA_CATALOG}
    catalog_map = {tag: (name, size, desc) for name, tag, size, desc in OLLAMA_CATALOG}

    for tag in sorted(installed):
        if tag in catalog_map:
            name, size, desc = catalog_map[tag]
            options.append((f"✅  {name}  ({size})", tag))
        else:
            options.append((f"✅  {tag}  — installé", tag))

    # Catalog models not yet installed
    for name, tag, size, desc in OLLAMA_CATALOG:
        if tag not in installed:
            options.append((f"↓  {name}  ({size})  — à télécharger", tag))

    if not options:
        options = [(f"↓  {name}  ({size})", tag) for name, tag, size, _ in OLLAMA_CATALOG]

    installed_tags = {v for _, v in options if v in installed}
    default = current_llm if current_llm in {v for _, v in options} else (
        next((v for _, v in options if v in installed), options[0][1])
    )
    return options, default


def _get_catalog_choices() -> list[tuple[str, str]]:
    """Return catalog choices annotated with install status."""
    installed = _get_installed_tags()
    choices = []
    for name, tag, size, desc in OLLAMA_CATALOG:
        status = "✅ installé" if tag in installed else f"{size}"
        choices.append((f"{name}  ·  {status}", tag))
    return choices


def _get_installed_table() -> list[list[str]]:
    """Return rows for the installed models table."""
    installed = _get_installed_tags()
    catalog_map = {tag: (name, size, desc) for name, tag, size, desc in OLLAMA_CATALOG}
    rows = []
    for tag in sorted(installed):
        if tag in catalog_map:
            name, size, desc = catalog_map[tag]
            rows.append([name, tag, size, desc])
        else:
            rows.append([tag, tag, "—", "Modèle personnalisé"])
    return rows or [["Aucun modèle installé", "", "", ""]]


def _get_el_voices(api_key: str) -> list[tuple[str, str]]:
    try:
        from elevenlabs.client import ElevenLabs  # type: ignore[import]
        client = ElevenLabs(api_key=api_key.strip())
        resp = client.voices.get_all()
        return [(v.name, v.name) for v in resp.voices]
    except Exception:
        return [("Rachel (par défaut)", "Rachel")]


def _is_ollama_running() -> bool:
    try:
        r = subprocess.run(["ollama", "list"], capture_output=True, timeout=3)
        return r.returncode == 0
    except Exception:
        return False


def _get_app_status() -> tuple[str, bool]:
    global _app_proc
    with _app_lock:
        if _app_proc is None or _app_proc.poll() is not None:
            return "⏹️  Arrêtée", False
        return f"✅  En cours — http://localhost:{APP_PORT}", True


def _tail_logs(n: int = 60) -> str:
    if not LOG_PATH.exists():
        return "Aucun log pour l'instant."
    try:
        lines = LOG_PATH.read_text(errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return ""


# ── Callbacks ─────────────────────────────────────────────────────────────────


def on_backend_change(backend: str):
    return (
        gr.update(visible=backend == "local"),
        gr.update(visible=backend == "openai"),
        gr.update(visible=backend == "gemini"),
    )


def on_tts_provider_change(provider: str):
    return (
        gr.update(visible=provider == "kokoro"),
        gr.update(visible=provider == "elevenlabs"),
    )


def on_load_el_voices(api_key: str):
    voices = _get_el_voices(api_key)
    choices = voices if voices else [("Rachel", "Rachel")]
    return gr.update(choices=choices, value=choices[0][1])


def on_refresh_llm():
    """Rescan installed models and update the LLM dropdown."""
    options, default = _get_llm_dropdown_options()
    return gr.update(choices=options, value=default)


def on_catalog_select(tag: str):
    """Show description of the selected catalog model."""
    if not tag:
        return ""
    catalog_map = {t: (n, s, d) for n, t, s, d in OLLAMA_CATALOG}
    installed = _get_installed_tags()
    if tag in catalog_map:
        name, size, desc = catalog_map[tag]
        status = "✅ Déjà installé" if tag in installed else f"Taille : **{size}**"
        return f"**{name}** (`{tag}`)  \n{desc}  \n{status}"
    return f"`{tag}`"


def on_refresh_catalog():
    """Refresh both the catalog dropdown and the installed table."""
    choices = _get_catalog_choices()
    table = _get_installed_table()
    return gr.update(choices=choices), table


def pull_model_stream(tag: str):
    """Stream ollama pull progress line by line."""
    tag = (tag or "").strip()
    if not tag:
        yield "⚠️  Sélectionnez un modèle dans le catalogue."
        return

    installed = _get_installed_tags()
    if tag in installed:
        yield f"✅  `{tag}` est déjà installé — rien à faire."
        return

    yield f"⬇️  Démarrage du téléchargement de `{tag}`…\n"
    try:
        proc = subprocess.Popen(
            ["ollama", "pull", tag],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        lines: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                lines.append(line)
                yield "\n".join(lines[-25:])
        proc.wait()
        if proc.returncode == 0:
            lines.append(f"\n✅  `{tag}` installé avec succès ! Cliquez sur « Actualiser » pour le voir dans la liste.")
        else:
            lines.append(f"\n❌  Erreur lors du téléchargement (code {proc.returncode}).")
        yield "\n".join(lines[-25:])
    except FileNotFoundError:
        yield "❌  `ollama` introuvable. Installez Ollama depuis https://ollama.com"
    except Exception as e:
        yield f"❌  Erreur : {e}"


def save_config(
    backend, stt, llm, tts_provider, tts_voice, tts_lang,
    el_key, el_voice, el_model,
    openai_key, openai_voice,
    gemini_key, gemini_voice,
    use_vision, vlm_model,
) -> str:
    _save("BACKEND_PROVIDER", backend)
    if backend == "local":
        _save("LOCAL_STT_MODEL", stt or "medium")
        _save("LOCAL_LLM_MODEL", llm or "llama3.2:3b")
        _save("USE_LOCAL_VISION", "1" if use_vision else "0")
        if vlm_model:
            _save("LOCAL_VISION_MODEL", vlm_model)
        _save("LOCAL_TTS_PROVIDER", tts_provider or "kokoro")
        if tts_provider == "kokoro":
            _save("LOCAL_TTS_VOICE", tts_voice or "af_heart")
            _save("LOCAL_TTS_LANG", tts_lang or "en-us")
        else:
            if el_key and el_key.strip():
                _save("ELEVENLABS_API_KEY", el_key.strip())
            if el_voice:
                _save("ELEVENLABS_VOICE", el_voice)
            if el_model:
                _save("ELEVENLABS_MODEL", el_model)
    elif backend == "openai":
        if openai_key and openai_key.strip():
            _save("OPENAI_API_KEY", openai_key.strip())
        if openai_voice:
            _save("OPENAI_VOICE_PREFERENCE", openai_voice)
    elif backend == "gemini":
        if gemini_key and gemini_key.strip():
            _save("GEMINI_API_KEY", gemini_key.strip())
        if gemini_voice:
            _save("GEMINI_VOICE_PREFERENCE", gemini_voice)
    return "✅ Configuration sauvegardée — redémarrez l'app pour appliquer."


def start_app() -> tuple[str, str]:
    global _app_proc
    with _app_lock:
        if _app_proc and _app_proc.poll() is None:
            return f"✅  En cours — http://localhost:{APP_PORT}", _tail_logs()

        if not _is_ollama_running():
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            time.sleep(2)

        LOG_PATH.write_text("")
        log_file = open(LOG_PATH, "w")

        env = {
            **os.environ,
            "GRADIO_SERVER_PORT": str(APP_PORT),
            "GRADIO_SERVER_NAME": "0.0.0.0",
        }

        cmd = [sys.executable, "-m", "reachy_mini_conversation_app.main", "--gradio"]
        if os.getenv("USE_LOCAL_VISION", "0") == "1":
            cmd.append("--local-vision")

        _app_proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_file,
            stderr=log_file,
            cwd=str(APP_DIR),
        )

    time.sleep(3)
    status, _ = _get_app_status()
    return status, _tail_logs()


def stop_app() -> tuple[str, str]:
    global _app_proc
    with _app_lock:
        if _app_proc and _app_proc.poll() is None:
            _app_proc.terminate()
            try:
                _app_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _app_proc.kill()
            _app_proc = None
            return "⏹️  Arrêtée", _tail_logs()
    return "⏹️  Déjà arrêtée", _tail_logs()


def refresh_status() -> tuple[str, str]:
    status, _ = _get_app_status()
    return status, _tail_logs()


# ── Usage helpers ─────────────────────────────────────────────────────────────

def _usage_summary_md(days: int) -> str:
    try:
        sys.path.insert(0, str(APP_DIR / "src"))
        from reachy_mini_conversation_app.usage_tracker import get_summary, get_stats
        s = get_summary(days)
        label = {1: "aujourd'hui", 7: "7 derniers jours", 30: "30 derniers jours", 0: "total"}.get(days, f"{days}j")
        header = (
            f"**{label}** — "
            f"**{s['requests']}** requêtes · "
            f"**{s['tokens']:,}** tokens · "
            f"**{s['audio_s']}s** audio · "
            f"**{s['chars']:,}** chars TTS"
        )
        rows = get_stats(days)
        if not rows:
            return header + "\n\n_Aucune donnée pour cette période._"
        lines = [header, "", "| Type | Modèle | Requêtes | Tokens in | Tokens out | Audio in | Audio out | Chars TTS | Latence moy |",
                 "|------|--------|----------|-----------|------------|----------|-----------|-----------|-------------|"]
        for r in rows:
            tok_in  = f"{r['tokens_in']:,}"  if r["tokens_in"]  else "—"
            tok_out = f"{r['tokens_out']:,}" if r["tokens_out"] else "—"
            aud_in  = f"{r['audio_in_s']}s"  if r["audio_in_s"]  else "—"
            aud_out = f"{r['audio_out_s']}s" if r["audio_out_s"] else "—"
            chars   = f"{r['chars_in']:,}"   if r["chars_in"]   else "—"
            lines.append(
                f"| **{r['type']}** | `{r['model']}` | {r['requests']} "
                f"| {tok_in} | {tok_out} | {aud_in} | {aud_out} | {chars} | {r['avg_latency_ms']} ms |"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"_Tracker non disponible : {e}_"


def _usage_events_table(n: int = 30) -> list[list]:
    try:
        sys.path.insert(0, str(APP_DIR / "src"))
        from reachy_mini_conversation_app.usage_tracker import get_recent_events
        events = get_recent_events(n)
        rows = []
        for e in events:
            detail = ""
            if e["type"] == "LLM":
                detail = f"{e['tokens_in']} in / {e['tokens_out']} out tokens"
            elif e["type"] == "STT":
                detail = f"{e['audio_in_s']}s audio"
            elif e["type"] == "TTS":
                detail = f"{e['chars_in']} chars → {e['audio_out_s']}s audio"
            elif e["type"] == "VLM":
                detail = f"{e['tokens_in']} in / {e['tokens_out']} out tokens"
            rows.append([e["ts"], e["type"], e["model"], detail, f"{e['latency_ms']} ms", e["source"]])
        return rows or [["—", "—", "—", "—", "—", "—"]]
    except Exception:
        return [["Tracker non disponible", "", "", "", "", ""]]


def on_reset_stats():
    try:
        from reachy_mini_conversation_app.usage_tracker import reset_stats
        reset_stats()
        return "✅ Stats réinitialisées."
    except Exception as e:
        return f"❌ {e}"


# ── App Store helpers ─────────────────────────────────────────────────────────

_APP_LIST_URL = "https://huggingface.co/datasets/pollen-robotics/reachy-mini-official-app-store/resolve/main/app-list.json"
_app_metadata_cache: dict[str, dict] = {}


def _fetch_app_list() -> list[str]:
    try:
        import urllib.request
        with urllib.request.urlopen(_APP_LIST_URL, timeout=8) as r:
            import json as _json
            return _json.loads(r.read())
    except Exception:
        return []


def _fetch_app_meta(space_id: str) -> dict:
    if space_id in _app_metadata_cache:
        return _app_metadata_cache[space_id]
    import urllib.request, re
    name = space_id.split("/")[-1].replace("_", " ").title()
    desc = "Application Reachy Mini"
    try:
        url = f"https://huggingface.co/spaces/{space_id}/resolve/main/pyproject.toml"
        with urllib.request.urlopen(url, timeout=5) as r:
            content = r.read().decode(errors="replace")
        m = re.search(r'description\s*=\s*["\']([^"\']+)["\']', content)
        if m:
            desc = m.group(1)
        m = re.search(r'^\s*name\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
        if m:
            name = m.group(1)
    except Exception:
        pass
    meta = {"name": name, "description": desc, "space_id": space_id,
            "url": f"https://huggingface.co/spaces/{space_id}"}
    _app_metadata_cache[space_id] = meta
    return meta


def _pip_installed(package_name: str) -> bool:
    r = subprocess.run(
        [sys.executable, "-m", "pip", "show", package_name],
        capture_output=True, text=True,
    )
    return r.returncode == 0


def _is_app_installed(space_id: str) -> bool:
    pkg = space_id.split("/")[-1].replace("-", "_").lower()
    return _pip_installed(pkg)


def fetch_official_apps() -> list[list[str]]:
    ids = _fetch_app_list()
    if not ids:
        return [["❌ Impossible de charger la liste", "", "", ""]]
    rows = []
    for sid in ids:
        meta = _fetch_app_meta(sid)
        status = "✅ Installée" if _is_app_installed(sid) else "↓ Disponible"
        rows.append([meta["name"], meta["description"], sid, status])
    return rows


def install_app_stream(space_id: str):
    space_id = (space_id or "").strip()
    if not space_id or "/" not in space_id:
        yield "⚠️  Entrez un identifiant HuggingFace valide (ex: pollen-robotics/reachy_mini_radio)"
        return
    yield f"⬇️  Installation de `{space_id}` depuis HuggingFace…\n"
    install_url = f"git+https://huggingface.co/spaces/{space_id}"
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "pip", "install", "--upgrade", install_url],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        lines: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                lines.append(line)
                yield "\n".join(lines[-30:])
        proc.wait()
        if proc.returncode == 0:
            lines.append(f"\n✅ `{space_id}` installée avec succès !")
            _app_metadata_cache.pop(space_id, None)
        else:
            lines.append(f"\n❌ Erreur (code {proc.returncode}).")
        yield "\n".join(lines[-30:])
    except Exception as e:
        yield f"❌ Erreur : {e}"


# ── Build UI ──────────────────────────────────────────────────────────────────

cfg = _load_env()
_llm_options, _llm_default = _get_llm_dropdown_options()

CSS = """
.gr-button-primary { font-size: 1.05em !important; min-height: 44px !important; }
.status-box textarea { font-family: monospace; font-size: 0.85em; }
.log-box textarea { font-family: monospace; font-size: 0.78em; background: #1e1e1e; color: #d4d4d4; }
.pull-log textarea { font-family: monospace; font-size: 0.82em; background: #0d1117; color: #58a6ff; min-height: 180px; }
.install-log textarea { font-family: monospace; font-size: 0.82em; background: #0d1117; color: #3fb950; min-height: 180px; }
footer { display: none !important; }
"""

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Sam Reachy mini custom",
    css=CSS,
) as app:

    gr.Markdown("# 🤖 Sam Reachy mini custom")

    with gr.Tabs():

        # ── TAB 1 : Configuration ─────────────────────────────────────────────
        with gr.Tab("⚙️  Configuration"):

            gr.Markdown("Configurez les modèles puis cliquez **Sauvegarder**. Redémarrez l'app pour appliquer.")

            backend_radio = gr.Radio(
                choices=BACKEND_OPTIONS,
                value=cfg["backend"],
                label="Mode / Backend",
                info="Local = gratuit, hors-ligne. Payant = meilleure qualité, nécessite une clé API.",
            )

            # ── Local ──────────────────────────────────────────────────────────
            with gr.Column(visible=(cfg["backend"] == "local")) as local_col:

                gr.Markdown("### 🟢 Modèles locaux")

                stt_drop = gr.Dropdown(
                    choices=STT_OPTIONS,
                    value=cfg["stt"],
                    label="🎙️  STT — Reconnaissance vocale  (faster-whisper)",
                    info="medium = meilleur compromis vitesse/qualité.",
                )

                with gr.Row():
                    llm_drop = gr.Dropdown(
                        choices=_llm_options,
                        value=_llm_default,
                        label="🧠  LLM — Modèle de langage  (Ollama)",
                        info="✅ installé · ↓ à télécharger via l'onglet Modèles Ollama.",
                        scale=5,
                    )
                    refresh_llm_btn = gr.Button("🔄", scale=1, min_width=60)

                gr.Markdown("### 👁️ Vision — Caméra")

                vision_check = gr.Checkbox(
                    value=cfg["use_vision"],
                    label="Activer la vision (caméra de Reachy)",
                    info="Le modèle vision se télécharge automatiquement depuis HuggingFace au premier démarrage.",
                )
                vlm_drop = gr.Dropdown(
                    choices=VLM_OPTIONS,
                    value=cfg["vlm_model"] if cfg["vlm_model"] in [v for _, v in VLM_OPTIONS] else VLM_OPTIONS[1][1],
                    label="Modèle de vision",
                    visible=cfg["use_vision"],
                )
                vision_check.change(lambda v: gr.update(visible=v), inputs=[vision_check], outputs=[vlm_drop])

                gr.Markdown("### 🔊 TTS — Synthèse vocale")

                tts_provider_radio = gr.Radio(
                    choices=TTS_PROVIDER_OPTIONS,
                    value=cfg["tts_provider"],
                    label="Fournisseur TTS",
                )

                with gr.Column(visible=(cfg["tts_provider"] == "kokoro")) as kokoro_col:
                    with gr.Row():
                        tts_voice_drop = gr.Dropdown(
                            choices=KOKORO_VOICES,
                            value=cfg["tts_voice"],
                            label="Voix Kokoro",
                            scale=2,
                        )
                        tts_lang_drop = gr.Dropdown(
                            choices=TTS_LANGS,
                            value=cfg["tts_lang"],
                            label="Langue",
                            scale=1,
                        )

                with gr.Column(visible=(cfg["tts_provider"] == "elevenlabs")) as el_col:
                    el_key_box = gr.Textbox(
                        value=cfg["el_key"],
                        label="ElevenLabs API Key",
                        type="password",
                        placeholder="sk_...",
                    )
                    el_model_drop = gr.Dropdown(
                        choices=EL_MODELS,
                        value=cfg["el_model"],
                        label="Modèle ElevenLabs",
                    )
                    with gr.Row():
                        el_voice_drop = gr.Dropdown(
                            choices=[("Rachel  (par défaut)", "Rachel")],
                            value=cfg["el_voice"] or "Rachel",
                            label="Voix ElevenLabs",
                            scale=3,
                            info="Cliquez 'Charger mes voix' après avoir entré votre clé.",
                        )
                        load_voices_btn = gr.Button("🔄  Charger mes voix", scale=1)
                    load_voices_btn.click(on_load_el_voices, inputs=[el_key_box], outputs=[el_voice_drop])

            # ── OpenAI ─────────────────────────────────────────────────────────
            with gr.Column(visible=(cfg["backend"] == "openai")) as openai_col:
                gr.Markdown(
                    "### 💳 OpenAI Realtime\n"
                    "**STT** gpt-4o-transcribe · **LLM** gpt-4o-realtime · **TTS** intégré  \n"
                    "💰 ~0,06 $/min d'audio"
                )
                openai_key_box = gr.Textbox(value=cfg["openai_key"], label="OpenAI API Key", type="password", placeholder="sk-...")
                openai_voice_drop = gr.Dropdown(
                    choices=OPENAI_VOICES,
                    value=cfg["openai_voice"] if cfg["openai_voice"] in [v for _, v in OPENAI_VOICES] else "cedar",
                    label="Voix",
                )

            # ── Gemini ─────────────────────────────────────────────────────────
            with gr.Column(visible=(cfg["backend"] == "gemini")) as gemini_col:
                gr.Markdown(
                    "### 🟡 Gemini Live\n"
                    "**STT + LLM + TTS** intégrés · Gratuit jusqu'à 2M tokens/mois  \n"
                    "🔑 Clé sur [aistudio.google.com](https://aistudio.google.com/apikey)"
                )
                gemini_key_box = gr.Textbox(value=cfg["gemini_key"], label="Gemini API Key", type="password", placeholder="AIza...")
                gemini_voice_drop = gr.Dropdown(
                    choices=GEMINI_VOICES,
                    value=cfg["gemini_voice"] if cfg["gemini_voice"] in [v for _, v in GEMINI_VOICES] else "Kore",
                    label="Voix",
                )

            gr.Markdown("---")
            save_btn = gr.Button("💾  Sauvegarder la configuration", variant="primary")
            save_status = gr.Textbox(label="Statut", interactive=False)

            backend_radio.change(on_backend_change, [backend_radio], [local_col, openai_col, gemini_col])
            tts_provider_radio.change(on_tts_provider_change, [tts_provider_radio], [kokoro_col, el_col])
            refresh_llm_btn.click(on_refresh_llm, outputs=[llm_drop])
            save_btn.click(
                save_config,
                inputs=[
                    backend_radio, stt_drop, llm_drop,
                    tts_provider_radio, tts_voice_drop, tts_lang_drop,
                    el_key_box, el_voice_drop, el_model_drop,
                    openai_key_box, openai_voice_drop,
                    gemini_key_box, gemini_voice_drop,
                    vision_check, vlm_drop,
                ],
                outputs=[save_status],
            )

        # ── TAB 2 : Modèles Ollama ────────────────────────────────────────────
        with gr.Tab("📦  Modèles Ollama"):

            gr.Markdown(
                "Gérez vos modèles de langage locaux. "
                "Les modèles installés apparaissent automatiquement dans la Configuration."
            )

            # Modèles installés
            with gr.Group():
                with gr.Row():
                    gr.Markdown("### ✅ Modèles installés")
                    refresh_catalog_btn = gr.Button("🔄  Actualiser", scale=0, min_width=120)

                installed_table = gr.Dataframe(
                    value=_get_installed_table(),
                    headers=["Nom", "Tag Ollama", "Taille", "Description"],
                    datatype=["str", "str", "str", "str"],
                    interactive=False,
                    wrap=True,
                )

            gr.Markdown("---")

            # Catalogue
            gr.Markdown("### 📚 Catalogue de modèles")
            gr.Markdown(
                "Sélectionnez un modèle pour voir sa description, puis cliquez **Télécharger**. "
                "Les modèles ✅ sont déjà installés."
            )

            catalog_drop = gr.Dropdown(
                choices=_get_catalog_choices(),
                value=None,
                label="Choisir un modèle",
                info="Classés par taille — les plus légers en premier.",
            )

            model_desc = gr.Markdown("_Sélectionnez un modèle pour voir sa description._")

            download_btn = gr.Button("⬇️  Télécharger ce modèle", variant="primary")

            pull_log = gr.Textbox(
                label="Progression du téléchargement",
                interactive=False,
                lines=10,
                max_lines=12,
                elem_classes=["pull-log"],
                placeholder="La progression s'affichera ici…",
            )

            gr.Markdown(
                "_Après installation, cliquez sur 🔄 Actualiser ci-dessus "
                "et sur le bouton 🔄 dans l'onglet Configuration pour mettre à jour les listes._"
            )

            catalog_drop.change(on_catalog_select, inputs=[catalog_drop], outputs=[model_desc])
            download_btn.click(pull_model_stream, inputs=[catalog_drop], outputs=[pull_log])
            refresh_catalog_btn.click(on_refresh_catalog, outputs=[catalog_drop, installed_table])

        # ── TAB 3 : Consommation ──────────────────────────────────────────────
        with gr.Tab("📊  Consommation"):

            gr.Markdown("Suivi de la consommation de chaque modèle (STT, LLM, TTS, VLM).")

            with gr.Row():
                period_radio = gr.Radio(
                    choices=[(  "Aujourd'hui", 1), ("7 jours", 7), ("30 jours", 30), ("Tout", 0)],
                    value=7, label="Période", scale=3,
                )
                refresh_usage_btn = gr.Button("🔄  Actualiser", scale=1, min_width=120)
                reset_usage_btn   = gr.Button("🗑️  Réinitialiser", scale=1, min_width=140, variant="stop")

            usage_summary = gr.Markdown(_usage_summary_md(7))
            reset_status  = gr.Textbox(label="", interactive=False, visible=True, max_lines=1)

            gr.Markdown("#### Événements récents")
            usage_table = gr.Dataframe(
                value=_usage_events_table(30),
                headers=["Horodatage", "Type", "Modèle", "Détail", "Latence", "Source"],
                datatype=["str","str","str","str","str","str"],
                interactive=False, wrap=True,
            )

            def _refresh_usage(days):
                return _usage_summary_md(int(days)), _usage_events_table(30)

            period_radio.change(_refresh_usage, [period_radio], [usage_summary, usage_table])
            refresh_usage_btn.click(_refresh_usage, [period_radio], [usage_summary, usage_table])
            reset_usage_btn.click(on_reset_stats, outputs=[reset_status])

            usage_timer = gr.Timer(10)
            usage_timer.tick(_refresh_usage, inputs=[period_radio], outputs=[usage_summary, usage_table])

        # ── TAB 4 : App Store ─────────────────────────────────────────────────
        with gr.Tab("🛒  App Store"):

            gr.Markdown(
                "Installez des applications Reachy Mini depuis HuggingFace.  \n"
                "Les apps sont injectées avec `OPENAI_BASE_URL` pointant vers Ollama local — "
                "elles utilisent vos modèles locaux quand c'est possible."
            )

            with gr.Row():
                load_apps_btn = gr.Button("🔄  Charger la liste officielle", variant="primary")

            apps_table = gr.Dataframe(
                value=[["Cliquez sur « Charger » pour voir les apps disponibles", "", "", ""]],
                headers=["Nom", "Description", "Space HuggingFace", "Statut"],
                datatype=["str","str","str","str"],
                interactive=False, wrap=True,
            )

            gr.Markdown("---")
            gr.Markdown("### ⬇️  Installer une app")
            gr.Markdown(
                "Copiez l'identifiant HuggingFace d'un Space (ex: `pollen-robotics/reachy_mini_radio`) "
                "et cliquez **Installer**."
            )

            with gr.Row():
                app_input = gr.Textbox(
                    label="Identifiant HuggingFace (org/repo)",
                    placeholder="pollen-robotics/reachy_mini_radio",
                    scale=4,
                )
                install_btn = gr.Button("⬇️  Installer", variant="primary", scale=1)

            install_log = gr.Textbox(
                label="Progression",
                interactive=False, lines=10, max_lines=12,
                elem_classes=["install-log"],
                placeholder="La progression s'affichera ici…",
            )

            gr.Markdown(
                "ℹ️  **Compatibilité locale** — Au lancement, les apps reçoivent automatiquement :  \n"
                "`OPENAI_BASE_URL=http://localhost:11434/v1` · `OPENAI_API_KEY=ollama`  \n"
                "Les apps qui utilisent le SDK OpenAI standard appelleront donc Ollama en local."
            )

            load_apps_btn.click(fetch_official_apps, outputs=[apps_table])
            install_btn.click(install_app_stream, inputs=[app_input], outputs=[install_log])

            # Pré-remplir l'input quand on clique sur une ligne du tableau
            def on_app_select(evt: gr.SelectData):
                try:
                    return evt.row_value[2]  # colonne "Space HuggingFace"
                except Exception:
                    return ""

            apps_table.select(on_app_select, outputs=[app_input])

        # ── TAB 5 : Lancement ─────────────────────────────────────────────────
        with gr.Tab("🚀  Lancement"):

            gr.Markdown(
                "Démarrez l'application Reachy Mini.  \n"
                f"Elle sera disponible sur **http://localhost:{APP_PORT}**"
            )

            status_box = gr.Textbox(
                value="⏹️  Arrêtée",
                label="Statut",
                interactive=False,
                elem_classes=["status-box"],
            )

            with gr.Row():
                start_btn = gr.Button("🚀  Démarrer", variant="primary", scale=2)
                stop_btn = gr.Button("⏹  Arrêter", variant="stop", scale=1)
                refresh_btn = gr.Button("🔄  Actualiser", scale=1)

            open_app_btn = gr.Button(
                f"↗  Ouvrir l'application  (http://localhost:{APP_PORT})",
                link=f"http://localhost:{APP_PORT}",
                variant="secondary",
            )

            log_box = gr.Textbox(
                value=_tail_logs(),
                label="Logs",
                interactive=False,
                lines=18,
                max_lines=18,
                elem_classes=["log-box"],
            )

            timer = gr.Timer(3)
            timer.tick(refresh_status, outputs=[status_box, log_box])

            start_btn.click(start_app, outputs=[status_box, log_box])
            stop_btn.click(stop_app, outputs=[status_box, log_box])
            refresh_btn.click(refresh_status, outputs=[status_box, log_box])


if __name__ == "__main__":
    print(f"\n🤖  Sam Reachy mini custom — http://localhost:{DASHBOARD_PORT}\n")
    app.launch(
        server_port=DASHBOARD_PORT,
        server_name="0.0.0.0",
        inbrowser=False,
        share=False,
        show_api=False,
    )
