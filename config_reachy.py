#!/usr/bin/env python3
"""Configuration UI for Reachy Mini.

Usage:
    python config_reachy.py
    # or via alias: reachy-config
"""

import os
import subprocess
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv, set_key


ENV_PATH = Path(__file__).parent / ".env"

# ── Model catalogues ──────────────────────────────────────────────────────────

BACKEND_OPTIONS = [
    ("🟢  Local — gratuit, 100% local  (faster-whisper + Ollama + Kokoro TTS)", "local"),
    ("💳  OpenAI Realtime — payant  (gpt-4o · STT + LLM + TTS intégrés)", "openai"),
    ("🟡  Gemini Live — gratuit limité  (gemini-3.1-flash-live-preview)", "gemini"),
]

STT_OPTIONS = [
    ("tiny  — ⚡ ultra-rapide · qualité basique  [gratuit]", "tiny"),
    ("base  — ⚡ très rapide · qualité correcte  [gratuit]", "base"),
    ("small  — 🔵 rapide · bonne qualité  [gratuit]", "small"),
    ("medium  — ⭐ recommandé · très bonne qualité  [gratuit]", "medium"),
    ("large-v3  — 🏆 meilleure qualité · plus lent  [gratuit]", "large-v3"),
]

RECOMMENDED_LLM = [
    ("llama3.2:1b  — ⚡ ultra-rapide · qualité basique  [gratuit]", "llama3.2:1b"),
    ("llama3.2:3b  — ⭐ recommandé · bon équilibre  [gratuit]", "llama3.2:3b"),
    ("llama3.1:8b  — 🔵 haute qualité · lent  [gratuit]", "llama3.1:8b"),
    ("qwen2.5:3b  — 🌍 multilingue · rapide  [gratuit]", "qwen2.5:3b"),
    ("qwen2.5:7b  — 🌍 multilingue · haute qualité  [gratuit]", "qwen2.5:7b"),
    ("mistral:7b  — 🔵 haute qualité  [gratuit]", "mistral:7b"),
    ("phi4-mini:3.8b  — 🔵 compact · performant  [gratuit]", "phi4-mini:3.8b"),
    ("gemma3:4b  — 🔵 Google · multilingue  [gratuit]", "gemma3:4b"),
]

KOKORO_VOICES = [
    ("af_heart  — American Female · chaleureuse  ⭐ [gratuit]", "af_heart"),
    ("af_bella  — American Female · bella  [gratuit]", "af_bella"),
    ("af_sarah  — American Female · sarah  [gratuit]", "af_sarah"),
    ("af_sky  — American Female · sky  [gratuit]", "af_sky"),
    ("am_adam  — American Male · adam  [gratuit]", "am_adam"),
    ("am_michael  — American Male · michael  [gratuit]", "am_michael"),
    ("bf_emma  — British Female · emma  [gratuit]", "bf_emma"),
    ("bf_isabella  — British Female · isabella  [gratuit]", "bf_isabella"),
    ("bm_george  — British Male · george  [gratuit]", "bm_george"),
    ("bm_lewis  — British Male · lewis  [gratuit]", "bm_lewis"),
]

TTS_LANGS = [
    ("English US  (en-us)", "en-us"),
    ("English GB  (en-gb)", "en-gb"),
    ("Français    (fr-fr) — expérimental", "fr-fr"),
    ("Spanish     (es)", "es"),
    ("Japanese    (ja)", "ja"),
    ("Chinese     (zh-a)", "zh-a"),
]

OPENAI_VOICES = [
    ("cedar  — chaleureux · recommandé  [payant]", "cedar"),
    ("alloy  [payant]", "alloy"),
    ("ash  [payant]", "ash"),
    ("ballad  [payant]", "ballad"),
    ("coral  [payant]", "coral"),
    ("echo  [payant]", "echo"),
    ("marin  [payant]", "marin"),
    ("sage  [payant]", "sage"),
    ("shimmer  [payant]", "shimmer"),
    ("verse  [payant]", "verse"),
]

GEMINI_VOICES = [
    ("Kore  — recommandé  [gratuit limité]", "Kore"),
    ("Aoede  [gratuit limité]", "Aoede"),
    ("Charon  [gratuit limité]", "Charon"),
    ("Fenrir  [gratuit limité]", "Fenrir"),
    ("Leda  [gratuit limité]", "Leda"),
    ("Orus  [gratuit limité]", "Orus"),
    ("Puck  [gratuit limité]", "Puck"),
    ("Zephyr  [gratuit limité]", "Zephyr"),
]

# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_installed_ollama_models() -> list[str]:
    try:
        r = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        names = []
        for line in r.stdout.strip().splitlines()[1:]:
            if line.strip():
                names.append(line.split()[0])
        return names
    except Exception:
        return []


def _build_llm_options() -> list[tuple[str, str]]:
    installed = set(_get_installed_ollama_models())
    rec_values = {v for _, v in RECOMMENDED_LLM}

    options: list[tuple[str, str]] = []

    # Installed and recommended — show at top with ✅
    for label, val in RECOMMENDED_LLM:
        if val in installed:
            options.append((label.replace("[gratuit]", "✅ installé  [gratuit]"), val))

    # Installed but not in recommended list
    for name in sorted(installed - rec_values):
        options.append((f"{name}  — ✅ installé  [gratuit]", name))

    # Recommended but not installed
    for label, val in RECOMMENDED_LLM:
        if val not in installed:
            options.append((label.replace("[gratuit]", "↓ à télécharger  [gratuit]"), val))

    return options if options else RECOMMENDED_LLM


def _load_config() -> dict:
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH, override=True)
    return {
        "backend": os.getenv("BACKEND_PROVIDER", "local"),
        "stt": os.getenv("LOCAL_STT_MODEL", "medium"),
        "llm": os.getenv("LOCAL_LLM_MODEL", "llama3.2:3b"),
        "tts_voice": os.getenv("LOCAL_TTS_VOICE", "af_heart"),
        "tts_lang": os.getenv("LOCAL_TTS_LANG", "en-us"),
        "openai_key": os.getenv("OPENAI_API_KEY", ""),
        "openai_voice": os.getenv("OPENAI_VOICE_PREFERENCE", "cedar"),
        "gemini_key": os.getenv("GEMINI_API_KEY", ""),
        "gemini_voice": os.getenv("GEMINI_VOICE_PREFERENCE", "Kore"),
    }


def _save_env(key: str, value: str) -> None:
    if not ENV_PATH.exists():
        ENV_PATH.write_text("")
    set_key(str(ENV_PATH), key, value)


# ── Gradio callbacks ──────────────────────────────────────────────────────────


def on_backend_change(backend: str):
    return (
        gr.update(visible=backend == "local"),
        gr.update(visible=backend == "openai"),
        gr.update(visible=backend == "gemini"),
    )


def save_config(
    backend, stt, llm, tts_voice, tts_lang,
    openai_key, openai_voice,
    gemini_key, gemini_voice,
) -> str:
    _save_env("BACKEND_PROVIDER", backend)

    if backend == "local":
        _save_env("LOCAL_STT_MODEL", stt or "medium")
        _save_env("LOCAL_LLM_MODEL", llm or "llama3.2:3b")
        _save_env("LOCAL_TTS_VOICE", tts_voice or "af_heart")
        _save_env("LOCAL_TTS_LANG", tts_lang or "en-us")

    elif backend == "openai":
        if openai_key and openai_key.strip():
            _save_env("OPENAI_API_KEY", openai_key.strip())
        if openai_voice:
            _save_env("OPENAI_VOICE_PREFERENCE", openai_voice)

    elif backend == "gemini":
        if gemini_key and gemini_key.strip():
            _save_env("GEMINI_API_KEY", gemini_key.strip())
        if gemini_voice:
            _save_env("GEMINI_VOICE_PREFERENCE", gemini_voice)

    return f"✅ Configuration sauvegardée dans {ENV_PATH}\n\nLancez l'application avec :  reachy-start"


def pull_model(model_name: str) -> str:
    name = (model_name or "").strip()
    if not name:
        return "⚠️ Entrez un nom de modèle  (ex : llama3.1:8b)"
    try:
        r = subprocess.run(
            ["ollama", "pull", name],
            capture_output=True, text=True, timeout=600,
        )
        if r.returncode == 0:
            return f"✅ {name} téléchargé. Fermez et relancez reachy-config pour le voir dans la liste."
        return f"❌ Erreur :\n{(r.stderr or r.stdout)[-400:]}"
    except subprocess.TimeoutExpired:
        return "⏳ Toujours en cours (10 min dépassées). Vérifiez avec `ollama list` dans votre terminal."
    except FileNotFoundError:
        return "❌ Ollama introuvable. Vérifiez qu'il est installé."
    except Exception as e:
        return f"❌ {e}"


# ── Build UI ──────────────────────────────────────────────────────────────────

cfg = _load_config()
llm_options = _build_llm_options()
init_backend = cfg["backend"]

CSS = """
.main-title { font-size: 1.4em; font-weight: 700; margin-bottom: 0.2em; }
.section-title { font-size: 1.1em; font-weight: 600; margin-top: 0.5em; border-bottom: 1px solid #e0e0e0; padding-bottom: 4px; }
.save-btn { min-height: 52px !important; font-size: 1.05em !important; }
"""

with gr.Blocks(theme=gr.themes.Soft(), title="Reachy Mini — Configuration", css=CSS) as app:

    gr.Markdown(
        "# 🤖 Reachy Mini — Configuration des modèles\n"
        "Choisissez vos modèles, sauvegardez, puis lancez **`reachy-start`** dans votre terminal."
    )

    backend_radio = gr.Radio(
        choices=BACKEND_OPTIONS,
        value=init_backend,
        label="🖥️  Mode / Backend",
        info="Local = gratuit, tout fonctionne hors-ligne. Payant = meilleure qualité, nécessite une clé API.",
    )

    # ── Local section ─────────────────────────────────────────────────────
    with gr.Column(visible=(init_backend == "local")) as local_col:
        gr.Markdown("### 🟢 Configuration locale")

        stt_drop = gr.Dropdown(
            choices=STT_OPTIONS,
            value=cfg["stt"],
            label="🎙️  STT — Reconnaissance vocale  (faster-whisper)",
            info="medium = meilleur compromis. large-v3 = meilleure qualité mais plus lent.",
        )

        llm_drop = gr.Dropdown(
            choices=llm_options,
            value=cfg["llm"] if any(v == cfg["llm"] for _, v in llm_options) else (llm_options[0][1] if llm_options else "llama3.2:3b"),
            label="🧠  LLM — Modèle de langage  (Ollama)",
            info="✅ = déjà installé · ↓ = à télécharger via le panneau ci-dessous.",
        )

        with gr.Row():
            tts_voice_drop = gr.Dropdown(
                choices=KOKORO_VOICES,
                value=cfg["tts_voice"],
                label="🔊  Voix TTS  (Kokoro-ONNX)",
                scale=2,
            )
            tts_lang_drop = gr.Dropdown(
                choices=TTS_LANGS,
                value=cfg["tts_lang"],
                label="Langue",
                scale=1,
            )

        with gr.Accordion("📥  Télécharger un modèle Ollama", open=False):
            gr.Markdown(
                "Entrez un nom de modèle Ollama et cliquez Télécharger.  \n"
                "Exemples : `gemma3:4b`, `llama3.1:8b`, `deepseek-r1:7b`, `phi4:14b`  \n"
                "Parcourez tous les modèles sur [ollama.com/library](https://ollama.com/library)."
            )
            with gr.Row():
                pull_input = gr.Textbox(
                    label="Nom du modèle",
                    placeholder="llama3.1:8b",
                    scale=3,
                )
                pull_btn = gr.Button("↓  Télécharger", scale=1, variant="secondary")
            pull_status = gr.Textbox(label="Statut du téléchargement", interactive=False, lines=2)
            pull_btn.click(pull_model, inputs=[pull_input], outputs=[pull_status])

    # ── OpenAI section ────────────────────────────────────────────────────
    with gr.Column(visible=(init_backend == "openai")) as openai_col:
        gr.Markdown(
            "### 💳 OpenAI Realtime\n"
            "**STT** : gpt-4o-transcribe  ·  **LLM** : gpt-4o-realtime  ·  **TTS** : OpenAI intégré  \n"
            "💰 Tarif estimé : ~0,06 $/min d'audio"
        )
        openai_key_box = gr.Textbox(
            value=cfg["openai_key"],
            label="OpenAI API Key",
            type="password",
            placeholder="sk-...",
        )
        openai_voice_drop = gr.Dropdown(
            choices=OPENAI_VOICES,
            value=cfg["openai_voice"] if cfg["openai_voice"] in [v for _, v in OPENAI_VOICES] else "cedar",
            label="🔊  Voix",
        )

    # ── Gemini section ────────────────────────────────────────────────────
    with gr.Column(visible=(init_backend == "gemini")) as gemini_col:
        gr.Markdown(
            "### 🟡 Gemini Live\n"
            "**STT + LLM + TTS** : intégrés dans Gemini Live  \n"
            "🎁 Gratuit jusqu'à 2M tokens/mois · Clé sur [aistudio.google.com](https://aistudio.google.com/apikey)"
        )
        gemini_key_box = gr.Textbox(
            value=cfg["gemini_key"],
            label="Gemini API Key",
            type="password",
            placeholder="AIza...",
        )
        gemini_voice_drop = gr.Dropdown(
            choices=GEMINI_VOICES,
            value=cfg["gemini_voice"] if cfg["gemini_voice"] in [v for _, v in GEMINI_VOICES] else "Kore",
            label="🔊  Voix",
        )

    # ── Save ──────────────────────────────────────────────────────────────
    gr.Markdown("---")
    save_btn = gr.Button("💾  Sauvegarder la configuration", variant="primary", elem_classes=["save-btn"])
    status_box = gr.Textbox(label="Statut", interactive=False, lines=2)

    # ── Wiring ────────────────────────────────────────────────────────────
    backend_radio.change(
        fn=on_backend_change,
        inputs=[backend_radio],
        outputs=[local_col, openai_col, gemini_col],
    )

    save_btn.click(
        fn=save_config,
        inputs=[
            backend_radio, stt_drop, llm_drop, tts_voice_drop, tts_lang_drop,
            openai_key_box, openai_voice_drop,
            gemini_key_box, gemini_voice_drop,
        ],
        outputs=[status_box],
    )


if __name__ == "__main__":
    print("🤖  Reachy Mini Config — http://localhost:7861")
    app.launch(server_port=7861, inbrowser=True, share=False, show_api=False)
