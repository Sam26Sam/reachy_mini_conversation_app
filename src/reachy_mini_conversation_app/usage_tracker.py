"""Persistent usage tracker — logs STT / LLM / TTS / VLM consumption to SQLite.

Database: ~/.reachy/usage.db
Thread-safe; all public functions can be called from any thread or coroutine.
"""

import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

_DB_DIR = Path.home() / ".reachy"
_DB_PATH = _DB_DIR / "usage.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,
    model_type  TEXT    NOT NULL,
    model_name  TEXT    NOT NULL,
    tokens_in   INTEGER DEFAULT 0,
    tokens_out  INTEGER DEFAULT 0,
    audio_in_s  REAL    DEFAULT 0,
    audio_out_s REAL    DEFAULT 0,
    chars_in    INTEGER DEFAULT 0,
    latency_ms  INTEGER DEFAULT 0,
    source      TEXT    DEFAULT 'conversation'
)
"""

_conn: Optional[sqlite3.Connection] = None
_lock = threading.Lock()


def _db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _DB_DIR.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        _conn.execute(_SCHEMA)
        _conn.commit()
    return _conn


def _insert(model_type: str, model_name: str, **kw) -> None:
    with _lock:
        try:
            _db().execute(
                "INSERT INTO usage_events "
                "(ts, model_type, model_name, tokens_in, tokens_out, "
                "audio_in_s, audio_out_s, chars_in, latency_ms, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    datetime.now().isoformat(),
                    model_type, model_name,
                    int(kw.get("tokens_in", 0)),
                    int(kw.get("tokens_out", 0)),
                    float(kw.get("audio_in_s", 0.0)),
                    float(kw.get("audio_out_s", 0.0)),
                    int(kw.get("chars_in", 0)),
                    int(kw.get("latency_ms", 0)),
                    str(kw.get("source", "conversation")),
                ),
            )
            _db().commit()
        except Exception:
            pass  # never crash the caller


# ── Public recording API ───────────────────────────────────────────────────────

def record_stt(model_name: str, audio_seconds: float, latency_ms: int, source: str = "conversation") -> None:
    _insert("STT", model_name, audio_in_s=audio_seconds, latency_ms=latency_ms, source=source)


def record_llm(model_name: str, tokens_in: int, tokens_out: int, latency_ms: int, source: str = "conversation") -> None:
    _insert("LLM", model_name, tokens_in=tokens_in, tokens_out=tokens_out, latency_ms=latency_ms, source=source)


def record_tts(model_name: str, chars_in: int, audio_out_seconds: float, latency_ms: int, source: str = "conversation") -> None:
    _insert("TTS", model_name, chars_in=chars_in, audio_out_s=audio_out_seconds, latency_ms=latency_ms, source=source)


def record_vlm(model_name: str, tokens_in: int, tokens_out: int, latency_ms: int, source: str = "conversation") -> None:
    _insert("VLM", model_name, tokens_in=tokens_in, tokens_out=tokens_out, latency_ms=latency_ms, source=source)


# ── Query API ─────────────────────────────────────────────────────────────────

def _since(days: int) -> str:
    if days <= 0:
        return "1970-01-01"
    return (datetime.now() - timedelta(days=days)).isoformat()


def get_stats(days: int = 7) -> list[dict]:
    """Aggregated stats grouped by model_type + model_name."""
    with _lock:
        rows = _db().execute(
            "SELECT model_type, model_name, "
            "COUNT(*) AS reqs, "
            "SUM(tokens_in), SUM(tokens_out), "
            "SUM(audio_in_s), SUM(audio_out_s), "
            "SUM(chars_in), CAST(AVG(latency_ms) AS INTEGER) "
            "FROM usage_events WHERE ts > ? "
            "GROUP BY model_type, model_name ORDER BY model_type, reqs DESC",
            (_since(days),),
        ).fetchall()
    return [
        dict(
            type=r[0], model=r[1], requests=r[2],
            tokens_in=r[3] or 0, tokens_out=r[4] or 0,
            audio_in_s=round(r[5] or 0.0, 1),
            audio_out_s=round(r[6] or 0.0, 1),
            chars_in=r[7] or 0, avg_latency_ms=r[8] or 0,
        )
        for r in rows
    ]


def get_summary(days: int = 1) -> dict:
    """Single-row totals for a quick headline display."""
    with _lock:
        r = _db().execute(
            "SELECT COUNT(*), "
            "COALESCE(SUM(tokens_in),0)+COALESCE(SUM(tokens_out),0), "
            "COALESCE(SUM(audio_in_s),0), "
            "COALESCE(SUM(chars_in),0) "
            "FROM usage_events WHERE ts > ?",
            (_since(days),),
        ).fetchone()
    return dict(requests=r[0] or 0, tokens=int(r[1] or 0),
                audio_s=round(r[2] or 0.0, 1), chars=r[3] or 0)


def get_recent_events(n: int = 20) -> list[dict]:
    """Latest n events for a live feed."""
    with _lock:
        rows = _db().execute(
            "SELECT ts, model_type, model_name, tokens_in, tokens_out, "
            "audio_in_s, audio_out_s, chars_in, latency_ms, source "
            "FROM usage_events ORDER BY id DESC LIMIT ?",
            (n,),
        ).fetchall()
    return [
        dict(ts=r[0][:19], type=r[1], model=r[2],
             tokens_in=r[3], tokens_out=r[4],
             audio_in_s=round(r[5], 1), audio_out_s=round(r[6], 1),
             chars_in=r[7], latency_ms=r[8], source=r[9])
        for r in rows
    ]


def reset_stats() -> None:
    with _lock:
        _db().execute("DELETE FROM usage_events")
        _db().commit()
