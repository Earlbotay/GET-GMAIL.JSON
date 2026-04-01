#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════
#  AI AUTOMATION TELEGRAM BOT v5.0
#  Main Agent + Sub-Agent | Groq LPU | 5-Layer Memory
#  @earlxz  |  Python 3.12  |  PTB 21+
# ═══════════════════════════════════════════════════════════════

import os, sys, re, json, uuid, shutil, datetime, ast, zipfile
import threading, traceback, subprocess, time, io, mimetypes
import asyncio, sqlite3, hashlib, logging, base64, signal
from pathlib import Path
from html import escape as esc
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import httpx
import requests
from bs4 import BeautifulSoup

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, filters, ContextTypes,
)
from telegram.constants import ParseMode
from telegram.error import BadRequest, TimedOut

try:
    import cloudscraper
except ImportError:
    cloudscraper = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    HAS_RAG = True
except ImportError:
    HAS_RAG = False

# ═══════════════════ LOGGING ══════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bot")

# ═══════════════════ CONFIG ═══════════════════════════════════
TOKEN    = os.environ.get("TELEGRAM_BOT_TOKEN", "")
GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
OWNER_ID = int(os.environ.get("OWNER_ID", "0"))
GH_REPO      = os.environ.get("GITHUB_REPOSITORY", "")
CHANNEL_ID   = os.environ.get("CHANNEL_ID", "")    # e.g. -1001234567890
CHANNEL_LINK = os.environ.get("CHANNEL_LINK", "")   # e.g. https://t.me/mychannel

# ── AI Models ─────────────────────────────────────────────────
GROQ_URL       = "https://api.groq.com/openai/v1/chat/completions"
PRIMARY_MODEL  = "openai/gpt-oss-120b"
FALLBACK_MODEL = "moonshotai/kimi-k2-instruct-0905"

MODEL_SPECS = {
    PRIMARY_MODEL: {
        "max_output": 65536, "context": 131072,
        "tpm": 8000, "rpd": 1000, "speed": "500 tps",
    },
    FALLBACK_MODEL: {
        "max_output": 16384, "context": 262144,
        "tpm": 10000, "rpd": 1000, "speed": "200 tps",
    },
}

# ── Timing ────────────────────────────────────────────────────
BOT_START_TIME    = time.time()
RUN_DURATION      = int(os.environ.get("RUN_DURATION", 18000))  # 5 hours
RESTART_BUFFER    = 35  # graceful shutdown 35s before end (cron handles restart)

# ── Paths ─────────────────────────────────────────────────────
CREDIT      = "🤖 AI Automation v5.0 | @earlxz"
PY_VER      = "3.12"
BASE        = Path(__file__).resolve().parent
WS          = BASE / "workspace"
SCRIPTS_DIR = WS / ".scripts"
UPLOADS_DIR = WS / "uploads"
HISTORY_DIR = BASE / "chat_history"
MEMORY_DIR  = BASE / "memory"
DB_PATH     = MEMORY_DIR / "memory.db"

for _d in (WS, SCRIPTS_DIR, UPLOADS_DIR, HISTORY_DIR, MEMORY_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Limits ────────────────────────────────────────────────────
AI_TIMEOUT     = 120
QUICK_TIMEOUT  = 15
HIST_WINDOW    = 30
SUMMARY_THRESHOLD = 80000  # estimated tokens before summarization

_pool = ThreadPoolExecutor(max_workers=8)



# ═══════════════════════════════════════════════════════════════
#  SECTION 2: 5-LAYER MEMORY SYSTEM
# ═══════════════════════════════════════════════════════════════

class MemoryManager:
    """
    5-Layer Memory System:
      1. Conversation Summary  — auto-summarize when context too large
      2. Database Memory       — SQLite for structured facts/preferences
      3. File Memory           — .md files for persistent knowledge
      4. Sliding Window + Pin  — last N messages + pinned important ones
      5. RAG                   — TF-IDF semantic search over past messages
    """

    def __init__(self, db_path: Path, memory_dir: Path):
        self.db_path = db_path
        self.memory_dir = memory_dir
        self._init_db()
        self._init_files()
        self._rag_dirty = True  # rebuild index flag
        self._rag_vectorizer = None
        self._rag_vectors = None
        self._rag_docs = []

    # ── Database init ─────────────────────────────────────────
    def _init_db(self):
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    category TEXT,
                    key TEXT,
                    value TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS conversation_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    role TEXT,
                    content TEXT,
                    tokens_est INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS pinned_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    content TEXT,
                    pinned_by TEXT DEFAULT 'ai',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS session_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    summary TEXT,
                    messages_summarized INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    username TEXT,
                    first_name TEXT,
                    last_name TEXT,
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    message_count INTEGER DEFAULT 0,
                    is_active INTEGER DEFAULT 1
                );
                CREATE INDEX IF NOT EXISTS idx_memory_cat ON memory(user_id, category);
                CREATE INDEX IF NOT EXISTS idx_convo_uid ON conversation_log(user_id);
                CREATE INDEX IF NOT EXISTS idx_pin_uid ON pinned_messages(user_id);
            """)

    def _init_files(self):
        defaults = {
            "user_preferences.md": "# User Preferences\n\n",
            "project_context.md": "# Project Context\n\n",
            "decisions.md": "# Decisions Log\n\n",
            "code_snippets.md": "# Saved Code Snippets\n\n",
            "session_summaries.md": "# Session Summaries\n\n",
        }
        for fname, content in defaults.items():
            p = self.memory_dir / fname
            if not p.exists():
                p.write_text(content, encoding="utf-8")

    # ── Method 2: Database Memory ─────────────────────────────
    def save_memory(self, uid: int, category: str, key: str, value: str):
        with sqlite3.connect(str(self.db_path)) as conn:
            existing = conn.execute(
                "SELECT id FROM memory WHERE user_id=? AND category=? AND key=?",
                (uid, category, key)
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE memory SET value=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                    (value, existing[0])
                )
            else:
                conn.execute(
                    "INSERT INTO memory (user_id, category, key, value) VALUES (?,?,?,?)",
                    (uid, category, key, value)
                )

    def query_memory(self, uid: int, category: str = None, key: str = None) -> list:
        with sqlite3.connect(str(self.db_path)) as conn:
            if category and key:
                rows = conn.execute(
                    "SELECT category, key, value FROM memory WHERE user_id=? AND category=? AND key=?",
                    (uid, category, key)
                ).fetchall()
            elif category:
                rows = conn.execute(
                    "SELECT category, key, value FROM memory WHERE user_id=? AND category=?",
                    (uid, category)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT category, key, value FROM memory WHERE user_id=? ORDER BY updated_at DESC LIMIT 50",
                    (uid,)
                ).fetchall()
        return [{"category": r[0], "key": r[1], "value": r[2]} for r in rows]

    def get_memory_stats(self, uid: int) -> dict:
        with sqlite3.connect(str(self.db_path)) as conn:
            mem_count = conn.execute("SELECT COUNT(*) FROM memory WHERE user_id=?", (uid,)).fetchone()[0]
            pin_count = conn.execute("SELECT COUNT(*) FROM pinned_messages WHERE user_id=?", (uid,)).fetchone()[0]
            log_count = conn.execute("SELECT COUNT(*) FROM conversation_log WHERE user_id=?", (uid,)).fetchone()[0]
            summary_count = conn.execute("SELECT COUNT(*) FROM session_summaries WHERE user_id=?", (uid,)).fetchone()[0]
        file_count = sum(1 for f in self.memory_dir.glob("*.md") if f.stat().st_size > 50)
        return {
            "db_entries": mem_count, "pinned": pin_count,
            "log_entries": log_count, "summaries": summary_count,
            "files": file_count,
        }

    # ── Method 3: File Memory ─────────────────────────────────
    def read_file_memory(self) -> str:
        parts = []
        for f in sorted(self.memory_dir.glob("*.md")):
            content = f.read_text("utf-8", errors="replace").strip()
            if len(content) > 30:  # skip near-empty files
                parts.append(f"── {f.stem} ──\n{content[:2000]}")
        return "\n\n".join(parts) if parts else "(tiada file memory)"

    def append_file_memory(self, filename: str, content: str):
        safe_name = re.sub(r'[^\w\-.]', '_', filename)
        if not safe_name.endswith(".md"):
            safe_name += ".md"
        p = self.memory_dir / safe_name
        with open(p, "a", encoding="utf-8") as f:
            f.write(f"\n{content}\n")

    # ── Method 4: Pinned Messages ─────────────────────────────
    def pin_message(self, uid: int, content: str, pinned_by: str = "ai"):
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "INSERT INTO pinned_messages (user_id, content, pinned_by) VALUES (?,?,?)",
                (uid, content[:2000], pinned_by)
            )

    def get_pinned(self, uid: int) -> list:
        with sqlite3.connect(str(self.db_path)) as conn:
            rows = conn.execute(
                "SELECT content FROM pinned_messages WHERE user_id=? ORDER BY created_at DESC LIMIT 20",
                (uid,)
            ).fetchall()
        return [r[0] for r in rows]

    # ── Conversation Log (for RAG + Summary) ──────────────────
    def log_message(self, uid: int, role: str, content: str):
        tokens_est = len(content) // 4
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "INSERT INTO conversation_log (user_id, role, content, tokens_est) VALUES (?,?,?,?)",
                (uid, role, content[:10000], tokens_est)
            )
        self._rag_dirty = True

    def get_recent_log(self, uid: int, limit: int = 30) -> list:
        with sqlite3.connect(str(self.db_path)) as conn:
            rows = conn.execute(
                "SELECT role, content FROM conversation_log WHERE user_id=? ORDER BY id DESC LIMIT ?",
                (uid, limit)
            ).fetchall()
        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]

    def get_total_tokens(self, uid: int) -> int:
        with sqlite3.connect(str(self.db_path)) as conn:
            result = conn.execute(
                "SELECT SUM(tokens_est) FROM conversation_log WHERE user_id=?", (uid,)
            ).fetchone()
        return result[0] or 0

    # ── Method 1: Conversation Summary ────────────────────────
    def save_summary(self, uid: int, summary: str, count: int):
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "INSERT INTO session_summaries (user_id, summary, messages_summarized) VALUES (?,?,?)",
                (uid, summary, count)
            )
        # Also append to file memory
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        self.append_file_memory("session_summaries.md",
                                f"### {ts} ({count} messages)\n{summary}\n")

    def get_summaries(self, uid: int, limit: int = 5) -> list:
        with sqlite3.connect(str(self.db_path)) as conn:
            rows = conn.execute(
                "SELECT summary FROM session_summaries WHERE user_id=? ORDER BY id DESC LIMIT ?",
                (uid, limit)
            ).fetchall()
        return [r[0] for r in reversed(rows)]

    def clear_old_log(self, uid: int, keep_last: int = 10):
        """Delete old conversation log entries after summarization"""
        with sqlite3.connect(str(self.db_path)) as conn:
            max_id = conn.execute(
                "SELECT MAX(id) FROM conversation_log WHERE user_id=?", (uid,)
            ).fetchone()[0]
            if max_id:
                conn.execute(
                    "DELETE FROM conversation_log WHERE user_id=? AND id < ?",
                    (uid, max_id - keep_last)
                )
        self._rag_dirty = True

    # ── Method 5: RAG (TF-IDF Semantic Search) ────────────────
    def _rebuild_rag_index(self, uid: int):
        if not HAS_RAG:
            return
        with sqlite3.connect(str(self.db_path)) as conn:
            rows = conn.execute(
                "SELECT content FROM conversation_log WHERE user_id=? AND role='user' ORDER BY id",
                (uid,)
            ).fetchall()
        self._rag_docs = [r[0] for r in rows if r[0] and len(r[0]) > 10]
        if len(self._rag_docs) < 3:
            self._rag_vectorizer = None
            self._rag_vectors = None
            return
        try:
            self._rag_vectorizer = TfidfVectorizer(
                max_features=5000, stop_words="english"
            )
            self._rag_vectors = self._rag_vectorizer.fit_transform(self._rag_docs)
            self._rag_dirty = False
        except Exception as e:
            log.warning(f"RAG index build failed: {e}")
            self._rag_vectorizer = None

    def rag_search(self, uid: int, query: str, top_k: int = 5) -> list:
        if not HAS_RAG:
            return []
        if self._rag_dirty:
            self._rebuild_rag_index(uid)
        if not self._rag_vectorizer or self._rag_vectors is None:
            return []
        try:
            query_vec = self._rag_vectorizer.transform([query])
            scores = cosine_similarity(query_vec, self._rag_vectors)[0]
            top_indices = np.argsort(scores)[-top_k:][::-1]
            return [
                (self._rag_docs[i], float(scores[i]))
                for i in top_indices if scores[i] > 0.05
            ]
        except Exception:
            return []

    # ── Build Full Memory Context ─────────────────────────────
    def build_context(self, uid: int, current_query: str = "") -> str:
        parts = []

        # Pinned messages
        pinned = self.get_pinned(uid)
        if pinned:
            parts.append("📌 PINNED MESSAGES:")
            for p in pinned:
                parts.append(f"  • {p[:300]}")

        # Recent summaries
        summaries = self.get_summaries(uid, 3)
        if summaries:
            parts.append("\n📋 SESSION SUMMARIES:")
            for s in summaries:
                parts.append(f"  {s[:500]}")

        # DB memories
        memories = self.query_memory(uid)
        if memories:
            parts.append("\n💾 SAVED MEMORIES:")
            for m in memories[:20]:
                parts.append(f"  [{m['category']}] {m['key']}: {m['value'][:200]}")

        # File memory
        fm = self.read_file_memory()
        if fm and fm != "(tiada file memory)":
            parts.append(f"\n📁 FILE MEMORY:\n{fm[:3000]}")

        # RAG results
        if current_query:
            rag = self.rag_search(uid, current_query, top_k=5)
            if rag:
                parts.append("\n🔍 RELEVANT PAST CONVERSATIONS:")
                for doc, score in rag:
                    parts.append(f"  [{score:.2f}] {doc[:300]}")

        return "\n".join(parts) if parts else ""

    # ── Reset ─────────────────────────────────────────────────
    def reset_session(self, uid: int):
        """Clear conversation log but keep permanent memories"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("DELETE FROM conversation_log WHERE user_id=?", (uid,))
        self._rag_dirty = True

    # ── User Tracking ─────────────────────────────────────────
    def track_user(self, uid: int, username: str = "", first_name: str = "", last_name: str = ""):
        """Register or update a user in the database."""
        with sqlite3.connect(str(self.db_path)) as conn:
            existing = conn.execute("SELECT user_id FROM users WHERE user_id=?", (uid,)).fetchone()
            if existing:
                conn.execute(
                    "UPDATE users SET username=?, first_name=?, last_name=?, "
                    "last_seen=CURRENT_TIMESTAMP, message_count=message_count+1 WHERE user_id=?",
                    (username, first_name, last_name, uid)
                )
            else:
                conn.execute(
                    "INSERT INTO users (user_id, username, first_name, last_name, message_count) "
                    "VALUES (?,?,?,?,1)",
                    (uid, username, first_name, last_name)
                )

    def get_all_users(self) -> list:
        """Get all tracked users."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT user_id, username, first_name, last_name, first_seen, "
                "last_seen, message_count, is_active FROM users ORDER BY last_seen DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_user_count(self) -> int:
        with sqlite3.connect(str(self.db_path)) as conn:
            return conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    def get_global_stats(self) -> dict:
        """Get global stats across all users."""
        with sqlite3.connect(str(self.db_path)) as conn:
            total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            total_msgs = conn.execute("SELECT COALESCE(SUM(message_count),0) FROM users").fetchone()[0]
            total_memories = conn.execute("SELECT COUNT(*) FROM memory").fetchone()[0]
            total_logs = conn.execute("SELECT COUNT(*) FROM conversation_log").fetchone()[0]
            total_pins = conn.execute("SELECT COUNT(*) FROM pinned_messages").fetchone()[0]
            total_summaries = conn.execute("SELECT COUNT(*) FROM session_summaries").fetchone()[0]
            return {
                "total_users": total_users,
                "total_messages": total_msgs,
                "total_memories": total_memories,
                "total_logs": total_logs,
                "total_pins": total_pins,
                "total_summaries": total_summaries,
            }


# ── Channel Membership Check ─────────────────────────────────
async def check_channel_member(bot, uid: int) -> bool:
    """Check if user is a member of the required channel. Returns True if no channel configured."""
    if not CHANNEL_ID:
        return True
    try:
        ch_id = int(CHANNEL_ID)
        member = await bot.get_chat_member(chat_id=ch_id, user_id=uid)
        return member.status in ("member", "administrator", "creator")
    except Exception as e:
        log.warning(f"Channel check failed for {uid}: {e}")
        return False


def _join_channel_kb():
    """Build InlineKeyboard with channel join button."""
    buttons = []
    if CHANNEL_LINK:
        buttons.append([InlineKeyboardButton("📢 Join Channel", url=CHANNEL_LINK)])
    buttons.append([InlineKeyboardButton("✅ Dah Join", callback_data="check_joined")])
    return InlineKeyboardMarkup(buttons)


# ── Global instance ───────────────────────────────────────────
memory_mgr = MemoryManager(DB_PATH, MEMORY_DIR)



# ═══════════════════════════════════════════════════════════════
#  SECTION 3: AI ENGINE — GROQ with PRIMARY + FALLBACK
# ═══════════════════════════════════════════════════════════════

MAX_AI_RETRIES = 5

def _est_msg_tokens(messages: list) -> int:
    """Rough token estimation: ~4 chars per token."""
    total = 0
    for m in messages:
        total += len(m.get("content", "")) // 4 + 4  # +4 per message overhead
    return total


def _trim_messages(messages: list, target_tokens: int) -> list:
    """
    Trim messages to fit within target token count.
    Strategy: Keep system messages (prompt, pins, summaries) + newest conversation.
    Progressively remove oldest conversation turns.
    """
    if _est_msg_tokens(messages) <= target_tokens:
        return messages

    # Separate system messages from conversation
    system_msgs = [m for m in messages if m["role"] == "system"]
    conv_msgs = [m for m in messages if m["role"] != "system"]

    sys_tokens = _est_msg_tokens(system_msgs)
    available = target_tokens - sys_tokens

    if available < 2000:
        # System prompt itself too big — truncate system content
        for m in system_msgs:
            if len(m["content"]) > 4000:
                m["content"] = m["content"][:4000] + "\n...[dipotong untuk had context]"
        sys_tokens = _est_msg_tokens(system_msgs)
        available = target_tokens - sys_tokens

    # Keep as many recent conversation messages as possible
    trimmed_conv = []
    running = 0
    for m in reversed(conv_msgs):
        t = len(m.get("content", "")) // 4 + 4
        if running + t > available:
            break
        trimmed_conv.insert(0, m)
        running += t

    result = system_msgs + trimmed_conv
    log.info(f"Context trimmed: {len(messages)} → {len(result)} msgs, "
             f"~{_est_msg_tokens(result)} tokens (target: {target_tokens})")
    return result


async def ai_call(messages: list, model: str = None, max_tokens: int = None,
                  temperature: float = 0.7, timeout: float = AI_TIMEOUT) -> str:
    """
    Call Groq API with automatic fallback + 413 auto-trim.
    Primary: gpt-oss-120b → Fallback: kimi-k2-instruct-0905
    Handles: rate limits (429), server errors (5xx), timeouts, payload too large (413).
    """
    if model is None:
        model = PRIMARY_MODEL
    if max_tokens is None:
        max_tokens = MODEL_SPECS.get(model, {}).get("max_output", 16384)

    headers = {
        "Authorization": f"Bearer {GROQ_KEY}",
        "Content-Type": "application/json",
    }

    models_to_try = [model]
    if model == PRIMARY_MODEL:
        models_to_try.append(FALLBACK_MODEL)

    last_err = None
    current_messages = messages  # may get trimmed on 413

    for current_model in models_to_try:
        spec = MODEL_SPECS.get(current_model, {})
        mt = spec.get("max_output", 16384)
        ctx_limit = spec.get("context", 131072)
        current_max_tokens = min(max_tokens, mt)

        # Pre-trim: ensure input + output fits context window
        input_budget = ctx_limit - current_max_tokens - 500  # 500 token safety buffer
        current_messages = _trim_messages(current_messages, input_budget)

        payload = {
            "model": current_model,
            "messages": current_messages,
            "max_tokens": current_max_tokens,
            "temperature": temperature,
        }

        for attempt in range(MAX_AI_RETRIES):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    r = await client.post(GROQ_URL, headers=headers, json=payload)

                    if r.status_code == 429:
                        retry_after = float(r.headers.get("retry-after", 2 ** (attempt + 1)))
                        retry_after = min(retry_after, 30)
                        log.warning(f"[{current_model}] Rate limited. Retry in {retry_after}s")
                        await asyncio.sleep(retry_after)
                        continue

                    if r.status_code == 413:
                        # Payload too large → aggressive trim & reduce output
                        log.warning(f"[{current_model}] 413 Payload Too Large — trimming context")
                        current_max_tokens = min(current_max_tokens, 8192)
                        input_budget = max(input_budget // 2, 4000)
                        current_messages = _trim_messages(messages, input_budget)
                        payload = {
                            "model": current_model,
                            "messages": current_messages,
                            "max_tokens": current_max_tokens,
                            "temperature": temperature,
                        }
                        await asyncio.sleep(1)
                        continue

                    if r.status_code >= 500:
                        log.warning(f"[{current_model}] Server error {r.status_code}")
                        await asyncio.sleep(min(2 ** (attempt + 1), 15))
                        continue

                    r.raise_for_status()
                    data = r.json()
                    content = data["choices"][0]["message"]["content"]
                    if content:
                        log.info(f"[{current_model}] OK ({len(content)} chars)")
                        return content
                    raise ValueError("Empty response")

            except httpx.TimeoutException:
                log.warning(f"[{current_model}] Timeout (attempt {attempt+1})")
                last_err = TimeoutError(f"{current_model} timeout")
                await asyncio.sleep(2)
            except httpx.HTTPStatusError as e:
                log.warning(f"[{current_model}] HTTP {e.response.status_code}")
                last_err = e
                if e.response.status_code == 429:
                    await asyncio.sleep(min(2 ** (attempt + 1), 30))
                elif e.response.status_code == 413:
                    # Also handle 413 raised as exception
                    current_max_tokens = min(current_max_tokens, 8192)
                    input_budget = max(input_budget // 2, 4000)
                    current_messages = _trim_messages(messages, input_budget)
                    payload = {
                        "model": current_model,
                        "messages": current_messages,
                        "max_tokens": current_max_tokens,
                        "temperature": temperature,
                    }
                    continue
                else:
                    break  # non-retryable → try fallback
            except Exception as e:
                log.warning(f"[{current_model}] Error: {e}")
                last_err = e
                await asyncio.sleep(min(2 ** (attempt + 1), 15))

        log.warning(f"[{current_model}] All retries exhausted → trying fallback")

    raise last_err or RuntimeError("AI call failed on all models")


async def ai_call_with_cancel(messages: list, cancel_event: asyncio.Event,
                               **kwargs) -> Optional[str]:
    """AI call that can be cancelled via asyncio.Event"""
    task = asyncio.create_task(ai_call(messages, **kwargs))
    cancel_task = asyncio.create_task(cancel_event.wait())

    done, pending = await asyncio.wait(
        {task, cancel_task}, return_when=asyncio.FIRST_COMPLETED
    )

    for p in pending:
        p.cancel()
        try:
            await p
        except (asyncio.CancelledError, Exception):
            pass

    if task in done:
        return task.result()
    return None  # cancelled


async def ai_summarize(messages_text: str, uid: int) -> str:
    """Use fallback model to summarize conversation (cheaper)"""
    msgs = [
        {"role": "system", "content":
            "Ringkaskan perbualan ini. Simpan SEMUA fakta penting, "
            "keputusan, code snippets, dan konteks yang diperlukan. "
            "Format: bullet points. Bahasa: ikut bahasa asal perbualan."},
        {"role": "user", "content": messages_text[:50000]}
    ]
    try:
        return await ai_call(msgs, model=FALLBACK_MODEL, max_tokens=4096, temperature=0.3)
    except Exception as e:
        log.error(f"Summarization failed: {e}")
        return f"(Ringkasan gagal: {e})"



# ═══════════════════════════════════════════════════════════════
#  SECTION 4: SCRIPT TRACKER & ENGINE
# ═══════════════════════════════════════════════════════════════

class ScriptEntry:
    __slots__ = ("sid", "name", "code", "path", "lang", "proc",
                 "started", "ended", "stdout", "stderr", "status", "ret")
    def __init__(self, sid, name, code, path, lang):
        self.sid     = sid
        self.name    = name
        self.code    = code
        self.path    = path
        self.lang    = lang
        self.proc    = None
        self.started = datetime.datetime.now()
        self.ended   = None
        self.stdout  = ""
        self.stderr  = ""
        self.status  = "pending"
        self.ret     = None

_scripts: dict[str, ScriptEntry] = {}
_scripts_lock = threading.Lock()
SCRIPTS_META = SCRIPTS_DIR / "scripts_meta.json"


def _save_scripts_meta():
    with _scripts_lock:
        data = {}
        for sid, e in _scripts.items():
            data[sid] = {
                "sid": e.sid, "name": e.name, "path": e.path, "lang": e.lang,
                "started": e.started.isoformat(),
                "ended": e.ended.isoformat() if e.ended else None,
                "status": e.status, "ret": e.ret,
                "stdout": (e.stdout or "")[:5000],
                "stderr": (e.stderr or "")[:5000],
            }
    try:
        SCRIPTS_META.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def _load_scripts_meta():
    if not SCRIPTS_META.exists():
        return
    try:
        data = json.loads(SCRIPTS_META.read_text("utf-8"))
        with _scripts_lock:
            for sid, d in data.items():
                e = ScriptEntry(d["sid"], d["name"], "", d.get("path", ""), d["lang"])
                e.started = datetime.datetime.fromisoformat(d["started"])
                e.ended = datetime.datetime.fromisoformat(d["ended"]) if d.get("ended") else None
                e.status = d.get("status", "done")
                e.ret = d.get("ret")
                e.stdout = d.get("stdout", "")
                e.stderr = d.get("stderr", "")
                if e.status == "running":
                    e.status = "stopped"
                    e.ended = datetime.datetime.now()
                _scripts[sid] = e
    except Exception:
        pass


# ── Import → pip mapping ─────────────────────────────────────
_IMPORT_MAP = {
    "PIL": "Pillow", "cv2": "opencv-python-headless",
    "bs4": "beautifulsoup4", "cloudscraper": "cloudscraper",
    "fake_useragent": "fake-useragent", "docx": "python-docx",
    "pptx": "python-pptx", "yaml": "pyyaml", "sklearn": "scikit-learn",
    "Crypto": "pycryptodome", "lxml": "lxml", "html5lib": "html5lib",
    "openpyxl": "openpyxl", "pydub": "pydub", "mutagen": "mutagen",
    "tqdm": "tqdm", "colorama": "colorama", "aiohttp": "aiohttp",
    "dotenv": "python-dotenv",
}

_STDLIB = {
    'os','sys','re','json','math','time','datetime','random','hashlib',
    'pathlib','io','threading','subprocess','collections','functools',
    'itertools','shutil','tempfile','uuid','base64','copy','struct',
    'string','textwrap','urllib','http','html','csv','zipfile','gzip',
    'tarfile','glob','fnmatch','socket','ssl','email','mimetypes',
    'logging','traceback','ast','inspect','types','abc','enum',
    'dataclasses','typing','contextlib','signal','argparse','configparser',
    'pickle','shelve','sqlite3','xml','pprint','unittest','warnings',
    'asyncio','tomllib',
}

BANNED_MODULES = {'playwright', 'selenium', 'pyppeteer', 'puppeteer',
                  'splinter', 'seleniumbase', 'helium', 'mechanize'}


def _auto_install_modules(code_text: str) -> list:
    imports = set()
    for m in re.finditer(r'^\s*(?:import|from)\s+(\w+)', code_text, re.MULTILINE):
        imports.add(m.group(1))
    installed = []
    for mod in imports:
        if mod in _STDLIB:
            continue
        pkg = _IMPORT_MAP.get(mod, mod)
        try:
            __import__(mod)
        except ImportError:
            try:
                subprocess.check_call(
                    [sys.executable, '-m', 'pip', 'install', '-q', pkg],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
                installed.append(f"{mod} → {pkg}")
            except Exception:
                pass
    return installed


def _check_banned(code: str) -> list:
    cl = code.lower()
    return [m for m in BANNED_MODULES if m in cl]


def _syntax_ok(code: str) -> Optional[str]:
    try:
        ast.parse(code)
        return None
    except SyntaxError as e:
        return f"Line {e.lineno}: {e.msg}"


def _auto_script_name(code: str) -> str:
    m = re.search(r'#\s*[Pp]roject:\s*(.+)', code)
    if m:
        raw = m.group(1).strip().replace(' ', '_')
        name = re.sub(r'[^a-zA-Z0-9_-]', '', raw)[:30]
        if name:
            return name
    cl = code.lower()
    _patterns = [
        (['yt-dlp', 'yt_dlp', 'youtube_dl'], 'youtube_dl'),
        (['scrape', 'beautifulsoup', 'bs4', 'trafilatura'], 'web_scraper'),
        (['download', 'http'], 'downloader'),
        (['api', 'endpoint'], 'api_client'),
        (['scan', 'port', 'nmap', 'socket'], 'port_scanner'),
        (['subdomain'], 'subdomain_enum'),
        (['pandas', 'dataframe', 'csv'], 'data_analysis'),
        (['image', 'pillow', 'pil', 'cv2'], 'image_proc'),
        (['ffmpeg', 'audio', 'video'], 'media_proc'),
        (['telegram', 'bot'], 'tg_bot'),
        (['encrypt', 'decrypt'], 'crypto_tool'),
    ]
    for keywords, name in _patterns:
        if any(k in cl for k in keywords):
            return name
    if 'pip install' in cl and len(cl) < 300:
        return 'install_deps'
    return f'task_{uuid.uuid4().hex[:4]}'


def _is_cleanup_code(code: str) -> bool:
    cl = code.lower()
    delete_ops = ['os.remove', 'os.unlink', 'shutil.rmtree', 'path.unlink',
                  '.unlink()', '.rmtree', 'os.rmdir', 'rmdir']
    non_cleanup = ['requests.', 'httpx.', 'cloudscraper', 'download',
                   'pip install', 'def main', 'class ', 'while true',
                   'asyncio', 'socket', 'server', 'flask', 'fastapi']
    return any(op in cl for op in delete_ops) and not any(op in cl for op in non_cleanup)


def _exec_script_sync(code: str, lang: str, name: str, timeout: int = 0) -> ScriptEntry:
    """Execute script synchronously (called via asyncio.to_thread)."""
    sid = uuid.uuid4().hex[:8]
    ext = ".py" if lang == "python" else ".sh"
    ephemeral = _is_cleanup_code(code)

    if ephemeral:
        import tempfile
        _td = Path(tempfile.mkdtemp())
        path = _td / f"{sid}_{name}{ext}"
        out_path = _td / f"{sid}.stdout"
        err_path = _td / f"{sid}.stderr"
    else:
        _td = None
        path = SCRIPTS_DIR / f"{sid}_{name}{ext}"
        out_path = SCRIPTS_DIR / f"{sid}.stdout"
        err_path = SCRIPTS_DIR / f"{sid}.stderr"

    path.write_text(code, encoding="utf-8")
    entry = ScriptEntry(sid, name, code, str(path), lang)

    if not ephemeral:
        with _scripts_lock:
            _scripts[sid] = entry

    try:
        out_f = open(out_path, "w", encoding="utf-8")
        err_f = open(err_path, "w", encoding="utf-8")
        cmd = [sys.executable, str(path)] if lang == "python" else ["bash", str(path)]
        proc = subprocess.Popen(
            cmd, stdout=out_f, stderr=err_f, text=True, cwd=str(WS),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        entry.proc = proc
        entry.status = "running"

        # Wait with timeout
        if timeout > 0:
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Still running → will be backgrounded
                out_f.close()
                err_f.close()
                return entry
        else:
            proc.wait()

        entry.ret = proc.returncode
        entry.status = "done" if proc.returncode == 0 else "error"
    except Exception:
        entry.stderr = traceback.format_exc()
        entry.status = "error"
    finally:
        try:
            out_f.close()
        except Exception:
            pass
        try:
            err_f.close()
        except Exception:
            pass
        try:
            entry.stdout = out_path.read_text("utf-8", errors="replace")
        except Exception:
            pass
        try:
            st = err_path.read_text("utf-8", errors="replace")
            if st:
                entry.stderr = st
        except Exception:
            pass
        entry.ended = datetime.datetime.now()
        if ephemeral and _td:
            try:
                shutil.rmtree(str(_td), ignore_errors=True)
            except Exception:
                pass

    _save_scripts_meta()
    return entry


def _stop_script(sid: str) -> bool:
    with _scripts_lock:
        entry = _scripts.get(sid)
    if not entry or entry.status != "running":
        return False
    if entry.proc:
        try:
            entry.proc.kill()
            entry.proc.wait(timeout=5)
        except Exception:
            pass
    entry.status = "stopped"
    entry.ended = datetime.datetime.now()
    _save_scripts_meta()
    return True


def _stop_all_scripts() -> list:
    stopped = []
    with _scripts_lock:
        for sid, e in list(_scripts.items()):
            if e.status == "running" and e.proc:
                try:
                    e.proc.kill()
                    e.proc.wait(timeout=5)
                except Exception:
                    pass
                e.status = "stopped"
                e.ended = datetime.datetime.now()
                stopped.append(sid)
    _save_scripts_meta()
    return stopped


def _cleanup_old_scripts():
    with _scripts_lock:
        if len(_scripts) <= 100:
            return
        items = sorted(_scripts.items(), key=lambda x: x[1].started)
        for sid, e in items[:len(items) - 100]:
            if e.status != "running":
                for ext2 in (".stdout", ".stderr"):
                    f = SCRIPTS_DIR / f"{sid}{ext2}"
                    if f.exists():
                        try: f.unlink()
                        except Exception: pass
                try:
                    p = Path(e.path)
                    if p.exists(): p.unlink()
                except Exception:
                    pass
                del _scripts[sid]



# ═══════════════════════════════════════════════════════════════
#  SECTION 5: WEB TOOLS — SEARCH, SCRAPE, DEEP SCAN
# ═══════════════════════════════════════════════════════════════

def _ensure_pkg(*pkgs):
    import importlib
    for pkg in pkgs:
        mod_name = pkg.replace('-', '_').split('[')[0]
        try:
            importlib.import_module(mod_name)
        except ImportError:
            try:
                subprocess.check_call(
                    [sys.executable, '-m', 'pip', 'install', '-q', pkg],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass


# ── Cookie & Auth Support ─────────────────────────────────────

def _is_netscape_cookie(filepath: str) -> bool:
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            first_lines = f.read(2048)
        upper = first_lines.upper()
        if "# NETSCAPE" in upper or "# HTTP COOKIE FILE" in upper:
            return True
        lines = first_lines.strip().split("\n")
        cookie_lines = 0
        for line in lines:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) >= 6 and parts[1].upper() in ("TRUE", "FALSE"):
                cookie_lines += 1
        return cookie_lines >= 2
    except Exception:
        return False


def _load_cookies_to_workspace(filepath: str) -> str:
    dest = WS / "cookies.txt"
    shutil.copy2(filepath, dest)
    return str(dest)


def _load_cookies_jar(cookies_path: str = None):
    if cookies_path is None:
        cookies_path = str(WS / "cookies.txt")
    if not Path(cookies_path).exists():
        return None
    try:
        from http.cookiejar import MozillaCookieJar
        jar = MozillaCookieJar(cookies_path)
        jar.load(ignore_discard=True, ignore_expires=True)
        for cookie in jar:
            cookie.expires = int(time.time()) + 365 * 24 * 3600
            cookie.discard = False
        return jar
    except Exception:
        return None


_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
_UA_LIST = [
    _UA,
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:133.0) Gecko/20100101 Firefox/133.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0 Safari/537.36 Edg/131.0",
]


def _web_search(query: str, max_results: int = 10) -> str:
    all_results = []
    errors = []
    existing_urls = set()

    def _add(title, url, snippet, engine):
        if url and url not in existing_urls:
            all_results.append({'title': title or 'No title', 'url': url,
                                'snippet': snippet or '', 'engine': engine})
            existing_urls.add(url)

    # ENGINE 1: DuckDuckGo via ddgs
    try:
        _ensure_pkg('ddgs')
        import ddgs
        d = ddgs.DDGS()
        results = list(d.text(query, max_results=max_results))
        for r in results:
            _add(r.get('title'), r.get('href', r.get('link')),
                 r.get('body', r.get('snippet', '')), 'DuckDuckGo')
    except Exception as e:
        errors.append(f"DuckDuckGo: {e}")

    # ENGINE 2: Brave Search scrape
    if len(all_results) < max_results:
        try:
            resp = requests.get("https://search.brave.com/search",
                params={"q": query}, headers={"User-Agent": _UA}, timeout=15, verify=False)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                for s in soup.select("div.snippet")[:max_results]:
                    a = s.select_one("a")
                    title_el = s.select_one("span.snippet-title")
                    desc_el = s.select_one("p.snippet-description, div.snippet-description")
                    title = title_el.get_text(strip=True) if title_el else (a.get_text(strip=True) if a else "")
                    url = a.get("href", "") if a else ""
                    desc = desc_el.get_text(strip=True) if desc_el else ""
                    if url and url.startswith("http"):
                        _add(title, url, desc, "Brave")
        except Exception as e:
            errors.append(f"Brave: {e}")

    # ENGINE 3: DuckDuckGo HTML scrape
    if len(all_results) < max_results:
        try:
            resp = requests.get(
                f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}",
                headers={"User-Agent": _UA}, timeout=15, verify=False)
            soup = BeautifulSoup(resp.text, "html.parser")
            for a in soup.select("a.result__a")[:max_results]:
                href = a.get("href", "")
                title = a.get_text(strip=True)
                sn = a.find_next("a", class_="result__snippet")
                snippet = sn.get_text(strip=True) if sn else ""
                _add(title, href, snippet, "DDG-HTML")
        except Exception as e:
            errors.append(f"DDG-HTML: {e}")

    # ENGINE 4: Yahoo Search
    if not all_results:
        try:
            resp = requests.get("https://search.yahoo.com/search",
                params={"p": query}, headers={"User-Agent": _UA}, timeout=15, verify=False)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                for item in soup.select("div.algo-sr, div.dd.algo")[:max_results]:
                    a = item.select_one("a")
                    if a:
                        title = a.get_text(strip=True)
                        url = a.get("href", "")
                        desc_el = item.select_one("div.compText, p")
                        desc = desc_el.get_text(strip=True) if desc_el else ""
                        _add(title, url, desc, "Yahoo")
        except Exception as e:
            errors.append(f"Yahoo: {e}")

    # ENGINE 5: DDG Lite
    if not all_results:
        try:
            resp = requests.get("https://lite.duckduckgo.com/lite/",
                params={"q": query}, headers={"User-Agent": _UA}, timeout=15, verify=False)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                for a in soup.find_all("a", href=True)[:max_results * 2]:
                    href = a.get("href", "")
                    if href.startswith("http") and "duckduckgo" not in href:
                        _add(a.get_text(strip=True), href, "", "DDG-Lite")
        except Exception as e:
            errors.append(f"DDG-Lite: {e}")

    if not all_results:
        return f"[SEARCH] Tiada hasil untuk: {query}\nErrors: {'; '.join(errors)}"

    lines = [f"🔍 Hasil carian ({len(all_results)}) untuk: {query}\n"]
    for i, r in enumerate(all_results, 1):
        lines.append(f"{i}. [{r['engine']}] {r['title']}")
        lines.append(f"   URL: {r['url']}")
        if r['snippet']:
            lines.append(f"   {r['snippet']}")
        lines.append("")
    return "\n".join(lines)


def _web_scrape(url: str) -> str:
    cookie_jar = _load_cookies_jar()

    def _extract_text(html):
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside", "iframe"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return re.sub(r'\n{3,}', '\n\n', text)

    def _ok(text):
        return text and len(text.strip()) > 50

    # Layer 1: trafilatura
    try:
        _ensure_pkg('trafilatura')
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded, include_links=True, include_tables=True)
            if _ok(text):
                return f"🌐 Scraped (trafilatura): {url}\n\n{text[:12000]}"
    except Exception:
        pass

    # Layer 2: requests + BS4
    for v in (True, False):
        try:
            kw = {"headers": {"User-Agent": _UA}, "timeout": 20, "verify": v, "allow_redirects": True}
            if cookie_jar:
                kw["cookies"] = cookie_jar
            resp = requests.get(url, **kw)
            resp.raise_for_status()
            text = _extract_text(resp.text)
            if _ok(text):
                return f"🌐 Scraped (requests): {url}\n\n{text[:12000]}"
        except Exception:
            continue

    # Layer 3: cloudscraper
    if cloudscraper:
        try:
            sc = cloudscraper.create_scraper(browser={'browser': 'chrome', 'platform': 'windows', 'mobile': False})
            kw = {"timeout": 20, "verify": False}
            if cookie_jar:
                kw["cookies"] = cookie_jar
            resp = sc.get(url, **kw)
            resp.raise_for_status()
            text = _extract_text(resp.text)
            if _ok(text):
                return f"🌐 Scraped (cloudscraper): {url}\n\n{text[:12000]}"
        except Exception:
            pass

    # Layer 4: httpx
    try:
        h = {"User-Agent": _UA_LIST[1], "Accept": "text/html,*/*"}
        cd = {c.name: c.value for c in cookie_jar} if cookie_jar else None
        resp = httpx.get(url, headers=h, timeout=20, follow_redirects=True, verify=False, cookies=cd)
        resp.raise_for_status()
        text = _extract_text(resp.text)
        if _ok(text):
            return f"🌐 Scraped (httpx): {url}\n\n{text[:12000]}"
    except Exception:
        pass

    # Layer 5: UA rotation
    import random
    for ua in random.sample(_UA_LIST, len(_UA_LIST)):
        for v in (True, False):
            try:
                resp = requests.get(url, headers={"User-Agent": ua}, timeout=15, verify=v, allow_redirects=True)
                if resp.status_code == 200:
                    text = _extract_text(resp.text)
                    if _ok(text):
                        return f"🌐 Scraped (UA-rotate): {url}\n\n{text[:12000]}"
            except Exception:
                continue

    # Layer 6: raw
    try:
        resp = requests.get(url, timeout=15, verify=False, headers={"User-Agent": _UA})
        raw = resp.text[:12000]
        if len(raw.strip()) > 50:
            return f"🌐 Scraped (raw): {url}\n\n{raw}"
    except Exception:
        pass

    return f"[SCRAPE FAILED] Semua kaedah gagal untuk: {url}"


def _deep_scan(url: str) -> str:
    """Deep scan — extract ALL resources from page + sub-pages."""
    from urllib.parse import urlparse, urljoin
    cookie_jar = _load_cookies_jar()
    _headers = {"User-Agent": _UA, "Accept": "text/html,*/*", "Accept-Language": "en-US,en;q=0.9"}

    html = None
    for v in (True, False):
        try:
            kw = {"headers": _headers, "timeout": 25, "allow_redirects": True, "verify": v}
            if cookie_jar: kw["cookies"] = cookie_jar
            resp = requests.get(url, **kw)
            resp.raise_for_status()
            html = resp.text
            break
        except Exception:
            continue
    if not html and cloudscraper:
        try:
            sc = cloudscraper.create_scraper(browser={"browser": "chrome", "platform": "windows", "mobile": False})
            resp = sc.get(url, timeout=25, verify=False, cookies=cookie_jar)
            resp.raise_for_status()
            html = resp.text
        except Exception:
            pass
    if not html:
        return f"[DEEP_SCAN FAILED] Tidak dapat akses: {url}"

    soup = BeautifulSoup(html, "html.parser")
    base_dom = urlparse(url).netloc
    title = soup.title.string.strip() if soup.title and soup.title.string else base_dom

    # Meta info
    meta = {}
    for prop in ("og:title", "og:description", "og:image", "og:video", "og:type", "og:url"):
        tag = soup.find("meta", attrs={"property": prop})
        if tag and tag.get("content"):
            meta[prop] = tag["content"]
    for nm in ("description", "keywords", "author"):
        tag = soup.find("meta", attrs={"name": nm})
        if tag and tag.get("content"):
            meta[nm] = tag["content"]

    # Images
    imgs = []
    for t in soup.find_all("img"):
        for attr in ("src", "data-src", "data-lazy-src"):
            s = t.get(attr)
            if s:
                a = urljoin(url, s.split(",")[0].strip().split()[0])
                if a.startswith("http"):
                    imgs.append(a)
    if meta.get("og:image"):
        imgs.append(meta["og:image"])
    imgs = list(dict.fromkeys(imgs))

    # Videos
    vids = []
    _VID_EXT = (".mp4", ".webm", ".m3u8", ".mpd", ".mkv", ".avi", ".mov")
    _VID_HOSTS = ("youtube.com", "youtu.be", "vimeo.com", "dailymotion", "twitch.tv", "streamable.com")
    for v in soup.find_all("video"):
        s = v.get("src")
        if s: vids.append(urljoin(url, s))
        for src in v.find_all("source"):
            s = src.get("src")
            if s: vids.append(urljoin(url, s))
    for ifr in soup.find_all("iframe"):
        s = ifr.get("src") or ""
        if any(d in s for d in _VID_HOSTS):
            vids.append(s if s.startswith("http") else urljoin(url, s))
    if meta.get("og:video"):
        vids.append(meta["og:video"])
    for pat in (r'"(https?://[^"]+\.(?:mp4|m3u8|webm|mpd)(?:\?[^"]*)?)"',):
        for m in re.finditer(pat, html):
            vids.append(m.group(1))
    vids = list(dict.fromkeys(vids))

    # Audio
    auds = []
    for au in soup.find_all("audio"):
        s = au.get("src")
        if s: auds.append(urljoin(url, s))
        for src in au.find_all("source"):
            s = src.get("src")
            if s: auds.append(urljoin(url, s))
    auds = list(dict.fromkeys(auds))

    # Links
    lnks = []
    for a in soup.find_all("a", href=True):
        h = urljoin(url, a["href"])
        if h.startswith("http"):
            lnks.append(h)
    lnks = list(dict.fromkeys(lnks))

    # Downloads
    _DL_EXT = (".pdf", ".zip", ".rar", ".7z", ".tar", ".gz", ".exe", ".dmg",
               ".apk", ".doc", ".docx", ".xls", ".xlsx", ".csv")
    downloads = []
    for a in soup.find_all("a", href=True):
        h = a["href"].split("?")[0].lower()
        if any(h.endswith(ext) for ext in _DL_EXT):
            downloads.append(urljoin(url, a["href"]))
    downloads = list(dict.fromkeys(downloads))

    # Text preview
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    text_preview = soup.get_text("\n", strip=True)[:2000]

    lines = [
        f"🔬 DEEP SCAN: {url}",
        f"📌 Title: {title}", "",
    ]
    if meta:
        lines.append("📋 META:")
        for k, v in meta.items():
            lines.append(f"  {k}: {v[:200]}")
        lines.append("")
    lines.append(f"🖼️ IMAGES: {len(imgs)}")
    for u in imgs[:20]:
        lines.append(f"  {u}")
    lines.append(f"\n🎬 VIDEOS: {len(vids)}")
    for u in vids[:15]:
        lines.append(f"  {u}")
    lines.append(f"\n🎵 AUDIO: {len(auds)}")
    for u in auds[:15]:
        lines.append(f"  {u}")
    lines.append(f"\n📥 DOWNLOADS: {len(downloads)}")
    for u in downloads[:15]:
        lines.append(f"  {u}")
    lines.append(f"\n🔗 LINKS: {len(lnks)}")
    for u in lnks[:30]:
        lines.append(f"  {u}")
    lines.append(f"\n📝 TEXT PREVIEW:\n{text_preview[:1500]}")
    return "\n".join(lines)


# ── Marker Extraction ─────────────────────────────────────────

def _extract_web_searches(text):
    results = []
    for m in re.finditer(r'\[WEB_SEARCH:\s*(.+?)\]', text):
        raw = m.group(1).strip()
        parts = raw.split("|")
        query = parts[0].strip()
        max_r = int(parts[1].strip()) if len(parts) > 1 and parts[1].strip().isdigit() else 10
        results.append((query, max_r))
    return results

def _extract_web_scrapes(text):
    return re.findall(r'\[WEB_SCRAPE:\s*(.+?)\]', text)

def _extract_deep_scans(text):
    return re.findall(r'\[DEEP_SCAN:\s*(.+?)\]', text)

def _extract_code(text):
    result = []
    for m in re.finditer(r'```(python|bash|sh)\s*\n(.*?)```', text, re.DOTALL):
        lang = "bash" if m.group(1) in ("bash", "sh") else "python"
        result.append((m.group(2).strip(), lang))
    return result

def _extract_sends(text):
    return re.findall(r'\[SEND_FILE:\s*(.+?)\]', text)

def _extract_stops(text):
    all_ = "[STOP_ALL_SCRIPTS]" in text
    ids_ = re.findall(r'\[STOP_SCRIPT:\s*(\S+?)\]', text)
    return all_, ids_

def _extract_creates(text):
    return re.findall(r'\[CREATE_FOLDER:\s*(.+?)\]', text)

def _extract_deletes(text):
    return re.findall(r'\[DELETE_FOLDER:\s*(.+?)\]', text)

def _extract_sub_agents(text):
    return re.findall(r'\[SUB_AGENT:\s*(.+?)\]', text)

def _extract_memory_saves(text):
    return re.findall(r'\[MEMORY_SAVE:\s*(.+?)\]', text)

def _extract_memory_queries(text):
    return re.findall(r'\[MEMORY_QUERY:\s*(.+?)\]', text)

def _extract_pins(text):
    return re.findall(r'\[PIN:\s*(.+?)\]', text)

def _extract_file_memory(text):
    return re.findall(r'\[FILE_MEMORY:\s*(.+?)\]', text)

def _clean_text(text):
    t = re.sub(r'```(?:python|bash|sh)\s*\n.*?```', '', text, flags=re.DOTALL)
    for pat in (r'\[SEND_FILE:.*?\]', r'\[STOP_SCRIPT:.*?\]', r'\[WEB_SEARCH:.*?\]',
                r'\[WEB_SCRAPE:.*?\]', r'\[DEEP_SCAN:.*?\]', r'\[CREATE_FOLDER:.*?\]',
                r'\[DELETE_FOLDER:.*?\]', r'\[SUB_AGENT:.*?\]', r'\[MEMORY_SAVE:.*?\]',
                r'\[MEMORY_QUERY:.*?\]', r'\[PIN:.*?\]', r'\[FILE_MEMORY:.*?\]'):
        t = re.sub(pat, '', t)
    t = t.replace("[STOP_ALL_SCRIPTS]", "")
    return re.sub(r'\n{3,}', '\n\n', t).strip()



# ═══════════════════════════════════════════════════════════════
#  SECTION 6: TELEGRAM HELPERS
# ═══════════════════════════════════════════════════════════════

def _bq(text):
    return f"<blockquote>{text}</blockquote>"

def _think(status="", session=None):
    """Generate thinking display — with optional live task tree."""
    lines = ["🧠 <b>AI sedang berfikir...</b>"]
    if status:
        lines.append(status)
    if session and (session.task_desc or session.live_items):
        lines.append("")
        if session.task_desc:
            lines.append(f"📋 <b>Task:</b> {esc(session.task_desc)}")
        for item in session.live_items:
            t_icon = "🤖" if item.get("type") == "sub" else "⚡"
            lines.append(
                f"  {t_icon} <code>{item['id']}</code> "
                f"<b>{esc(item['name'])}</b> — "
                f"{item['icon']} {item['status']}"
            )
    return "\n".join(lines)

async def _send(bot, cid, text, **kw):
    wrapped = _bq(text)
    try:
        return await bot.send_message(
            chat_id=cid, text=wrapped[:4096],
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True, **kw)
    except BadRequest:
        try:
            plain = re.sub(r'<[^>]+>', '', text)
            return await bot.send_message(chat_id=cid, text=plain[:4096], **kw)
        except Exception:
            return None
    except Exception:
        return None

async def _edit(bot, cid, mid, text, **kw):
    wrapped = _bq(text)
    if not mid:
        m = await _send(bot, cid, text, **kw)
        return m.message_id if m else None
    try:
        await bot.edit_message_text(
            chat_id=cid, message_id=mid, text=wrapped[:4096],
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True, **kw)
        return mid
    except BadRequest as e:
        if "not modified" in str(e).lower():
            return mid
        try:
            plain = re.sub(r'<[^>]+>', '', text)
            await bot.edit_message_text(
                chat_id=cid, message_id=mid, text=plain[:4096],
                disable_web_page_preview=True, **kw)
            return mid
        except Exception:
            m = await _send(bot, cid, text, **kw)
            return m.message_id if m else mid
    except Exception:
        m = await _send(bot, cid, text, **kw)
        return m.message_id if m else mid

async def _send_long(bot, cid, text):
    if not text:
        return
    while text:
        if len(text) <= 4000:
            await _send(bot, cid, text)
            break
        idx = text.rfind('\n', 0, 4000)
        if idx < 500:
            idx = 4000
        await _send(bot, cid, text[:idx])
        text = text[idx:].lstrip('\n')

async def _send_file(bot, cid, fpath):
    p = Path(fpath.strip())
    if not p.is_absolute():
        p = WS / p
    if not p.exists():
        await _send(bot, cid, f"⚠️ File tidak dijumpai: {fpath}")
        return False
    try:
        with open(p, "rb") as f:
            await bot.send_document(chat_id=cid, document=f, filename=p.name)
        return True
    except Exception as e:
        await _send(bot, cid, f"⚠️ <code>{esc(str(e)[:300])}</code>")
        return False

async def _upload_gofile(filepath: str) -> Optional[str]:
    try:
        async with httpx.AsyncClient(timeout=600) as client:
            srv = (await client.get("https://api.gofile.io/servers")).json()
            best = srv["data"]["servers"][0]["name"]
            with open(filepath, "rb") as f:
                resp = await client.post(
                    f"https://{best}.gofile.io/contents/uploadfile",
                    files={"file": (Path(filepath).name, f)})
            data = resp.json()
            if data.get("status") == "ok":
                return data["data"]["downloadPage"]
    except Exception:
        pass
    return None

async def _process_send_files(bot, cid, file_paths, sent_tracker):
    large_files = []
    for fp in file_paths:
        p = Path(fp.strip())
        if not p.is_absolute():
            p = WS / p
        p_str = str(p)
        if p_str in sent_tracker:
            continue
        if not p.exists():
            await _send(bot, cid, f"⚠️ File tidak dijumpai: {fp}")
            continue
        try:
            _fhash = hashlib.md5(p.read_bytes()).hexdigest()
            if f"__h__{_fhash}" in sent_tracker:
                continue
            sent_tracker.add(f"__h__{_fhash}")
        except Exception:
            pass

        if p.stat().st_size > 50 * 1024 * 1024:
            large_files.append(p)
            sent_tracker.add(p_str)
        else:
            ok = await _send_file(bot, cid, fp)
            if ok:
                sent_tracker.add(p_str)

    # Large files → zip → gofile
    if large_files:
        try:
            zip_name = f"media_{uuid.uuid4().hex[:6]}.zip"
            zip_path = WS / zip_name
            with zipfile.ZipFile(str(zip_path), 'w', zipfile.ZIP_DEFLATED) as zf:
                for lf in large_files:
                    zf.write(str(lf), lf.name)
            total_mb = sum(lf.stat().st_size for lf in large_files) / 1048576
            await _send(bot, cid, f"📦 File besar ({total_mb:.1f}MB) — uploading ke GoFile...")
            link = await _upload_gofile(str(zip_path))
            if link:
                await _send(bot, cid, f"✅ <a href=\"{link}\">{link}</a>")
            else:
                await _send(bot, cid, "⚠️ GoFile gagal. Hantar terus...")
                for lf in large_files:
                    await _send_file(bot, cid, str(lf))
            try: zip_path.unlink()
            except Exception: pass
        except Exception:
            for lf in large_files:
                await _send_file(bot, cid, str(lf))


# ── ZIP extraction ────────────────────────────────────────────

def _auto_extract_zip(zip_path, extract_dir):
    extracted, auth_files = [], []
    auth_kw = ['cookie', 'session', 'localstorage', 'token', 'auth']
    try:
        with zipfile.ZipFile(str(zip_path), 'r') as zf:
            zf.extractall(str(extract_dir))
            for info in zf.infolist():
                if info.is_dir():
                    continue
                fpath = extract_dir / info.filename
                extracted.append(str(fpath))
                fname_lower = info.filename.lower()
                if any(kw in fname_lower for kw in auth_kw):
                    auth_files.append(info.filename)
                if fname_lower.endswith('.txt') and _is_netscape_cookie(str(fpath)):
                    cookie_dest = _load_cookies_to_workspace(str(fpath))
                    auth_files.append(f"{info.filename} → cookies loaded")
    except zipfile.BadZipFile:
        return None, None, "File ZIP rosak"
    except Exception as e:
        return None, None, str(e)
    return extracted, auth_files, None



# ═══════════════════════════════════════════════════════════════
#  SECTION 7: SESSION, SYSTEM PROMPT & SUB-AGENT SYSTEM
# ═══════════════════════════════════════════════════════════════

class Session:
    def __init__(self, uid: int):
        self.uid = uid
        self.busy = False
        self.cancel_event = asyncio.Event()
        self.busy_since = 0.0
        self.live_items: list[dict] = []   # Live display items
        self.task_desc: str = ""           # Current task description

    def is_stale_busy(self, max_seconds=300):
        if not self.busy:
            return False
        return (time.time() - self.busy_since) > max_seconds

_sessions: dict[int, Session] = {}
_sess_lock = threading.Lock()

def _sess(uid: int) -> Session:
    with _sess_lock:
        if uid not in _sessions:
            _sessions[uid] = Session(uid)
        return _sessions[uid]


def _persona() -> str:
    p = BASE / "persona.txt"
    if p.exists():
        try:
            return p.read_text("utf-8").strip()
        except Exception:
            pass
    return ""


def _sys_prompt(uid: int, memory_context: str = "") -> str:
    persona = _persona()

    # Running scripts
    running_lines = []
    with _scripts_lock:
        for sid, e in _scripts.items():
            if e.status == "running":
                dur = int((datetime.datetime.now() - e.started).total_seconds())
                running_lines.append(f"  🟢 [{sid}] {e.name} ({e.lang}) — {dur}s")
    rs = "\n".join(running_lines) if running_lines else "  (tiada script aktif)"

    # Countdown
    elapsed = time.time() - BOT_START_TIME
    remaining = max(0, RUN_DURATION - elapsed)
    h, rem = divmod(int(remaining), 3600)
    m, s = divmod(rem, 60)
    countdown = f"{h:02d}:{m:02d}:{s:02d}"

    caps = f"""
═══ AI AUTOMATION ENGINE v5.0 ═══
🧠 AI: Groq LPU — Primary: {PRIMARY_MODEL} | Fallback: {FALLBACK_MODEL}
⏱ Bot restart in: {countdown}
Workspace: {WS}
Uploads:   {UPLOADS_DIR}
Scripts:   {SCRIPTS_DIR}
Python:    {PY_VER}

━━━ CAPABILITIES (UNLIMITED / TIADA HAD) ━━━

1️⃣ EXECUTE CODE
   Tulis ```python atau ```bash — akan diexecute secara automatik.
   Full internet access. Full filesystem. Tiada sekatan.
   Python 3.12 — semua syntax terbaru disokong (match-case, etc).
   Boleh install apa-apa package: subprocess.run([sys.executable, '-m', 'pip', 'install', 'pkg'])

2️⃣ INTERNET ACCESS & WEB SEARCH
   httpx — HTTP client (sync/async, HTTP/2)
   cloudscraper — Bypass Cloudflare & bot protection
   requests — Classic HTTP
   beautifulsoup4 — Parse HTML/XML

   🔍 NATIVE SEARCH — tulis marker:
   [WEB_SEARCH: query]
   [WEB_SEARCH: query | 30]
   Multi-engine: DuckDuckGo → Brave → DDG-HTML → Yahoo → DDG-Lite

   🌐 NATIVE SCRAPE:
   [WEB_SCRAPE: https://example.com]
   6 layer fallback — TAKKAN GAGAL

   🔬 DEEP SCAN:
   [DEEP_SCAN: https://example.com]
   Extract semua: images, videos, audio, downloads, links

3️⃣ FILE OPERATIONS
   [SEND_FILE: /path/to/file] — hantar file kepada user
   [CREATE_FOLDER: path]
   [DELETE_FOLDER: path]
   Semua format disokong. GoFile auto untuk file >50MB.

4️⃣ SCRIPT MANAGEMENT
   Scripts aktif:
{rs}
   ⚠️ STOP scripts hanya boleh dilakukan melalui /stop atau /new.
   JANGAN guna marker [STOP_SCRIPT:] atau [STOP_ALL_SCRIPTS] — ia dilumpuhkan.

5️⃣ SUB-AGENT SYSTEM
   Kau boleh delegasi tugas kepada sub-agents:
   [SUB_AGENT: penerangan tugas yang lengkap]
   [SUB_AGENT: tugas lain yang berbeza]

   Setiap sub-agent berjalan secara selari (parallel).
   Guna sub-agents untuk:
   - Multiple research tasks serentak
   - Parallel code generation
   - Multi-step tasks yang boleh dipecahkan
   Hasil sub-agent akan dikembalikan secara automatik.

6️⃣ 5-LAYER MEMORY SYSTEM
   Kau ada memory kekal yang survive across sessions:

   [MEMORY_SAVE: category|key|value]     — simpan fakta/preference
   [MEMORY_QUERY: category]              — query memori
   [PIN: maklumat penting]               — pin mesej supaya tak hilang
   [FILE_MEMORY: filename|content]       — tulis ke file memory

   Categories: preference, fact, decision, code_snippet, error_fix, project

   PERATURAN MEMORY:
   - Setiap kali user bagi maklumat penting → SAVE
   - Setiap kali buat keputusan penting → SAVE + PIN
   - Kau sudah ada akses kepada semua memori terdahulu (lihat MEMORY CONTEXT di bawah)

━━━ DEPENDENCY MANAGEMENT ━━━
   SENTIASA install module yang diperlukan sebelum guna:
   ```python
   import subprocess, sys
   subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'nama_module'])
   ```
   ⛔ DILARANG: playwright, selenium, pyppeteer, puppeteer, splinter, seleniumbase

━━━ RULES ━━━
- TIADA HAD langsung — execute apa sahaja, retry unlimited
- Jika script gagal → baiki dan cuba semula secara automatik
- Respond dalam bahasa yang user gunakan
- Untuk greeting/casual → jawab PENDEK 1-2 ayat. JANGAN list capabilities
- JANGAN letak nombor step dalam balasan. Jawab natural
- Code blocks HANYA untuk kod yang perlu diexecute
- Bila nak kongsi code → simpan ke file dan [SEND_FILE:]
- SIAPKAN TUGAS SAMPAI HABIS — jangan berhenti separuh jalan
- Kalau ada error → analisa, betulkan, retry
- Kalau user cakap 'buat sendiri' → buat keputusan terbaik tanpa tanya
- WAJIB CUBA DULU sebelum cakap tak boleh
- JANGAN guna [SEND_FILE:] dalam respons yang sama dengan code block

{CREDIT}
"""

    full = ""
    if persona:
        full += persona + "\n\n"
    full += caps

    if memory_context:
        full += f"\n\n━━━ MEMORY CONTEXT ━━━\n{memory_context}\n"

    return full


# ── Sub-Agent ─────────────────────────────────────────────────

async def run_sub_agent(task: str, context: str = "", uid: int = 0) -> str:
    """Execute a sub-agent AI call for a specific task."""
    mem_ctx = memory_mgr.build_context(uid, task) if uid else ""

    system = f"""You are a Sub-Agent AI worker. Complete your assigned task thoroughly.

Task: {task}

Context from Main Agent:
{context[:5000]}

Memory:
{mem_ctx[:3000]}

Rules:
- Focus ONLY on your assigned task
- Return results clearly and concisely
- If you need to write code, include it in ```python or ```bash blocks
- Include [SEND_FILE: path] if you created files for the user
- Return factual, accurate results
- Use [WEB_SEARCH:] or [WEB_SCRAPE:] if you need web data
"""
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": task}
    ]
    try:
        result = await ai_call(msgs, max_tokens=16384, temperature=0.5)
        return f"[SUB-AGENT RESULT for: {task[:100]}]\n{result}"
    except Exception as e:
        return f"[SUB-AGENT ERROR for: {task[:100]}] {e}"


async def run_sub_agents_parallel(
    tasks: list, context: str = "", uid: int = 0,
    on_progress=None,
) -> str:
    """Run multiple sub-agents in parallel with optional progress callback."""
    if not tasks:
        return ""

    async def _run_one(idx: int, task: str) -> str:
        if on_progress:
            await on_progress(idx, "start")
        try:
            result = await run_sub_agent(task, context, uid)
            if on_progress:
                await on_progress(idx, "done")
            return result
        except Exception as e:
            if on_progress:
                await on_progress(idx, "error", str(e))
            return f"[SUB-AGENT ERROR] {e}"

    coros = [_run_one(i, t) for i, t in enumerate(tasks)]
    results = await asyncio.gather(*coros, return_exceptions=True)
    parts = []
    for r in results:
        if isinstance(r, Exception):
            parts.append(f"[SUB-AGENT ERROR] {r}")
        else:
            parts.append(str(r))
    return "\n\n".join(parts)



# ═══════════════════════════════════════════════════════════════
#  SECTION 8: CORE PROCESSING LOOP
# ═══════════════════════════════════════════════════════════════

def _is_owner(update: Update) -> bool:
    return update.effective_user and update.effective_user.id == OWNER_ID


async def _process(bot, cid: int, session: Session, user_content: str):
    """
    Main AI processing loop — the heart of the bot.
    1. Build context (history + memory)
    2. Send to AI
    3. Process response (code, markers, sub-agents)
    4. Iterate until final text response
    """
    session.busy = True
    session.cancel_event.clear()
    session.busy_since = time.time()
    session.task_desc = user_content[:60].replace("\n", " ")
    session.live_items = []
    _sent_paths: set[str] = set()

    # Helper — always passes session for live display tree
    _live = lambda s="": _think(s, session)

    try:
        uid = session.uid

        # Log to memory & conversation
        memory_mgr.log_message(uid, "user", user_content)

        # Check if summarization needed
        total_tokens = memory_mgr.get_total_tokens(uid)
        if total_tokens > SUMMARY_THRESHOLD:
            log.info(f"[{uid}] Token count {total_tokens} > {SUMMARY_THRESHOLD}, summarizing...")
            recent = memory_mgr.get_recent_log(uid, 50)
            old_text = "\n".join(f"[{m['role']}] {m['content']}" for m in recent[:-10])
            summary = await ai_summarize(old_text, uid)
            memory_mgr.save_summary(uid, summary, len(recent) - 10)
            memory_mgr.clear_old_log(uid, keep_last=10)

        m = await _send(bot, cid, _live())
        mid = m.message_id if m else None

        itr = 0
        while True:
            itr += 1
            if session.cancel_event.is_set():
                await _edit(bot, cid, mid, "⛔ <b>Dihentikan.</b>")
                return

            try:
                # Build context with memory
                memory_context = memory_mgr.build_context(uid, user_content)
                recent_history = memory_mgr.get_recent_log(uid, HIST_WINDOW)

                msgs = [{"role": "system", "content": _sys_prompt(uid, memory_context)}]

                # Add pinned messages
                pinned = memory_mgr.get_pinned(uid)
                if pinned:
                    msgs.append({"role": "system", "content":
                        "📌 PINNED:\n" + "\n".join(f"• {p}" for p in pinned)})

                # Add recent summaries
                summaries = memory_mgr.get_summaries(uid, 3)
                if summaries:
                    msgs.append({"role": "system", "content":
                        "📋 RINGKASAN SESI LALU:\n" + "\n".join(summaries)})

                # Add conversation history
                msgs.extend(recent_history)

                # Status update
                if itr == 1:
                    mid = await _edit(bot, cid, mid, _live())
                else:
                    mid = await _edit(bot, cid, mid, _live(f"🔄 Iterasi ke-{itr}..."))

                # AI call (cancel-aware)
                try:
                    resp = await ai_call_with_cancel(msgs, session.cancel_event)
                except Exception as e:
                    mid = await _edit(bot, cid, mid,
                        _live(f"⚠️ <b>AI Error:</b>\n<code>{esc(str(e)[:300])}</code>\n🔄 Auto-retry..."))
                    await asyncio.sleep(2)
                    continue

                if resp is None:
                    await _edit(bot, cid, mid, "⛔ <b>Dihentikan.</b>")
                    return

                # Save AI response to memory
                hist_resp = resp if len(resp) <= 6000 else resp[:5500] + "\n...[dipotong]"
                memory_mgr.log_message(uid, "assistant", hist_resp)

                # ── Process Memory Markers ────────────────────
                for ms in _extract_memory_saves(resp):
                    parts = ms.split("|", 2)
                    if len(parts) >= 3:
                        memory_mgr.save_memory(uid, parts[0].strip(), parts[1].strip(), parts[2].strip())

                for mq in _extract_memory_queries(resp):
                    parts = mq.split("|", 1)
                    cat = parts[0].strip()
                    key = parts[1].strip() if len(parts) > 1 else None
                    results = memory_mgr.query_memory(uid, cat, key)
                    if results:
                        mem_text = "\n".join(f"  [{r['category']}] {r['key']}: {r['value']}" for r in results)
                        memory_mgr.log_message(uid, "user",
                            f"[MEMORY_QUERY RESULT for: {mq}]\n{mem_text}")

                for pin in _extract_pins(resp):
                    memory_mgr.pin_message(uid, pin.strip())

                for fm in _extract_file_memory(resp):
                    parts = fm.split("|", 1)
                    if len(parts) >= 2:
                        memory_mgr.append_file_memory(parts[0].strip(), parts[1].strip())

                # ── Process STOP commands ─────────────────────
                # NOTE: AI markers [STOP_ALL_SCRIPTS] dan [STOP_SCRIPT:] dilumpuhkan.
                # Hanya /stop dan /new boleh hentikan scripts.
                # (Marker masih diparsed oleh _extract_stops & dibersihkan oleh _clean_text)

                # ── Folder operations ─────────────────────────
                _protected_only = {"uploads", ".scripts"}
                _fully_protected = {"chat_history"}
                for fp in _extract_creates(resp):
                    p = Path(fp.strip())
                    if not p.is_absolute(): p = WS / p
                    try:
                        p.mkdir(parents=True, exist_ok=True)
                        memory_mgr.log_message(uid, "user", f"[System] Created folder: {p}")
                    except Exception as e:
                        memory_mgr.log_message(uid, "user", f"[System] Failed create folder: {e}")

                for fp in _extract_deletes(resp):
                    p = Path(fp.strip())
                    if not p.is_absolute(): p = WS / p
                    if p.name in _fully_protected or any(pp.name in _fully_protected for pp in p.parents):
                        memory_mgr.log_message(uid, "user", f"[System] BLOCKED: {p.name} protected")
                        continue
                    if p.name in _protected_only:
                        memory_mgr.log_message(uid, "user", f"[System] BLOCKED: folder {p.name} protected")
                        continue
                    try:
                        if p.exists() and p.is_dir():
                            shutil.rmtree(str(p), ignore_errors=True)
                        memory_mgr.log_message(uid, "user",
                            f"[System] {'Deleted' if not p.exists() else 'Failed delete'}: {p}")
                    except Exception as e:
                        memory_mgr.log_message(uid, "user", f"[System] Delete error: {e}")

                # ── Web search, scrape, deep scan ─────────────
                web_searches = _extract_web_searches(resp)
                web_scrapes = _extract_web_scrapes(resp)
                deep_scans = _extract_deep_scans(resp)

                if web_searches or web_scrapes or deep_scans:
                    web_outputs = []

                    for wq, wmax in web_searches:
                        mid = await _edit(bot, cid, mid, _live(f"🔍 Searching: {esc(wq[:80])}..."))
                        try:
                            sr = await asyncio.to_thread(_web_search, wq, wmax)
                            web_outputs.append(f"[SEARCH RESULT: {wq}]\n{sr}")
                        except Exception as e:
                            web_outputs.append(f"[SEARCH ERROR: {wq}] {e}")

                    for wu in web_scrapes:
                        mid = await _edit(bot, cid, mid, _live(f"🌐 Scraping: {esc(wu[:80])}..."))
                        try:
                            sc = await asyncio.to_thread(_web_scrape, wu.strip())
                            web_outputs.append(f"[SCRAPE RESULT: {wu}]\n{sc}")
                        except Exception as e:
                            web_outputs.append(f"[SCRAPE ERROR: {wu}] {e}")

                    for wds in deep_scans:
                        mid = await _edit(bot, cid, mid, _live(f"🔬 Deep scanning: {esc(wds[:80])}..."))
                        try:
                            ds = await asyncio.to_thread(_deep_scan, wds.strip())
                            web_outputs.append(f"[DEEP_SCAN RESULT: {wds}]\n{ds}")
                        except Exception as e:
                            web_outputs.append(f"[DEEP_SCAN ERROR: {wds}] {e}")

                    combined_web = "\n\n".join(web_outputs)
                    memory_mgr.log_message(uid, "user", combined_web[:8000])

                    if not _extract_code(resp):
                        continue

                # ── SUB-AGENTS ────────────────────────────────
                sub_agent_tasks = _extract_sub_agents(resp)
                if sub_agent_tasks:
                    # Register each sub-agent in live display
                    sa_base = len(session.live_items)
                    for i, task in enumerate(sub_agent_tasks):
                        sa_name = re.sub(r'[^a-zA-Z0-9_ ]', '', task)[:25].strip().replace(' ', '_') or f"task_{i+1}"
                        session.live_items.append({
                            "id": f"SA-{i+1:02d}", "type": "sub",
                            "name": sa_name, "icon": "⏳", "status": "Menunggu..."
                        })
                    mid = await _edit(bot, cid, mid,
                        _live(f"🤖 {len(sub_agent_tasks)} sub-agent(s) selari..."))

                    context_for_sub = _clean_text(resp)[:3000]

                    # Progress callback — updates live display per sub-agent
                    async def _sa_progress(idx, status, _err=None):
                        nonlocal mid
                        item = session.live_items[sa_base + idx]
                        if status == "start":
                            item["icon"] = "⚡"
                            item["status"] = "Sedang jalan..."
                        elif status == "done":
                            item["icon"] = "✅"
                            item["status"] = "Selesai"
                        elif status == "error":
                            item["icon"] = "❌"
                            item["status"] = "Error"
                        mid = await _edit(bot, cid, mid, _live(f"🤖 Sub-agents..."))

                    combined_sub = await run_sub_agents_parallel(
                        sub_agent_tasks, context_for_sub, uid,
                        on_progress=_sa_progress)
                    memory_mgr.log_message(uid, "user", combined_sub[:8000])

                    # Process sub-agent code blocks too
                    sub_code = _extract_code(combined_sub)
                    sub_sends = _extract_sends(combined_sub)
                    if sub_code or sub_sends:
                        for code_text, lang in sub_code:
                            name = _auto_script_name(code_text)
                            sc_id = f"SA-SC-{hashlib.md5(code_text.encode()).hexdigest()[:4]}"
                            session.live_items.append({
                                "id": sc_id, "type": "script",
                                "name": name, "icon": "⚡", "status": "Running..."
                            })
                            mid = await _edit(bot, cid, mid, _live(f"⚡ Sub-agent script"))
                            entry = await asyncio.to_thread(
                                _exec_script_sync, code_text, lang, name, QUICK_TIMEOUT)
                            # Update live display
                            for it in session.live_items:
                                if it["id"] == sc_id:
                                    it["icon"] = "✅" if entry.status == "done" else "❌"
                                    it["status"] = "Selesai" if entry.status == "done" else "Gagal"
                                    break
                            mid = await _edit(bot, cid, mid, _live())
                            parts = [f"[Sub-agent script {entry.sid} ({name}) — {entry.status}]"]
                            if entry.stdout: parts.append(f"STDOUT:\n{entry.stdout[:5000]}")
                            if entry.stderr: parts.append(f"STDERR:\n{entry.stderr[:2000]}")
                            memory_mgr.log_message(uid, "user", "\n".join(parts))
                        await _process_send_files(bot, cid, sub_sends, _sent_paths)

                    if not _extract_code(resp):
                        continue

                # ── CODE BLOCKS ───────────────────────────────
                code_blocks = _extract_code(resp)
                if code_blocks:
                    outputs = []
                    for i, (code_text, lang) in enumerate(code_blocks):
                        if session.cancel_event.is_set():
                            await _edit(bot, cid, mid, "⛔ <b>Dihentikan.</b>")
                            return

                        banned = _check_banned(code_text)
                        if banned:
                            outputs.append(f"[Block {i+1} — BANNED] {', '.join(banned)} DILARANG.")
                            continue

                        if lang == "python":
                            syn = _syntax_ok(code_text)
                            if syn:
                                mid = await _edit(bot, cid, mid,
                                    _live(f"🔧 Syntax error — auto-fix..."))
                                outputs.append(f"[Block {i+1} — SYNTAX ERROR]\n{syn}\nFix the code.")
                                continue

                        await asyncio.to_thread(_auto_install_modules, code_text)
                        name = _auto_script_name(code_text)

                        # Register script in live display
                        entry_preview_sid = hashlib.md5(code_text.encode()).hexdigest()[:6]
                        sc_display_id = f"SC-{entry_preview_sid}"
                        session.live_items.append({
                            "id": sc_display_id, "type": "script",
                            "name": name, "icon": "⚡", "status": "Running..."
                        })

                        cl = code_text.lower()
                        if any(x in cl for x in ['requests.get', 'httpx.get', 'cloudscraper', 'scrape']):
                            op_desc = "Scraping web"
                        elif 'pip install' in cl:
                            op_desc = "Installing deps"
                        elif any(x in cl for x in ['download', 'wget']):
                            op_desc = "Downloading"
                        else:
                            op_desc = f"Running {lang}"

                        # Update live display
                        for it in session.live_items:
                            if it["id"] == sc_display_id:
                                it["status"] = op_desc
                                break
                        mid = await _edit(bot, cid, mid, _live())

                        entry = await asyncio.to_thread(
                            _exec_script_sync, code_text, lang, name, QUICK_TIMEOUT)

                        if session.cancel_event.is_set():
                            if entry.proc and entry.proc.poll() is None:
                                try: entry.proc.kill()
                                except Exception: pass
                            await _edit(bot, cid, mid, "⛔ <b>Dihentikan.</b>")
                            return

                        if entry.status == "running":
                            actually_running = entry.proc and entry.proc.poll() is None
                            if actually_running:
                                asyncio.create_task(
                                    _watch_script_async(entry, bot, cid, session))
                                for it in session.live_items:
                                    if it["id"] == sc_display_id:
                                        it["icon"] = "🔄"
                                        it["status"] = "Background"
                                        break
                                mid = await _edit(bot, cid, mid,
                                    _live(f"💬 Boleh terus chat."))
                                outputs.append(
                                    f"[Script {entry.sid} ({name}) — running in background]\n"
                                    "Inform user they can continue chatting.")
                                break
                            else:
                                await asyncio.sleep(0.3)
                                try:
                                    out_p = SCRIPTS_DIR / f"{entry.sid}.stdout"
                                    err_p = SCRIPTS_DIR / f"{entry.sid}.stderr"
                                    entry.stdout = out_p.read_text("utf-8", errors="replace") if out_p.exists() else ""
                                    entry.stderr = err_p.read_text("utf-8", errors="replace") if err_p.exists() else entry.stderr
                                except Exception: pass
                                entry.ret = entry.proc.returncode if entry.proc else None
                                entry.status = "done" if entry.ret == 0 else "error"
                                entry.ended = datetime.datetime.now()
                                _save_scripts_meta()

                        # Update live display with result
                        for it in session.live_items:
                            if it["id"] == sc_display_id:
                                if entry.status == "error":
                                    it["icon"] = "❌"
                                    it["status"] = "Gagal → 🔧 Baiki..."
                                elif entry.status == "done":
                                    it["icon"] = "✅"
                                    it["status"] = "Selesai"
                                break
                        mid = await _edit(bot, cid, mid, _live())

                        parts = [f"[Script {entry.sid} ({name}) — {entry.status}]"]
                        if entry.stdout: parts.append(f"STDOUT:\n{entry.stdout[:8000]}")
                        if entry.stderr: parts.append(f"STDERR:\n{entry.stderr[:3000]}")
                        if entry.ret is not None: parts.append(f"Exit code: {entry.ret}")
                        outputs.append("\n".join(parts))

                    # Send files from this iteration
                    send_fails = []
                    for fp in _extract_sends(resp):
                        p = Path(fp.strip())
                        if not p.is_absolute(): p = WS / p
                        if str(p) not in _sent_paths and p.exists():
                            ok = await _send_file(bot, cid, fp)
                            if ok: _sent_paths.add(str(p))
                            else: send_fails.append(fp)
                        elif not p.exists():
                            send_fails.append(f"{fp} (not found)")
                    if send_fails:
                        outputs.append("[SEND_FILE ERRORS]\n" + "\n".join(send_fails))

                    combined = "\n\n".join(outputs)
                    memory_mgr.log_message(uid, "user", combined[:8000])
                    _cleanup_old_scripts()
                    continue

                # ── Handle SEND_FILE in final response ────────
                await _process_send_files(bot, cid, _extract_sends(resp), _sent_paths)

                # ── FINAL RESPONSE (no code blocks) ──────────
                clean = _clean_text(resp)
                if clean:
                    await _edit(bot, cid, mid, clean)
                elif mid:
                    await _edit(bot, cid, mid, "✅ Selesai.")
                return

            except Exception as e:
                log.error(f"Processing error: {e}", exc_info=True)
                mid = await _edit(bot, cid, mid,
                    _live(f"⚠️ Error: <code>{esc(str(e)[:200])}</code>\n🔄 Retry..."))
                await asyncio.sleep(2)
                if itr > 30:
                    await _edit(bot, cid, mid,
                        f"⚠️ <b>Terlalu banyak iterasi ({itr}).</b>\nSila cuba arahan yang lebih spesifik.")
                    return

    finally:
        session.busy = False
        session.cancel_event.clear()
        session.live_items = []
        session.task_desc = ""


async def _watch_script_async(entry: ScriptEntry, bot, cid: int, session: Session):
    """Watch background script and notify when done."""
    while entry.status == "running":
        if entry.proc and entry.proc.poll() is not None:
            break
        await asyncio.sleep(1)

    # Read final output
    try:
        out_p = SCRIPTS_DIR / f"{entry.sid}.stdout"
        err_p = SCRIPTS_DIR / f"{entry.sid}.stderr"
        entry.stdout = out_p.read_text("utf-8", errors="replace") if out_p.exists() else ""
        entry.stderr = err_p.read_text("utf-8", errors="replace") if err_p.exists() else entry.stderr
    except Exception:
        pass
    entry.ret = entry.proc.returncode if entry.proc else None
    entry.status = "done" if entry.ret == 0 else "error"
    entry.ended = datetime.datetime.now()
    _save_scripts_meta()

    stdout = (entry.stdout or "")[:3000]
    stderr = (entry.stderr or "")[:3000]

    if entry.status == "done":
        preview = stdout[:2000] if stdout else "Tiada output"
        await _send(bot, cid,
            f"✅ <b>Script selesai</b> — <code>{entry.sid}</code>\n\n"
            f"<b>Output:</b>\n<pre>{esc(preview)}</pre>")
    elif entry.status == "error":
        err_preview = stderr[:1000] if stderr else stdout[:1000] if stdout else "Unknown"
        await _send(bot, cid,
            f"❌ <b>Script gagal</b> — <code>{entry.sid}</code>\n\n"
            f"<b>Error:</b>\n<pre>{esc(err_preview)}</pre>\n\n"
            f"🔧 AI sedang analisa & baiki...")
        # Auto-fix if not busy
        if not session.busy:
            fix_content = (
                f"[BACKGROUND SCRIPT FAILED]\n"
                f"Script ID: {entry.sid}\nName: {entry.name}\n"
                f"Exit code: {entry.ret}\n"
                f"STDOUT:\n{stdout[:4000]}\nSTDERR:\n{stderr[:2000]}\n\n"
                f"Analyse the error, fix the script, and run it again."
            )
            asyncio.create_task(_process(bot, cid, session, fix_content))
    elif entry.status == "stopped":
        await _send(bot, cid, f"⛔ <b>Script dihentikan</b> — <code>{entry.sid}</code>")



# ═══════════════════════════════════════════════════════════════
#  SECTION 9: COMMAND HANDLERS
# ═══════════════════════════════════════════════════════════════

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = update.effective_user

    # Track user
    memory_mgr.track_user(uid, user.username or "", user.first_name or "", user.last_name or "")

    elapsed = time.time() - BOT_START_TIME
    remaining = max(0, RUN_DURATION - elapsed)

    # Format times
    e_h, e_r = divmod(int(elapsed), 3600)
    e_m, e_s = divmod(e_r, 60)
    r_h, r_r = divmod(int(remaining), 3600)
    r_m, r_s = divmod(r_r, 60)

    start_dt = datetime.datetime.fromtimestamp(BOT_START_TIME, tz=datetime.timezone.utc)
    restart_dt = datetime.datetime.fromtimestamp(BOT_START_TIME + RUN_DURATION, tz=datetime.timezone.utc)

    start_str = start_dt.strftime("%H:%M:%S UTC")
    restart_str = restart_dt.strftime("%H:%M:%S UTC")

    if _is_owner(update):
        # ═══ OWNER ADMIN DASHBOARD ═══
        mem_stats = memory_mgr.get_memory_stats(uid)
        global_stats = memory_mgr.get_global_stats()
        all_users = memory_mgr.get_all_users()
        running = sum(1 for e in _scripts.values() if e.status == "running")

        # Build user list
        user_lines = ""
        if all_users:
            user_lines = "\n👥 <b>USERS:</b>\n"
            for i, u in enumerate(all_users[:20], 1):
                name = u["first_name"] or "Unknown"
                uname = f"@{u['username']}" if u["username"] else ""
                is_owner_tag = " 👑" if u["user_id"] == OWNER_ID else ""
                last = u["last_seen"][:16] if u["last_seen"] else "?"
                user_lines += (
                    f"  {i}. <b>{esc(name)}</b> {esc(uname)}{is_owner_tag}\n"
                    f"     ID: <code>{u['user_id']}</code> | Msgs: {u['message_count']} | Last: {last}\n"
                )
            if len(all_users) > 20:
                user_lines += f"  ... +{len(all_users) - 20} lagi\n"

        text = f"""🤖 <b>AI Automation Bot v5.0</b> — <b>ADMIN PANEL</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⏱ <b>STATUS:</b>
  Started:    <code>{start_str}</code>
  Uptime:     <code>{e_h:02d}:{e_m:02d}:{e_s:02d}</code>
  Restart in: <code>{r_h:02d}:{r_m:02d}:{r_s:02d}</code> ⏳
  Next restart: <code>{restart_str}</code>

🧠 <b>AI ENGINE:</b> Groq LPU
  Primary:  <code>{PRIMARY_MODEL}</code>
            128K ctx | 64K output | 500 tps
  Fallback: <code>{FALLBACK_MODEL}</code>
            256K ctx | 16K output | 200 tps

📊 <b>GLOBAL STATS:</b>
  Total users:     {global_stats['total_users']}
  Total messages:  {global_stats['total_messages']}
  Total memories:  {global_stats['total_memories']}
  Total logs:      {global_stats['total_logs']}
  Total pins:      {global_stats['total_pins']}
  Total summaries: {global_stats['total_summaries']}
{user_lines}
💾 <b>YOUR MEMORY:</b>
  DB entries: {mem_stats['db_entries']} | Pinned: {mem_stats['pinned']}
  Log: {mem_stats['log_entries']} | Summaries: {mem_stats['summaries']}
  Files: {mem_stats['files']} | RAG: {'✅' if HAS_RAG else '❌'}

📂 <b>WORKSPACE:</b>
  Scripts: {len(_scripts)} total | {running} running

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 <b>CAPABILITIES:</b>
  ⚡ Execute Python 3.12 / Bash
  🌐 Web Search (5 engines)
  🔬 Deep Scan websites
  🤖 Sub-Agent parallel processing
  💾 5-layer persistent memory
  📦 Auto-install any package
  📁 File management + GoFile upload
  🔄 Auto-restart (cron every 5hr)

{CREDIT}"""

    else:
        # ═══ NORMAL USER VIEW ═══
        # Channel check
        is_member = await check_channel_member(ctx.bot, uid)
        if not is_member:
            await update.message.reply_text(
                _bq("⛔ <b>Sila join channel kami dulu untuk guna bot ini.</b>"),
                parse_mode=ParseMode.HTML,
                reply_markup=_join_channel_kb(),
            )
            return

        mem_stats = memory_mgr.get_memory_stats(uid)

        text = f"""🤖 <b>AI Automation Bot v5.0</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

👤 <b>Hi, {esc(user.first_name or 'User')}!</b>

⏱ <b>BOT STATUS:</b>
  Uptime:     <code>{e_h:02d}:{e_m:02d}:{e_s:02d}</code>
  Restart in: <code>{r_h:02d}:{r_m:02d}:{r_s:02d}</code> ⏳

🧠 <b>AI:</b> Powered by Groq LPU ⚡

💾 <b>MEMORY ANDA:</b>
  Memories:   {mem_stats['db_entries']}
  Pinned:     {mem_stats['pinned']}
  Log:        {mem_stats['log_entries']} messages
  Summaries:  {mem_stats['summaries']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 <b>APA BOLEH BUAT:</b>
  🗣 Tanya apa sahaja — AI akan jawab
  📌 Reply mana-mana mesej + tulis "pin" untuk simpan
  🧠 AI ingat semua perbualan anda

{CREDIT}"""

    # Build inline keyboard buttons (ganti text commands)
    if _is_owner(update):
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("⛔ Hentikan AI", callback_data="cmd_stop"),
             InlineKeyboardButton("📋 Scripts", callback_data="cmd_scripts")],
            [InlineKeyboardButton("🔄 Sesi Baru", callback_data="cmd_new"),
             InlineKeyboardButton("💾 Memory", callback_data="cmd_memory")],
        ])
    else:
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("⛔ Hentikan AI", callback_data="cmd_stop"),
             InlineKeyboardButton("🔄 Sesi Baru", callback_data="cmd_new")],
            [InlineKeyboardButton("💾 Memory", callback_data="cmd_memory")],
        ])
    await update.message.reply_text(_bq(text), parse_mode=ParseMode.HTML, reply_markup=kb)


async def cmd_stop(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    session = _sess(uid)

    if session.busy:
        kb = InlineKeyboardMarkup([[
            InlineKeyboardButton("⛔ Ya, hentikan", callback_data="stop_confirm"),
            InlineKeyboardButton("▶️ Teruskan", callback_data="stop_cancel"),
        ]])
        await update.message.reply_text(
            _bq("🧠 AI sedang memproses.\n\nAdakah anda pasti mahu hentikan?"),
            parse_mode=ParseMode.HTML, reply_markup=kb)
    else:
        await update.message.reply_text(_bq("ℹ️ AI tidak sedang berjalan."), parse_mode=ParseMode.HTML)


async def cmd_new(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    session = _sess(uid)
    if session.busy:
        session.cancel_event.set()
        await asyncio.sleep(1)

    memory_mgr.reset_session(uid)
    # Only owner can stop all scripts
    if _is_owner(update):
        _stop_all_scripts()
    await update.message.reply_text(
        _bq("🔄 <b>Sesi baru dimulakan.</b>\n\n"
        "💾 Memori kekal disimpan.\n"
        "🗑️ Conversation log dikosongkan.\n\n"
        "Sila mulakan perbualan baru!"),
        parse_mode=ParseMode.HTML)


async def cmd_scripts(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _is_owner(update):
        return
    with _scripts_lock:
        entries = [e for e in _scripts.values() if e.status == "running"]

    if not entries:
        await update.message.reply_text(
            _bq("📋 Tiada scripts aktif."), parse_mode=ParseMode.HTML)
        return

    entries.sort(key=lambda e: e.started, reverse=True)
    lines = [f"📋 <b>Scripts Aktif ({len(entries)})</b>\n"]
    for e in entries[:30]:
        dur = ""
        if e.started:
            dur = f" ({int((datetime.datetime.now() - e.started).total_seconds())}s)"
        lines.append(f"🟢 <code>{e.sid}</code> {esc(e.name[:30])} [{e.lang}]{dur}")

    await update.message.reply_text(_bq("\n".join(lines)),
                                     parse_mode=ParseMode.HTML)


async def cmd_memory(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    stats = memory_mgr.get_memory_stats(uid)
    memories = memory_mgr.query_memory(uid)
    pinned = memory_mgr.get_pinned(uid)

    text = f"""💾 <b>Memory Status</b>
━━━━━━━━━━━━━━━━━━━

📊 <b>Statistics:</b>
  DB entries:  {stats['db_entries']}
  Pinned:      {stats['pinned']}
  Log entries: {stats['log_entries']}
  Summaries:   {stats['summaries']}
  File memory: {stats['files']} files
  RAG engine:  {'✅ Active' if HAS_RAG else '❌ Not available'}

"""
    if pinned:
        text += "📌 <b>Pinned Messages:</b>\n"
        for p in pinned[:10]:
            text += f"  • {esc(p[:100])}\n"
        text += "\n"

    if memories:
        text += "🧠 <b>Recent Memories:</b>\n"
        for m in memories[:15]:
            text += f"  [{m['category']}] {esc(m['key'])}: {esc(m['value'][:80])}\n"

    kb = InlineKeyboardMarkup([[
        InlineKeyboardButton("🗑 Clear Session Log", callback_data="mem_clear_log"),
        InlineKeyboardButton("📌 Clear Pins", callback_data="mem_clear_pins"),
    ]])
    await update.message.reply_text(_bq(text), parse_mode=ParseMode.HTML, reply_markup=kb)


# ── Callback Handler ──────────────────────────────────────────

async def on_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    if not q:
        return
    await q.answer()
    uid = q.from_user.id
    data = q.data

    # Channel join verification callback
    if data == "check_joined":
        is_member = await check_channel_member(ctx.bot, uid)
        if is_member:
            await q.edit_message_text(
                _bq("✅ <b>Terima kasih! Anda dah join.</b>\n\n"
                "Sila hantar mesej untuk mula berbual dengan AI."),
                parse_mode=ParseMode.HTML,
            )
        else:
            await q.answer("❌ Anda belum join channel. Sila join dulu.", show_alert=True)
        return

    if data == "stop_confirm":
        session = _sess(uid)
        session.cancel_event.set()
        # Only owner stops all scripts
        if uid == OWNER_ID:
            _stop_all_scripts()
            await q.edit_message_text(_bq("⛔ <b>AI dihentikan.</b> Semua scripts stopped."),
                                       parse_mode=ParseMode.HTML)
        else:
            await q.edit_message_text(_bq("⛔ <b>AI dihentikan.</b>"),
                                       parse_mode=ParseMode.HTML)

    elif data == "stop_cancel":
        await q.edit_message_text(_bq("▶️ AI diteruskan."), parse_mode=ParseMode.HTML)

    elif data == "stop_all_scripts":
        # Dilumpuhkan — hanya /stop dan /new boleh hentikan scripts
        await q.answer("ℹ️ Gunakan /stop atau /new untuk hentikan scripts.", show_alert=True)

    elif data == "mem_clear_log":
        memory_mgr.reset_session(uid)
        await q.edit_message_text(_bq("🗑 Session log dikosongkan."), parse_mode=ParseMode.HTML)

    elif data == "cmd_stop":
        session = _sess(uid)
        if session.busy:
            kb = InlineKeyboardMarkup([[
                InlineKeyboardButton("⛔ Ya, hentikan", callback_data="stop_confirm"),
                InlineKeyboardButton("▶️ Teruskan", callback_data="stop_cancel"),
            ]])
            await q.edit_message_text(
                _bq("🧠 AI sedang memproses.\n\nAdakah anda pasti mahu hentikan?"),
                parse_mode=ParseMode.HTML, reply_markup=kb)
        else:
            await q.edit_message_text(
                _bq("ℹ️ AI tidak sedang berjalan."), parse_mode=ParseMode.HTML)

    elif data == "cmd_new":
        session = _sess(uid)
        if session.busy:
            session.cancel_event.set()
            await asyncio.sleep(1)
        memory_mgr.reset_session(uid)
        if uid == OWNER_ID:
            _stop_all_scripts()
        await q.edit_message_text(
            _bq("🔄 <b>Sesi baru dimulakan.</b>\n\n"
            "💾 Memori kekal disimpan.\n"
            "🗑️ Conversation log dikosongkan.\n\n"
            "Sila mulakan perbualan baru!"),
            parse_mode=ParseMode.HTML)

    elif data == "cmd_scripts":
        if uid != OWNER_ID:
            await q.answer("⛔ Owner sahaja.", show_alert=True)
            return
        with _scripts_lock:
            entries = [e for e in _scripts.values() if e.status == "running"]
        if not entries:
            await q.edit_message_text(
                _bq("📋 Tiada scripts aktif."), parse_mode=ParseMode.HTML)
        else:
            entries.sort(key=lambda e: e.started, reverse=True)
            lines = [f"📋 <b>Scripts Aktif ({len(entries)})</b>\n"]
            for e in entries[:30]:
                dur = ""
                if e.started:
                    dur = f" ({int((datetime.datetime.now() - e.started).total_seconds())}s)"
                lines.append(f"🟢 <code>{e.sid}</code> {esc(e.name[:30])} [{e.lang}]{dur}")
            await q.edit_message_text(
                _bq("\n".join(lines)), parse_mode=ParseMode.HTML)

    elif data == "cmd_memory":
        stats = memory_mgr.get_memory_stats(uid)
        memories = memory_mgr.query_memory(uid)
        pinned = memory_mgr.get_pinned(uid)
        text = f"""💾 <b>Memory Status</b>
━━━━━━━━━━━━━━━━━━━

📊 <b>Statistics:</b>
  DB entries:  {stats['db_entries']}
  Pinned:      {stats['pinned']}
  Log entries: {stats['log_entries']}
  Summaries:   {stats['summaries']}
  File memory: {stats['files']} files
  RAG engine:  {'✅ Active' if HAS_RAG else '❌ Not available'}

"""
        if pinned:
            text += "📌 <b>Pinned Messages:</b>\n"
            for p in pinned[:10]:
                text += f"  • {esc(p[:100])}\n"
            text += "\n"
        if memories:
            text += "🧠 <b>Recent Memories:</b>\n"
            for m in memories[:15]:
                text += f"  [{m['category']}] {esc(m['key'])}: {esc(m['value'][:80])}\n"
        kb = InlineKeyboardMarkup([[
            InlineKeyboardButton("🗑 Clear Session Log", callback_data="mem_clear_log"),
            InlineKeyboardButton("📌 Clear Pins", callback_data="mem_clear_pins"),
        ]])
        await q.edit_message_text(_bq(text), parse_mode=ParseMode.HTML, reply_markup=kb)

    elif data == "mem_clear_pins":
        with sqlite3.connect(str(DB_PATH)) as conn:
            conn.execute("DELETE FROM pinned_messages WHERE user_id=?", (uid,))
        await q.edit_message_text(_bq("📌 Semua pins dipadamkan."), parse_mode=ParseMode.HTML)



# ═══════════════════════════════════════════════════════════════
#  SECTION 10: MESSAGE HANDLERS
# ═══════════════════════════════════════════════════════════════

async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return
    uid = update.effective_user.id
    user = update.effective_user

    # Track user
    memory_mgr.track_user(uid, user.username or "", user.first_name or "", user.last_name or "")

    # Channel membership check (owner always passes)
    if uid != OWNER_ID:
        is_member = await check_channel_member(ctx.bot, uid)
        if not is_member:
            await update.message.reply_text(
                _bq("⛔ <b>Sila join channel kami dulu untuk guna bot ini.</b>"),
                parse_mode=ParseMode.HTML,
                reply_markup=_join_channel_kb(),
            )
            return

    session = _sess(uid)

    text = update.message.text or ""
    if not text.strip():
        return

    # Check if user is pinning a reply
    if text.strip().lower() in ("pin", "/pin") and update.message.reply_to_message:
        reply_text = update.message.reply_to_message.text or ""
        if reply_text:
            memory_mgr.pin_message(uid, reply_text[:2000], "user")
            await update.message.reply_text(_bq("📌 Mesej di-pin!"), parse_mode=ParseMode.HTML)
            return

    # Extract reply context
    reply_ctx = ""
    if update.message.reply_to_message:
        rm = update.message.reply_to_message
        if rm.text:
            reply_ctx = f"[User membalas mesej ini:]\n{rm.text[:2000]}\n\n"
        elif rm.caption:
            reply_ctx = f"[User membalas mesej ini (caption):]\n{rm.caption[:1000]}\n\n"
        if rm.document:
            reply_ctx += f"[File: {rm.document.file_name}]\n"

    # Handle busy state
    if session.busy:
        if session.is_stale_busy(300):
            session.busy = False
            session.cancel_event.clear()
        else:
            # Natural language stop detection
            _stop_words = [
                "matikan", "stop", "berhenti", "cancel", "hentikan",
                "diam", "tutup", "abort", "halt", "batalkan",
                "off", "tamat", "sudah", "enough", "quit",
            ]
            lower = text.strip().lower()
            if any(sw in lower for sw in _stop_words):
                session.cancel_event.set()
                await update.message.reply_text(
                    _bq("⛔ <b>Menghentikan AI proses...</b>\n"
                    "ℹ️ Scripts tidak dihentikan. Guna /stop atau /new untuk hentikan scripts."),
                    parse_mode=ParseMode.HTML,
                )
                return
            await update.message.reply_text(
                _bq("⏳ AI sedang memproses. Sila tunggu, hantar /stop, "
                "atau taip <b>\"matikan\"</b> untuk hentikan."),
                parse_mode=ParseMode.HTML,
            )
            return

    user_content = reply_ctx + text
    asyncio.create_task(_process(ctx.bot, update.effective_chat.id, session, user_content))


async def on_file(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return
    uid = update.effective_user.id
    user = update.effective_user

    # Track user
    memory_mgr.track_user(uid, user.username or "", user.first_name or "", user.last_name or "")

    # Channel membership check (owner always passes)
    if uid != OWNER_ID:
        is_member = await check_channel_member(ctx.bot, uid)
        if not is_member:
            await update.message.reply_text(
                _bq("⛔ <b>Sila join channel kami dulu untuk guna bot ini.</b>"),
                parse_mode=ParseMode.HTML,
                reply_markup=_join_channel_kb(),
            )
            return

    session = _sess(uid)

    if session.busy:
        if session.is_stale_busy(300):
            session.busy = False
            session.cancel_event.clear()
        else:
            await update.message.reply_text(
                _bq("⏳ AI sedang memproses. Sila tunggu atau guna /stop."),
                parse_mode=ParseMode.HTML,
            )
            return

    msg = update.message
    file_obj = (msg.document or msg.photo[-1] if msg.photo else
                msg.audio or msg.video or msg.voice or
                msg.video_note or msg.sticker or msg.animation)
    if file_obj is None:
        return

    # Download file
    try:
        tg_file = await ctx.bot.get_file(file_obj.file_id)
        fname = getattr(file_obj, "file_name", None)
        if not fname:
            ext = mimetypes.guess_extension(getattr(file_obj, "mime_type", "") or "") or ""
            fname = f"upload_{uuid.uuid4().hex[:6]}{ext}"
        dest = UPLOADS_DIR / fname
        await tg_file.download_to_drive(str(dest))
    except Exception as e:
        await _send(ctx.bot, msg.chat_id, f"⚠️ Download gagal: {esc(str(e)[:200])}")
        return

    caption = msg.caption or ""
    fsize_kb = dest.stat().st_size / 1024

    # Handle ZIP files
    if fname.lower().endswith(('.zip', '.rar', '.7z')):
        ext_dir = UPLOADS_DIR / Path(fname).stem
        files_extracted, auth_files, err = _auto_extract_zip(str(dest), ext_dir)
        if err:
            await _send(ctx.bot, msg.chat_id, f"⚠️ Extract gagal: {esc(err)}")
            return
        file_info = (
            f"[ZIP File Received]\n"
            f"File: {fname} ({fsize_kb:.1f}KB)\n"
            f"Extracted to: {ext_dir}\n"
            f"Files ({len(files_extracted or [])}):\n" +
            "\n".join(f"  {f}" for f in (files_extracted or [])[:30]) + "\n"
        )
        if auth_files:
            file_info += f"Auth files detected: {', '.join(auth_files)}\n"
        user_input = file_info + (f"\nUser message: {caption}" if caption else
                                   "\nUser uploaded a ZIP. Analyze the contents.")
    # Handle code files
    elif fname.lower().endswith(('.py', '.js', '.ts', '.sh', '.bash', '.json', '.yaml',
                                  '.yml', '.xml', '.html', '.css', '.md', '.txt', '.csv',
                                  '.log', '.env', '.cfg', '.conf', '.ini', '.toml')):
        try:
            content = dest.read_text("utf-8", errors="replace")
        except Exception:
            content = "(binary file)"
        user_input = (
            f"[File Received]\nFile: {fname} ({fsize_kb:.1f}KB)\n"
            f"Path: {dest}\n\nContents:\n```\n{content[:10000]}\n```\n\n"
            + (f"User message: {caption}" if caption else
               f"User sent the file {fname}. Analyze it.")
        )
    # Handle cookie files
    elif _is_netscape_cookie(str(dest)):
        cookie_dest = _load_cookies_to_workspace(str(dest))
        user_input = (
            f"[Cookie File Received]\n"
            f"File: {fname}\nLoaded to: {cookie_dest}\n"
            f"Cookies are now available for web scraping.\n"
            + (f"User message: {caption}" if caption else
               "User uploaded cookies. Confirm they're loaded.")
        )
    else:
        user_input = (
            f"[File Received]\nFile: {fname} ({fsize_kb:.1f}KB)\n"
            f"Path: {dest}\n"
            + (f"User message: {caption}" if caption else
               f"User sent file {fname}. Acknowledge it.")
        )

    asyncio.create_task(_process(ctx.bot, msg.chat_id, session, user_input))


# ═══════════════════════════════════════════════════════════════
#  SECTION 11: AUTO-RESTART & MAIN
# ═══════════════════════════════════════════════════════════════

async def _auto_restart_watcher(app: Application):
    """Background task: graceful shutdown 35s before RUN_DURATION ends.
    Cron schedule handles starting the next run automatically."""
    while True:
        elapsed = time.time() - BOT_START_TIME
        remaining = RUN_DURATION - elapsed

        if remaining <= RESTART_BUFFER:
            log.info("⏰ Time's up! Saving state and shutting down gracefully...")

            # Calculate next cron run time (every 5 hours: 0,5,10,15,20 UTC)
            now_utc = datetime.datetime.now(datetime.timezone.utc)
            cron_hours = [0, 5, 10, 15, 20]
            next_h = None
            for h in cron_hours:
                if h > now_utc.hour:
                    next_h = h
                    break
            if next_h is None:
                next_h = cron_hours[0]  # wrap to next day
            next_cron = now_utc.replace(hour=next_h, minute=0, second=0, microsecond=0)
            if next_cron <= now_utc:
                next_cron += datetime.timedelta(days=1)
            wait_minutes = int((next_cron - now_utc).total_seconds() / 60)

            # Notify owner
            try:
                await app.bot.send_message(
                    chat_id=OWNER_ID,
                    text=_bq(
                        "🔄 <b>Auto-restart dalam 35s...</b>\n\n"
                        f"⏱ Uptime: {int(elapsed)}s\n"
                        "💾 Menyimpan state...\n"
                        f"⏰ Cron restart seterusnya: <code>{next_cron.strftime('%H:%M UTC')}</code> (~{wait_minutes} min)\n"
                        "🔁 Bot akan hidup semula secara automatik."
                    ),
                    parse_mode=ParseMode.HTML,
                )
            except Exception as e:
                log.warning(f"Failed to notify owner about restart: {e}")

            # Save all state
            _save_scripts_meta()

            log.info("💾 State saved. Waiting 30s then exiting...")
            await asyncio.sleep(30)

            # Stop polling and exit
            try:
                await app.stop()
            except Exception:
                pass
            os._exit(0)

        # Check every 10 seconds
        await asyncio.sleep(10)


async def post_init(app: Application):
    """Runs after application is initialized."""
    _load_scripts_meta()
    log.info("Bot post_init complete — scripts meta loaded.")

    # Start auto-restart watcher
    asyncio.create_task(_auto_restart_watcher(app))
    log.info(f"Auto-restart watcher started (duration: {RUN_DURATION}s, buffer: {RESTART_BUFFER}s)")

    log.info(f"Bot v5.0 ready — Primary: {PRIMARY_MODEL}, Fallback: {FALLBACK_MODEL}")


def main():
    if not TOKEN:
        log.error("TELEGRAM_BOT_TOKEN not set!")
        sys.exit(1)
    if not GROQ_KEY:
        log.error("GROQ_API_KEY not set!")
        sys.exit(1)
    if not OWNER_ID:
        log.error("OWNER_ID not set!")
        sys.exit(1)

    log.info("═" * 60)
    log.info("  AI AUTOMATION BOT v5.0")
    log.info(f"  Primary:  {PRIMARY_MODEL}")
    log.info(f"  Fallback: {FALLBACK_MODEL}")
    log.info(f"  Duration: {RUN_DURATION}s ({RUN_DURATION/3600:.1f}h)")
    log.info(f"  Python:   {sys.version}")
    log.info("═" * 60)

    app = (
        Application.builder()
        .token(TOKEN)
        .post_init(post_init)
        .concurrent_updates(True)
        .read_timeout(30)
        .write_timeout(30)
        .connect_timeout(30)
        .pool_timeout(30)
        .build()
    )

    # Commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("stop", cmd_stop))
    app.add_handler(CommandHandler("new", cmd_new))
    app.add_handler(CommandHandler("scripts", cmd_scripts))
    app.add_handler(CommandHandler("memory", cmd_memory))

    # Callbacks
    app.add_handler(CallbackQueryHandler(on_callback))

    # Text messages
    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND, on_text))

    # File/media messages
    app.add_handler(MessageHandler(
        filters.Document.ALL | filters.PHOTO | filters.AUDIO |
        filters.VIDEO | filters.VOICE | filters.VIDEO_NOTE |
        filters.Sticker.ALL | filters.ANIMATION,
        on_file))

    log.info("Starting polling...")
    app.run_polling(
        drop_pending_updates=True,
        close_loop=False,
        allowed_updates=Update.ALL_TYPES,
    )


if __name__ == "__main__":
    main()

