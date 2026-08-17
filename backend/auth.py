from __future__ import annotations

import secrets
import sqlite3
import time
from typing import Optional

from werkzeug.security import check_password_hash, generate_password_hash

DB_PATH = "users.db"
SESSION_TTL_SECONDS = 60 * 60 * 24 * 7  # 7 days


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            expires_at REAL NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def register_user(username: str, password: str) -> bool:
    """
    Create a new user account. Returns False if the username is already taken
    or the inputs are invalid.
    """
    username = (username or "").strip()
    if not username or not password:
        return False

    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, generate_password_hash(password)),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def verify_user(username: str, password: str) -> bool:
    """Check a username/password combination against the stored hash."""
    username = (username or "").strip()
    if not username or not password:
        return False

    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT password_hash FROM users WHERE username = ?", (username,)
        ).fetchone()
    finally:
        conn.close()

    if not row:
        return False
    return check_password_hash(row[0], password)


def user_exists(username: str) -> bool:
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT 1 FROM users WHERE username = ?", (username,)
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def make_thread_id(username: str, raw_id: Optional[str] = None) -> str:
    """
    Build a namespaced thread_id so every thread is tied to its owner.
    Format: "<username>::<uuid-or-raw_id>"
    """
    import uuid

    suffix = raw_id or str(uuid.uuid4())
    return f"{username}::{suffix}"


def thread_owner(thread_id: str) -> Optional[str]:
    """Extract the owning username from a namespaced thread_id, if present."""
    if thread_id and "::" in thread_id:
        return thread_id.split("::", 1)[0]
    return None


# -------------------
# Sessions (keeps login alive across a browser refresh)
# -------------------
def create_session(username: str) -> str:
    """Issue a new session token for a user and persist it with an expiry."""
    conn = _get_conn()
    token = secrets.token_urlsafe(32)
    expires_at = time.time() + SESSION_TTL_SECONDS
    try:
        conn.execute(
            "INSERT INTO sessions (token, username, expires_at) VALUES (?, ?, ?)",
            (token, username, expires_at),
        )
        conn.commit()
    finally:
        conn.close()
    return token


def validate_session(token: Optional[str]) -> Optional[str]:
    """Return the username for a valid, unexpired session token, else None."""
    if not token:
        return None

    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT username, expires_at FROM sessions WHERE token = ?", (token,)
        ).fetchone()
    finally:
        conn.close()

    if not row:
        return None

    username, expires_at = row
    if expires_at < time.time():
        delete_session(token)
        return None
    return username


def delete_session(token: Optional[str]) -> None:
    """Invalidate a session token (used on logout)."""
    if not token:
        return
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()
    finally:
        conn.close()