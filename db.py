"""SQLite storage for escalation / support-request logging.

Used by the FastAPI backend (app.py) and the Streamlit frontend
(streamlit_app.py) so human-handoff requests are persisted and can
be reviewed in an admin view.
"""
import sqlite3
from datetime import datetime
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "carepal.db")


def _get_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the escalations table if it doesn't exist."""
    conn = _get_conn()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS escalations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT,
            signal TEXT,             -- 'NOT_FOUND' | 'ESCALATE' | NULL
            name TEXT,
            phone TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def log_event(
    question: str,
    answer: str = "",
    signal: str | None = None,
    name: str | None = None,
    phone: str | None = None,
) -> int:
    """Insert an escalation / support event. Returns the new row id."""
    conn = _get_conn()
    cur = conn.execute(
        """
        INSERT INTO escalations (question, answer, signal, name, phone, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            question,
            answer,
            signal,
            name,
            phone,
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def get_events(limit: int = 200) -> list[dict]:
    """Return the most recent escalation events, newest first."""
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT id, question, answer, signal, name, phone, created_at
        FROM escalations
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_unanswered_count() -> int:
    """Count events where signal is NOT_FOUND or ESCALATE."""
    conn = _get_conn()
    row = conn.execute(
        """
        SELECT COUNT(*) AS c FROM escalations
        WHERE signal IN ('NOT_FOUND', 'ESCALATE')
        """
    ).fetchone()
    conn.close()
    return row["c"] if row else 0
