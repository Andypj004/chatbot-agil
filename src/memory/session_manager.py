"""Persistent session manager for conversation history using SQLite."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import sqlite3
import threading


class SessionManager:
    """Store and query conversations in a local SQLite database."""

    def __init__(self, db_path: str):
        self._lock = threading.Lock()
        self._db_path = db_path
        db_parent = Path(db_path).parent
        db_parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(tz=timezone.utc).isoformat()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    text TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    provider TEXT,
                    model TEXT,
                    used_rag INTEGER,
                    sources_json TEXT,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_session_created
                ON messages (session_id, created_at)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_sessions_updated
                ON sessions (updated_at)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    filename TEXT,
                    source TEXT,
                    file_type TEXT,
                    file_hash TEXT,
                    uploaded_at TEXT NOT NULL,
                    UNIQUE(session_id, document_id),
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_session_documents_session
                ON session_documents (session_id, uploaded_at)
                """
            )
            conn.commit()

    def create_session(self, session_id: str, title: Optional[str] = None) -> None:
        now = self._now_iso()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO sessions (session_id, title, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(session_id) DO NOTHING
                    """,
                    (session_id, title, now, now),
                )
                conn.commit()

    def append_message(
        self,
        session_id: str,
        role: str,
        text: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        used_rag: Optional[bool] = None,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        now = self._now_iso()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO sessions (session_id, title, created_at, updated_at)
                    VALUES (?, NULL, ?, ?)
                    ON CONFLICT(session_id) DO NOTHING
                    """,
                    (session_id, now, now),
                )
                conn.execute(
                    """
                    INSERT INTO messages
                    (session_id, role, text, created_at, provider, model, used_rag, sources_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        role,
                        text,
                        now,
                        provider,
                        model,
                        int(used_rag) if used_rag is not None else None,
                        json.dumps(sources or []),
                    ),
                )
                conn.execute(
                    "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                    (now, session_id),
                )
                conn.commit()

    def get_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        query = """
            SELECT id, session_id, role, text, created_at, provider, model, used_rag, sources_json
            FROM messages
            WHERE session_id = ?
            ORDER BY id ASC
        """
        params: List[Any] = [session_id]

        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        messages: List[Dict[str, Any]] = []
        for row in rows:
            messages.append(
                {
                    "id": row["id"],
                    "session_id": row["session_id"],
                    "role": row["role"],
                    "text": row["text"],
                    "created_at": row["created_at"],
                    "provider": row["provider"],
                    "model": row["model"],
                    "used_rag": bool(row["used_rag"]) if row["used_rag"] is not None else None,
                    "sources": json.loads(row["sources_json"] or "[]"),
                }
            )
        return messages

    def list_sessions(self, limit: int = 50, query: Optional[str] = None) -> List[Dict[str, Any]]:
        sql = """
            SELECT
                s.session_id,
                s.title,
                s.created_at,
                s.updated_at,
                COUNT(m.id) AS message_count,
                (
                    SELECT text
                    FROM messages lm
                    WHERE lm.session_id = s.session_id
                    ORDER BY lm.id DESC
                    LIMIT 1
                ) AS last_message
            FROM sessions s
            LEFT JOIN messages m ON m.session_id = s.session_id
        """
        params: List[Any] = []

        if query:
            sql += (
                " WHERE s.session_id LIKE ?"
                " OR s.title LIKE ?"
                " OR EXISTS (SELECT 1 FROM messages sm WHERE sm.session_id = s.session_id AND sm.text LIKE ?)"
            )
            q = f"%{query}%"
            params.extend([q, q, q])

        sql += " GROUP BY s.session_id ORDER BY s.updated_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [
            {
                "session_id": row["session_id"],
                "title": row["title"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "message_count": row["message_count"],
                "last_message": row["last_message"] or "",
            }
            for row in rows
        ]

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT session_id, title, created_at, updated_at
                FROM sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()

        if not row:
            return None

        return {
            "session_id": row["session_id"],
            "title": row["title"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def update_session_title(self, session_id: str, title: str) -> bool:
        now = self._now_iso()
        normalized_title = title.strip()
        if not normalized_title:
            return False

        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    UPDATE sessions
                    SET title = ?, updated_at = ?
                    WHERE session_id = ?
                    """,
                    (normalized_title, now, session_id),
                )
                conn.commit()

        return cursor.rowcount > 0

    def get_message_count(self, session_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(1) AS count FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return int(row["count"]) if row else 0

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute("DELETE FROM session_documents WHERE session_id = ?", (session_id,))
                conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
                conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                conn.commit()

    def add_session_document(
        self,
        session_id: str,
        document_id: str,
        filename: str,
        source: str,
        file_type: str,
        file_hash: str,
    ) -> None:
        now = self._now_iso()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO sessions (session_id, title, created_at, updated_at)
                    VALUES (?, NULL, ?, ?)
                    ON CONFLICT(session_id) DO NOTHING
                    """,
                    (session_id, now, now),
                )
                conn.execute(
                    """
                    INSERT INTO session_documents
                    (session_id, document_id, filename, source, file_type, file_hash, uploaded_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id, document_id)
                    DO UPDATE SET
                        filename = excluded.filename,
                        source = excluded.source,
                        file_type = excluded.file_type,
                        file_hash = excluded.file_hash,
                        uploaded_at = excluded.uploaded_at
                    """,
                    (session_id, document_id, filename, source, file_type, file_hash, now),
                )
                conn.execute(
                    "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                    (now, session_id),
                )
                conn.commit()

    def list_session_documents(self, session_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT document_id, filename, source, file_type, file_hash, uploaded_at
                FROM session_documents
                WHERE session_id = ?
                ORDER BY uploaded_at DESC
                """,
                (session_id,),
            ).fetchall()

        return [
            {
                "document_id": row["document_id"],
                "filename": row["filename"],
                "source": row["source"],
                "file_type": row["file_type"],
                "file_hash": row["file_hash"],
                "scope": "session_chat",
                "session_id": session_id,
                "uploaded_at": row["uploaded_at"],
            }
            for row in rows
        ]

    def get_session_documents_by_ids(self, session_id: str, document_ids: List[str]) -> List[Dict[str, Any]]:
        if not document_ids:
            return self.list_session_documents(session_id)

        placeholders = ",".join(["?"] * len(document_ids))
        query = f"""
            SELECT document_id, filename, source, file_type, file_hash, uploaded_at
            FROM session_documents
            WHERE session_id = ? AND document_id IN ({placeholders})
            ORDER BY uploaded_at DESC
        """
        params: List[Any] = [session_id, *document_ids]

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return [
            {
                "document_id": row["document_id"],
                "filename": row["filename"],
                "source": row["source"],
                "file_type": row["file_type"],
                "file_hash": row["file_hash"],
                "scope": "session_chat",
                "session_id": session_id,
                "uploaded_at": row["uploaded_at"],
            }
            for row in rows
        ]

    def remove_session_document(self, session_id: str, document_id: str) -> bool:
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    DELETE FROM session_documents
                    WHERE session_id = ? AND document_id = ?
                    """,
                    (session_id, document_id),
                )
                conn.commit()
        return cursor.rowcount > 0

    def clear_session_documents(self, session_id: str) -> int:
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    "DELETE FROM session_documents WHERE session_id = ?",
                    (session_id,),
                )
                conn.commit()
        return cursor.rowcount
