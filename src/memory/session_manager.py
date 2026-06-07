"""Persistent session manager for conversation history using SQLite."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
import sqlite3
import threading
import unicodedata
from uuid import uuid4

from src.memory.concept_tracker import extract_concepts
from src.core.security import (
    assess_agile_level,
    agile_level_label,
    generate_token,
    hash_password,
    hash_token,
    normalize_agile_level,
    verify_password,
)


class SessionManager:
    """Store and query conversations in a local SQLite database."""

    def __init__(self, db_path: str):
        self._lock = threading.Lock()
        self._requested_db_path = Path(db_path)
        self._db_path = self._resolve_db_path(self._requested_db_path)
        db_parent = self._db_path.parent
        db_parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def _resolve_db_path(self, db_path: Path) -> Path:
        """Return a writable DB path, copying the existing DB when needed."""
        if not db_path.exists():
            return db_path

        if os.access(db_path, os.W_OK):
            return db_path

        fallback_path = db_path.with_name(f"{db_path.stem}.writable{db_path.suffix}")
        try:
            shutil.copy2(db_path, fallback_path)
            return fallback_path
        except Exception:
            return db_path

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
                    user_id TEXT,
                    title TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            session_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
            }
            if "user_id" not in session_columns:
                conn.execute("ALTER TABLE sessions ADD COLUMN user_id TEXT")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE,
                    full_name TEXT NOT NULL,
                    account_type TEXT NOT NULL,
                    knowledge_level INTEGER NOT NULL,
                    agile_adoption_level INTEGER NOT NULL,
                    agile_adoption_label TEXT NOT NULL,
                    questionnaire_answers_json TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    password_salt TEXT NOT NULL,
                    auth_token_hash TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_login_at TEXT
                )
                """
            )
            user_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()
            }
            if "full_name" not in user_columns:
                conn.execute("ALTER TABLE users ADD COLUMN full_name TEXT")
            if "account_type" not in user_columns:
                conn.execute("ALTER TABLE users ADD COLUMN account_type TEXT")
            if "knowledge_level" not in user_columns:
                conn.execute(
                    "ALTER TABLE users ADD COLUMN knowledge_level INTEGER NOT NULL DEFAULT 1"
                )
            if "agile_adoption_level" not in user_columns:
                conn.execute(
                    "ALTER TABLE users ADD COLUMN agile_adoption_level INTEGER NOT NULL DEFAULT 1"
                )
            if "agile_adoption_label" not in user_columns:
                conn.execute(
                    "ALTER TABLE users ADD COLUMN agile_adoption_label TEXT NOT NULL DEFAULT 'Ninguno'"
                )
            if "questionnaire_answers_json" not in user_columns:
                conn.execute(
                    "ALTER TABLE users ADD COLUMN questionnaire_answers_json TEXT NOT NULL DEFAULT '[]'"
                )
            if "password_hash" not in user_columns:
                conn.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")
            if "password_salt" not in user_columns:
                conn.execute("ALTER TABLE users ADD COLUMN password_salt TEXT")
            if "auth_token_hash" not in user_columns:
                conn.execute("ALTER TABLE users ADD COLUMN auth_token_hash TEXT")
            if "created_at" not in user_columns:
                conn.execute("ALTER TABLE users ADD COLUMN created_at TEXT")
            if "updated_at" not in user_columns:
                conn.execute("ALTER TABLE users ADD COLUMN updated_at TEXT")
            if "last_login_at" not in user_columns:
                conn.execute("ALTER TABLE users ADD COLUMN last_login_at TEXT")
            if "is_admin" not in user_columns:
                conn.execute(
                    "ALTER TABLE users ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0"
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
            existing_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(messages)").fetchall()
            }
            if "attachments_json" not in existing_columns:
                conn.execute("ALTER TABLE messages ADD COLUMN attachments_json TEXT")
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
                CREATE INDEX IF NOT EXISTS idx_sessions_user_updated
                ON sessions (user_id, updated_at)
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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_concepts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    concept TEXT NOT NULL,
                    mention_count INTEGER NOT NULL DEFAULT 1,
                    first_mentioned TEXT NOT NULL,
                    last_mentioned TEXT NOT NULL,
                    UNIQUE(session_id, concept),
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_session_concepts_session
                ON session_concepts (session_id, last_mentioned)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_citations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    citation_key TEXT NOT NULL,
                    document_id TEXT,
                    filename TEXT,
                    source TEXT,
                    page INTEGER,
                    section TEXT,
                    scope TEXT,
                    excerpt TEXT,
                    mention_count INTEGER NOT NULL DEFAULT 1,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    UNIQUE(session_id, citation_key),
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_session_citations_session
                ON session_citations (session_id, last_seen)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS form_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    form_id TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(session_id, form_id),
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_form_states_session
                ON form_states (session_id, updated_at)
                """
            )
            conn.commit()

    @staticmethod
    def _sanitize_email(email: str) -> str:
        return " ".join(str(email or "").strip().lower().split())

    @staticmethod
    def _serialize_user_row(row: sqlite3.Row | None) -> Optional[Dict[str, Any]]:
        if row is None:
            return None

        agile_level = int(row["agile_adoption_level"] or 1)
        knowledge_level = int(row["knowledge_level"] or 1)
        return {
            "user_id": row["user_id"],
            "email": row["email"],
            "full_name": row["full_name"],
            "account_type": row["account_type"],
            "knowledge_level": knowledge_level,
            "agile_adoption_level": agile_level,
            "agile_adoption_label": row["agile_adoption_label"]
            or agile_level_label(agile_level),
            "questionnaire_answers": json.loads(
                row["questionnaire_answers_json"] or "[]"
            ),
            "is_admin": bool(row["is_admin"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "last_login_at": row["last_login_at"],
        }

    def create_session(
        self,
        session_id: str,
        title: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> None:
        now = self._now_iso()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO sessions (session_id, user_id, title, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        user_id = COALESCE(sessions.user_id, excluded.user_id),
                        title = COALESCE(sessions.title, excluded.title),
                        updated_at = excluded.updated_at
                    """,
                    (session_id, user_id, title, now, now),
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
        attachments: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        now = self._now_iso()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO sessions (session_id, user_id, title, created_at, updated_at)
                    VALUES (?, NULL, NULL, ?, ?)
                    ON CONFLICT(session_id) DO NOTHING
                    """,
                    (session_id, now, now),
                )
                conn.execute(
                    """
                    INSERT INTO messages
                    (session_id, role, text, created_at, provider, model, used_rag, sources_json, attachments_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        json.dumps(attachments or []),
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
            SELECT id, session_id, role, text, created_at, provider, model, used_rag, sources_json, attachments_json
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
                    "used_rag": (
                        bool(row["used_rag"]) if row["used_rag"] is not None else None
                    ),
                    "sources": json.loads(row["sources_json"] or "[]"),
                    "attachments": json.loads(row["attachments_json"] or "[]"),
                }
            )
        return messages

    def list_sessions(
        self,
        limit: int = 50,
        query: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        sql = """
            SELECT
                s.session_id,
                s.user_id,
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
        conditions: List[str] = []

        if user_id is None:
            conditions.append("s.user_id IS NULL")
        else:
            conditions.append("s.user_id = ?")
            params.append(user_id)

        if query:
            conditions.append(
                "(s.session_id LIKE ? OR s.title LIKE ? OR EXISTS (SELECT 1 FROM messages sm WHERE sm.session_id = s.session_id AND sm.text LIKE ?))"
            )
            q = f"%{query}%"
            params.extend([q, q, q])

        if conditions:
            sql += " WHERE " + " AND ".join(conditions)

        sql += " GROUP BY s.session_id ORDER BY s.updated_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [
            {
                "session_id": row["session_id"],
                "user_id": row["user_id"],
                "title": row["title"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "message_count": row["message_count"],
                "last_message": row["last_message"] or "",
            }
            for row in rows
        ]

    def get_session(
        self, session_id: str, user_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT session_id, user_id, title, created_at, updated_at
                FROM sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()

        if not row:
            return None

        owner_id = row["user_id"]
        if user_id is None:
            if owner_id is not None:
                return None
        elif owner_id != user_id:
            return None

        return {
            "session_id": row["session_id"],
            "user_id": row["user_id"],
            "title": row["title"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def get_session_record(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return a raw session row without applying ownership filtering."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT session_id, user_id, title, created_at, updated_at
                FROM sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()

        if not row:
            return None

        return {
            "session_id": row["session_id"],
            "user_id": row["user_id"],
            "title": row["title"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def upsert_form_state(
        self, session_id: str, form_id: str, state: Dict[str, Any]
    ) -> None:
        now = self._now_iso()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO sessions (session_id, user_id, title, created_at, updated_at)
                    VALUES (?, NULL, NULL, ?, ?)
                    ON CONFLICT(session_id) DO NOTHING
                    """,
                    (session_id, now, now),
                )
                conn.execute(
                    """
                    INSERT INTO form_states (session_id, form_id, state_json, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(session_id, form_id)
                    DO UPDATE SET state_json = excluded.state_json,
                                  updated_at = excluded.updated_at
                    """,
                    (session_id, form_id, json.dumps(state), now),
                )
                conn.execute(
                    "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                    (now, session_id),
                )
                conn.commit()

    def get_form_state(self, session_id: str, form_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT state_json
                FROM form_states
                WHERE session_id = ? AND form_id = ?
                """,
                (session_id, form_id),
            ).fetchone()

        if not row:
            return None

        try:
            return json.loads(row["state_json"])
        except Exception:
            return None

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

    def record_concepts(self, session_id: str, *texts: str) -> list[str]:
        now = self._now_iso()
        concepts: list[str] = []
        for text in texts:
            concepts.extend(extract_concepts(text))

        unique_concepts = list(dict.fromkeys(concepts))
        if not unique_concepts:
            return []

        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO sessions (session_id, user_id, title, created_at, updated_at)
                    VALUES (?, NULL, NULL, ?, ?)
                    ON CONFLICT(session_id) DO NOTHING
                    """,
                    (session_id, now, now),
                )
                for concept in unique_concepts:
                    existing = conn.execute(
                        """
                        SELECT mention_count
                        FROM session_concepts
                        WHERE session_id = ? AND concept = ?
                        """,
                        (session_id, concept),
                    ).fetchone()
                    if existing:
                        conn.execute(
                            """
                            UPDATE session_concepts
                            SET mention_count = mention_count + 1,
                                last_mentioned = ?
                            WHERE session_id = ? AND concept = ?
                            """,
                            (now, session_id, concept),
                        )
                    else:
                        conn.execute(
                            """
                            INSERT INTO session_concepts
                            (session_id, concept, mention_count, first_mentioned, last_mentioned)
                            VALUES (?, ?, 1, ?, ?)
                            """,
                            (session_id, concept, now, now),
                        )
                conn.execute(
                    "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                    (now, session_id),
                )
                conn.commit()

        return unique_concepts

    @staticmethod
    def _normalize_citation_text(value: Any) -> str:
        text = unicodedata.normalize("NFKD", str(value or "").lower())
        text = "".join(char for char in text if not unicodedata.combining(char))
        return " ".join(text.split())

    @classmethod
    def _citation_key(cls, citation: Dict[str, Any]) -> str:
        payload = [
            cls._normalize_citation_text(citation.get("document_id")),
            cls._normalize_citation_text(citation.get("filename")),
            cls._normalize_citation_text(citation.get("source")),
            cls._normalize_citation_text(citation.get("page")),
            cls._normalize_citation_text(citation.get("section")),
            cls._normalize_citation_text(citation.get("scope") or "global_rag"),
            cls._normalize_citation_text(citation.get("excerpt")),
        ]
        return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))

    def record_source_citations(
        self, session_id: str, citations: List[Dict[str, Any]]
    ) -> List[str]:
        now = self._now_iso()
        unique_citations: list[tuple[str, Dict[str, Any]]] = []
        seen_keys: set[str] = set()

        for citation in citations:
            key = self._citation_key(citation)
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            unique_citations.append((key, citation))

        if not unique_citations:
            return []

        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO sessions (session_id, user_id, title, created_at, updated_at)
                    VALUES (?, NULL, NULL, ?, ?)
                    ON CONFLICT(session_id) DO NOTHING
                    """,
                    (session_id, now, now),
                )
                for key, citation in unique_citations:
                    page = citation.get("page")
                    try:
                        page_value = (
                            int(page)
                            if page is not None and str(page).isdigit()
                            else None
                        )
                    except Exception:
                        page_value = None

                    existing = conn.execute(
                        """
                        SELECT mention_count
                        FROM session_citations
                        WHERE session_id = ? AND citation_key = ?
                        """,
                        (session_id, key),
                    ).fetchone()
                    if existing:
                        conn.execute(
                            """
                            UPDATE session_citations
                            SET mention_count = mention_count + 1,
                                last_seen = ?
                            WHERE session_id = ? AND citation_key = ?
                            """,
                            (now, session_id, key),
                        )
                    else:
                        conn.execute(
                            """
                            INSERT INTO session_citations
                            (session_id, citation_key, document_id, filename, source, page, section, scope, excerpt, mention_count, first_seen, last_seen)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                            """,
                            (
                                session_id,
                                key,
                                citation.get("document_id"),
                                citation.get("filename"),
                                citation.get("source"),
                                page_value,
                                citation.get("section"),
                                citation.get("scope") or "global_rag",
                                citation.get("excerpt"),
                                now,
                                now,
                            ),
                        )
                conn.execute(
                    "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                    (now, session_id),
                )
                conn.commit()

        return [key for key, _citation in unique_citations]

    def get_recent_source_citation_keys(
        self, session_id: str, limit: int = 25
    ) -> List[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT citation_key
                FROM session_citations
                WHERE session_id = ?
                ORDER BY last_seen DESC, mention_count DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()

        return [row["citation_key"] for row in rows]

    def has_seen_concept(self, session_id: str, concept: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT 1
                FROM session_concepts
                WHERE session_id = ? AND concept = ?
                LIMIT 1
                """,
                (session_id, concept),
            ).fetchone()
        return row is not None

    def get_session_concepts(
        self, session_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT concept, mention_count, first_mentioned, last_mentioned
                FROM session_concepts
                WHERE session_id = ?
                ORDER BY last_mentioned DESC, mention_count DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()

        return [
            {
                "concept": row["concept"],
                "mention_count": row["mention_count"],
                "first_mentioned": row["first_mentioned"],
                "last_mentioned": row["last_mentioned"],
            }
            for row in rows
        ]

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "DELETE FROM session_documents WHERE session_id = ?", (session_id,)
                )
                conn.execute(
                    "DELETE FROM session_citations WHERE session_id = ?", (session_id,)
                )
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
                    INSERT INTO sessions (session_id, user_id, title, created_at, updated_at)
                    VALUES (?, NULL, NULL, ?, ?)
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
                    (
                        session_id,
                        document_id,
                        filename,
                        source,
                        file_type,
                        file_hash,
                        now,
                    ),
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

    def get_session_documents_by_ids(
        self, session_id: str, document_ids: List[str]
    ) -> List[Dict[str, Any]]:
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

    def create_user(
        self,
        email: str,
        password: str,
        full_name: str,
        account_type: str,
        knowledge_level: int,
        questionnaire_answers: Optional[List[Dict[str, Any]]] = None,
        is_admin: bool = False,
    ) -> Dict[str, Any]:
        normalized_email = self._sanitize_email(email)
        if not normalized_email:
            raise ValueError("email is required")

        if self.get_user_by_email(normalized_email) is not None:
            raise ValueError("email already registered")

        password_hash, password_salt = hash_password(password)
        assessment = assess_agile_level(questionnaire_answers or [])
        agile_level = (
            assessment["level"]
            if questionnaire_answers
            else normalize_agile_level(knowledge_level)
        )
        now = self._now_iso()
        user_id = str(uuid4())

        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO users (
                        user_id, email, full_name, account_type, knowledge_level,
                        agile_adoption_level, agile_adoption_label,
                        questionnaire_answers_json, password_hash, password_salt,
                        auth_token_hash, is_admin, created_at, updated_at, last_login_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, NULL)
                    """,
                    (
                        user_id,
                        normalized_email,
                        full_name.strip(),
                        account_type,
                        normalize_agile_level(knowledge_level),
                        agile_level,
                        assessment["label"],
                        json.dumps(questionnaire_answers or []),
                        password_hash,
                        password_salt,
                        1 if is_admin else 0,
                        now,
                        now,
                    ),
                )
                conn.commit()

        profile = self.get_user_profile(user_id)
        if profile is None:
            raise ValueError("failed to create user")
        return profile

    def _fetch_user_row(
        self,
        *,
        email: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[sqlite3.Row]:
        query = ["SELECT * FROM users"]
        params: List[Any] = []
        if email is not None:
            query.append("WHERE email = ?")
            params.append(self._sanitize_email(email))
        elif user_id is not None:
            query.append("WHERE user_id = ?")
            params.append(user_id)
        else:
            return None

        with self._connect() as conn:
            return conn.execute(" ".join(query), params).fetchone()

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        return self._serialize_user_row(self._fetch_user_row(email=email))

    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        return self._serialize_user_row(self._fetch_user_row(user_id=user_id))

    def get_user_by_token(self, token: str) -> Optional[Dict[str, Any]]:
        token_hash = hash_token(token)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE auth_token_hash = ?",
                (token_hash,),
            ).fetchone()
        return self._serialize_user_row(row)

    def issue_user_token(self, user_id: str) -> str:
        token = generate_token()
        token_hash = hash_token(token)
        now = self._now_iso()

        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE users SET auth_token_hash = ?, updated_at = ? WHERE user_id = ?",
                    (token_hash, now, user_id),
                )
                conn.commit()

        return token

    def authenticate_user(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        row = self._fetch_user_row(email=email)
        if row is None:
            return None

        if not verify_password(password, row["password_hash"], row["password_salt"]):
            return None

        now = self._now_iso()
        token = generate_token()
        token_hash = hash_token(token)

        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE users SET auth_token_hash = ?, last_login_at = ?, updated_at = ? WHERE user_id = ?",
                    (token_hash, now, now, row["user_id"]),
                )
                conn.commit()

        profile = self.get_user_profile(row["user_id"])
        if profile is None:
            return None

        profile["access_token"] = token
        return profile

    def update_user_agile_profile(
        self,
        user_id: str,
        questionnaire_answers: List[Dict[str, Any]],
        knowledge_level: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        assessment = assess_agile_level(questionnaire_answers)
        now = self._now_iso()

        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    UPDATE users
                    SET knowledge_level = COALESCE(?, knowledge_level),
                        agile_adoption_level = ?,
                        agile_adoption_label = ?,
                        questionnaire_answers_json = ?,
                        updated_at = ?
                    WHERE user_id = ?
                    """,
                    (
                        (
                            normalize_agile_level(knowledge_level)
                            if knowledge_level is not None
                            else None
                        ),
                        assessment["level"],
                        assessment["label"],
                        json.dumps(questionnaire_answers or []),
                        now,
                        user_id,
                    ),
                )
                conn.commit()

        if cursor.rowcount <= 0:
            return None
        return self.get_user_profile(user_id)

    def build_user_context(self, user_id: str) -> str:
        profile = self.get_user_profile(user_id)
        if not profile:
            return ""

        parts = [f"Usuario autenticado: {profile['full_name']}"]
        parts.append(f"perfil: {profile['account_type']}")
        parts.append(
            f"nivel declarado: {profile['knowledge_level']} ({agile_level_label(profile['knowledge_level'])})"
        )
        parts.append(
            f"nivel estimado: {profile['agile_adoption_level']} ({profile['agile_adoption_label']})"
        )
        return "; ".join(parts)
