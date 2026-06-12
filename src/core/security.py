"""Security helpers for password hashing, token hashing, and agile assessment scoring."""

from __future__ import annotations

from hashlib import pbkdf2_hmac, sha256
import hmac
import secrets
from typing import Any, Dict, Iterable, List, Tuple


PASSWORD_HASH_ITERATIONS = 390_000
PASSWORD_HASH_BYTES = 32
TOKEN_BYTES = 32

_AGILE_LEVEL_LABELS = {
    1: "Ninguno",
    2: "Inicial",
    3: "Intermedio",
    4: "Avanzado",
}

_ANSWER_LEVELS = {
    "d": 1,
    "a": 2,
    "b": 3,
    "c": 4,
}


def hash_password(password: str, salt: bytes | None = None) -> Tuple[str, str]:
    """Return a salted password hash and the salt encoded as hex."""
    salt_bytes = salt or secrets.token_bytes(16)
    derived = pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt_bytes,
        PASSWORD_HASH_ITERATIONS,
        dklen=PASSWORD_HASH_BYTES,
    )
    return derived.hex(), salt_bytes.hex()


def verify_password(password: str, password_hash: str, salt_hex: str) -> bool:
    """Check whether a password matches a stored hash."""
    try:
        salt_bytes = bytes.fromhex(salt_hex)
    except Exception:
        return False

    candidate_hash, _ = hash_password(password, salt=salt_bytes)
    return hmac.compare_digest(candidate_hash, password_hash)


def generate_token() -> str:
    """Generate a user token for API authentication."""
    return secrets.token_urlsafe(TOKEN_BYTES)


def hash_token(token: str) -> str:
    """Hash an auth token before storing it."""
    return sha256(token.encode("utf-8")).hexdigest()


def normalize_agile_level(level: int | None) -> int:
    """Clamp an agile level to the supported 1..4 range."""
    try:
        numeric_level = int(level) if level is not None else 1
    except Exception:
        numeric_level = 1
    return max(1, min(4, numeric_level))


def agile_level_label(level: int | None) -> str:
    """Map a numeric agile level to its human-readable label."""
    return _AGILE_LEVEL_LABELS[normalize_agile_level(level)]


def _extract_answer_letter(item: Any) -> str:
    if isinstance(item, str):
        return item.strip().lower()
    if isinstance(item, dict):
        value = item.get("answer") or item.get("value") or item.get("option")
        return str(value or "").strip().lower()
    return str(item or "").strip().lower()


def assess_agile_level(answers: Iterable[Any]) -> Dict[str, Any]:
    """Compute an agile adoption level from questionnaire answers."""
    letters: List[str] = []
    scores: List[int] = []

    for answer in answers:
        letter = _extract_answer_letter(answer)
        if letter not in _ANSWER_LEVELS:
            continue
        letters.append(letter)
        scores.append(_ANSWER_LEVELS[letter])

    if not scores:
        return {
            "level": 1,
            "label": _AGILE_LEVEL_LABELS[1],
            "answers": letters,
            "average_score": 1.0,
        }

    average_score = sum(scores) / len(scores)
    level = normalize_agile_level(round(average_score))
    return {
        "level": level,
        "label": _AGILE_LEVEL_LABELS[level],
        "answers": letters,
        "average_score": round(average_score, 2),
    }


def build_user_profile_context(user_profile: Dict[str, Any] | None) -> str:
    """Build a concise user context block for prompts."""
    if not user_profile:
        return ""

    full_name = user_profile.get("full_name") or user_profile.get("name") or "Usuario"
    account_type = user_profile.get("account_type") or user_profile.get("role")
    declared_level = user_profile.get("knowledge_level")
    assessed_level = user_profile.get("agile_adoption_level")
    assessed_label = user_profile.get("agile_adoption_label")

    parts = [f"Usuario: {full_name}"]
    if account_type:
        parts.append(f"perfil: {account_type}")
    if declared_level is not None:
        parts.append(
            f"nivel declarado: {normalize_agile_level(declared_level)} ({agile_level_label(declared_level)})"
        )
    if assessed_level is not None:
        label = assessed_label or agile_level_label(assessed_level)
        parts.append(
            f"nivel estimado: {normalize_agile_level(assessed_level)} ({label})"
        )

    return "; ".join(parts)
