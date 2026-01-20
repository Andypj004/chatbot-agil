"""
Input validation utilities.
Security and data validation functions (RNF5).
"""

import re
import uuid
from typing import Optional


def validate_email(email: str) -> bool:
    """
    Validate email format.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid email format
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_session_id(session_id: str) -> bool:
    """
    Validate session ID format.
    
    Args:
        session_id: Session ID to validate
        
    Returns:
        True if valid UUID format
    """
    try:
        uuid.UUID(session_id)
        return True
    except (ValueError, AttributeError):
        return False


def sanitize_input(text: str, max_length: int = 1000) -> str:
    """
    Sanitize user input for security (RNF5).
    
    Args:
        text: Text to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized text
    """
    if not text:
        return ""
    
    # Trim whitespace
    text = text.strip()
    
    # Enforce maximum length
    if len(text) > max_length:
        text = text[:max_length]
    
    # Remove potentially dangerous characters
    # (Keep basic punctuation and alphanumeric)
    text = re.sub(r'[<>{}[\]\\]', '', text)
    
    return text


def validate_query_length(query: str, min_length: int = 3, max_length: int = 500) -> tuple[bool, Optional[str]]:
    """
    Validate query length.
    
    Args:
        query: Query text
        min_length: Minimum required length
        max_length: Maximum allowed length
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    query_length = len(query.strip())
    
    if query_length < min_length:
        return False, f"Query too short (minimum {min_length} characters)"
    
    if query_length > max_length:
        return False, f"Query too long (maximum {max_length} characters)"
    
    return True, None


def validate_file_extension(filename: str, allowed_extensions: list) -> bool:
    """
    Validate file extension.
    
    Args:
        filename: Filename to check
        allowed_extensions: List of allowed extensions (e.g., ['.pdf', '.docx'])
        
    Returns:
        True if extension is allowed
    """
    if not filename:
        return False
    
    extension = filename.lower()[filename.rfind('.'):]
    return extension in [ext.lower() for ext in allowed_extensions]


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe storage.
    
    Args:
        filename: Filename to sanitize
        
    Returns:
        Safe filename
    """
    # Remove path separators and special characters
    filename = re.sub(r'[/\\:*?"<>|]', '', filename)
    
    # Replace spaces with underscores
    filename = filename.replace(' ', '_')
    
    # Limit length
    if len(filename) > 255:
        # Keep extension
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        filename = name[:240] + ('.' + ext if ext else '')
    
    return filename
