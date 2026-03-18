"""File handling utilities"""

from typing import Optional
from pathlib import Path
import aiofiles
from fastapi import UploadFile

from src.core.logger import get_logger

logger = get_logger()

ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.docx', '.doc', '.md', '.markdown'}


async def save_uploaded_file(
    upload_file: UploadFile,
    destination_dir: str = "data/uploads"
) -> str:
    """Save an uploaded file to disk
    
    Args:
        upload_file: FastAPI UploadFile object
        destination_dir: Directory to save the file
        
    Returns:
        Path to the saved file
        
    Raises:
        ValueError: If file type is not allowed
    """
    # Create destination directory if it doesn't exist
    dest_path = Path(destination_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # Validate file type
    file_extension = get_file_extension(upload_file.filename)
    if not validate_file_type(file_extension):
        raise ValueError(
            f"File type '{file_extension}' is not allowed. "
            f"Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Generate file path
    file_path = dest_path / upload_file.filename
    
    # Save file
    logger.info(f"Saving uploaded file: {upload_file.filename}")
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await upload_file.read()
        await out_file.write(content)
    
    logger.info(f"File saved successfully: {file_path}")
    return str(file_path)


def get_file_extension(filename: str) -> str:
    """Get file extension from filename
    
    Args:
        filename: Name of the file
        
    Returns:
        File extension (e.g., '.pdf', '.txt')
    """
    return Path(filename).suffix.lower()


def validate_file_type(extension: str) -> bool:
    """Validate if file extension is allowed
    
    Args:
        extension: File extension to validate
        
    Returns:
        True if extension is allowed, False otherwise
    """
    return extension in ALLOWED_EXTENSIONS


def get_allowed_extensions() -> list:
    """Get list of allowed file extensions
    
    Returns:
        List of allowed extensions
    """
    return list(ALLOWED_EXTENSIONS)
