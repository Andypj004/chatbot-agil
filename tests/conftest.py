"""Pytest configuration and fixtures"""

import pytest
import os
from pathlib import Path

# Set test environment variables
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["LOG_LEVEL"] = "ERROR"


@pytest.fixture(scope="session")
def test_data_dir():
    """Create and return test data directory"""
    data_dir = Path("tests/data")
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture(scope="session")
def sample_text_file(test_data_dir):
    """Create a sample text file for testing"""
    file_path = test_data_dir / "sample.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("This is a sample text file for testing. " * 50)
    yield file_path
    # Cleanup
    if file_path.exists():
        file_path.unlink()


@pytest.fixture
def mock_settings():
    """Mock settings for testing"""
    from unittest.mock import Mock

    settings = Mock()
    settings.default_llm_provider = "openai"
    settings.default_model = "gpt-4"
    settings.temperature = 0.7
    settings.max_tokens = 2000
    settings.chunk_size = 500
    settings.chunk_overlap = 50
    settings.top_k_results = 5
    return settings
