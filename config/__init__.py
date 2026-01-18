"""
Configuration module for the Educational Agile Chatbot.
Handles global settings, prompts, and environment variables.
"""

from .settings import settings
from .prompts import SYSTEM_PROMPTS

__all__ = ["settings", "SYSTEM_PROMPTS"]
