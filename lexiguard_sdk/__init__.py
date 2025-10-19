# lexiguard_sdk/__init__.py
"""
LexiGuard SDK - Legal Document Analysis Toolkit

A Python SDK for analyzing legal documents using Google Gemini AI.

Usage:
    from lexiguard_sdk import LexiGuard
    
    lg = LexiGuard(api_key="YOUR_API_KEY")
    result = lg.analyze_text("Contract text here...")
"""

__version__ = "1.0.0"
__author__ = "LexiGuard Team"

from .core import LexiGuard, LexiGuardError
from .file_utils import FileParser, FileParsingError, analyze_file_quick

__all__ = [
    "LexiGuard",
    "LexiGuardError",
    "FileParser",
    "FileParsingError",
    "analyze_file_quick"
]