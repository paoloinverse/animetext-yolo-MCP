"""AnimeText OCR Studio.

Anime/manga text-block detection (deepghs/AnimeText_yolo) + crop-level OCR
(JustANormalTinkerer/hayai-ocr-v2.7-supernova), with a PySide6 GUI and an MCP
remote-control server. See PLAN.md for the architecture and part breakdown.
"""

__version__ = "0.1.0"
APP_NAME = "animetext-studio"

from .core.types import ErrorCode, StudioError  # noqa: E402
from .core.config import AppConfig  # noqa: E402

__all__ = ["__version__", "APP_NAME", "AppConfig", "StudioError", "ErrorCode"]
