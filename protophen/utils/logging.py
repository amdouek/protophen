"""
Logging configuration for ProToPhen.

This module provides a consistent logging setup using loguru,
with support for console and file output, and rich formatting.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from loguru import logger

# Import Logger for type checking only
if TYPE_CHECKING:
    from loguru import Logger

# Remove default handler
logger.remove()

# Track if logging has been configured
_configured = False


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str | Path] = None,
    show_time: bool = True,
    show_level: bool = True,
    rich_traceback: bool = True,
) -> None:
    """
    Configure logging for ProToPhen.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        show_time: Whether to show timestamps
        show_level: Whether to show log levels
        rich_traceback: Whether to use rich tracebacks
        
    Example:
        >>> from protophen.utils.logging import setup_logging, logger
        >>> setup_logging(level="DEBUG", log_file="experiment.log")
        >>> logger.info("Starting experiment")
    """
    global _configured
    
    # Clear any existing handlers if reconfiguring
    if _configured:
        logger.remove()
    
    # Build format string
    format_parts = []
    if show_time:
        format_parts.append("<green>{time:YYYY-MM-DD HH:mm:ss}</green>")
    if show_level:
        format_parts.append("<level>{level: <8}</level>")
    format_parts.append("<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>")
    format_parts.append("<level>{message}</level>")
    
    format_string = " | ".join(format_parts)
    
    # Add console handler
    logger.add(
        sys.stderr,
        format=format_string,
        level=level,
        colorize=True,
        backtrace=rich_traceback,
        diagnose=rich_traceback,
    )
    
    # Add file handler if specified
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # File format without colours
        file_format = format_string.replace("<green>", "").replace("</green>", "")
        file_format = file_format.replace("<level>", "").replace("</level>", "")
        file_format = file_format.replace("<cyan>", "").replace("</cyan>", "")
        
        logger.add(
            log_path,
            format=file_format,
            level=level,
            rotation="10 MB",
            retention="1 week",
            compression="zip",
        )
    
    _configured = True
    logger.debug(f"Logging configured: level={level}, file={log_file}")


def get_logger(name: str = "protophen") -> "Logger":
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (used in log messages)
        
    Returns:
        Logger instance
        
    Example:
        >>> logger = get_logger("protophen.embeddings")
        >>> logger.info("Extracting embeddings...")
    """
    return logger.bind(name=name)


# Default configuration (INFO level, console only)
if not _configured:
    setup_logging(level="INFO")


# Re-export logger for convenience
__all__ = ["logger", "setup_logging", "get_logger"]