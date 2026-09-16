# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured logging infrastructure for UAEF."""

import logging
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, Optional

from uaef.config import get_config


class StructuredFormatter(logging.Formatter):
    """Custom formatter that adds structured fields to log records."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structured fields."""
        # Add timestamp
        record.timestamp = datetime.utcnow().isoformat()
        
        # Add structured fields if present
        if hasattr(record, "structured_data"):
            structured = record.structured_data
            record.msg = f"{record.msg} | {self._format_structured(structured)}"
        
        return super().format(record)
    
    def _format_structured(self, data: Dict[str, Any]) -> str:
        """Format structured data as key=value pairs."""
        return " ".join(f"{k}={v}" for k, v in data.items())


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[Path] = None,
    enable_console: bool = True
) -> None:
    """
    Set up logging infrastructure for UAEF.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        enable_console: Whether to enable console logging
    """
    # Get log level from config if not provided
    if log_level is None:
        config = get_config()
        log_level = config.log_level
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create root logger
    root_logger = logging.getLogger("uaef")
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = StructuredFormatter(
        fmt="%(timestamp)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    
    # Add console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    root_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Name of the logger (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"uaef.{name}")


def log_with_context(
    logger: logging.Logger,
    level: int,
    message: str,
    **context: Any
) -> None:
    """
    Log a message with structured context data.
    
    Args:
        logger: Logger instance
        level: Logging level
        message: Log message
        **context: Additional context fields
    """
    # Create a log record with structured data
    extra = {"structured_data": context}
    logger.log(level, message, extra=extra)


@contextmanager
def log_operation(
    operation_name: str,
    logger: Optional[logging.Logger] = None,
    **context: Any
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager for logging operations with timing and status.
    
    Args:
        operation_name: Name of the operation
        logger: Logger instance (uses root logger if None)
        **context: Additional context fields
        
    Yields:
        Dictionary for adding additional context during operation
        
    Example:
        with log_operation("evaluate_trace", trace_id=trace_id) as ctx:
            result = evaluate(trace)
            ctx["score"] = result.overall_score
    """
    if logger is None:
        logger = get_logger("operations")
    
    start_time = datetime.utcnow()
    operation_context = dict(context)
    operation_context["operation"] = operation_name
    
    log_with_context(
        logger,
        logging.INFO,
        f"Starting operation: {operation_name}",
        **operation_context
    )
    
    try:
        # Yield context dict that can be updated during operation
        yield operation_context
        
        # Calculate duration
        duration = (datetime.utcnow() - start_time).total_seconds()
        operation_context["duration_seconds"] = duration
        operation_context["status"] = "success"
        
        log_with_context(
            logger,
            logging.INFO,
            f"Completed operation: {operation_name}",
            **operation_context
        )
        
    except Exception as e:
        # Calculate duration
        duration = (datetime.utcnow() - start_time).total_seconds()
        operation_context["duration_seconds"] = duration
        operation_context["status"] = "failed"
        operation_context["error"] = str(e)
        operation_context["error_type"] = type(e).__name__
        
        log_with_context(
            logger,
            logging.ERROR,
            f"Failed operation: {operation_name}",
            **operation_context
        )
        
        # Re-raise the exception
        raise


class LoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds default context to all log messages.
    
    Example:
        adapter = LoggerAdapter(logger, {"trace_id": trace_id})
        adapter.info("Processing trace")  # Automatically includes trace_id
    """
    
    def process(self, msg: str, kwargs: Any) -> tuple:
        """Add context to log message."""
        extra = kwargs.get("extra", {})
        
        # Merge adapter context with message context
        if "structured_data" in extra:
            extra["structured_data"].update(self.extra)
        else:
            extra["structured_data"] = dict(self.extra)
        
        kwargs["extra"] = extra
        return msg, kwargs


# Initialize logging on module import
setup_logging()
