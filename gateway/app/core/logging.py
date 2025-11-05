# app/core/logging.py
import logging
import sys
from logging.config import dictConfig
from uuid import uuid4
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware that attaches a unique request ID to every request.
    Adds it to logs so tracing multiple requests is easy.
    """

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


def configure_logging(level: str = "INFO"):
    """
    Configure structured logging with timestamp, level, and request_id.
    This should be called once in get_app().
    """
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": (
                    "%(asctime)s | %(levelname)s | %(name)s | "
                    "request_id=%(request_id)s | %(message)s"
                ),
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
                "formatter": "default",
            },
        },
        "root": {
            "level": level.upper(),
            "handlers": ["console"],
        },
    }
    dictConfig(logging_config)

    # Add a Filter to always include request_id even if absent
    class RequestIDFilter(logging.Filter):
        def filter(self, record):
            if not hasattr(record, "request_id"):
                record.request_id = "-"
            return True

    for handler in logging.getLogger().handlers:
        handler.addFilter(RequestIDFilter())
