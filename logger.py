"""Module for the contextual logger."""

import logging
from threading import local
from typing import Any, Dict, MutableMapping, Optional, Tuple

_THREAD_LOCAL_VARS = local()

_THREAD_LOCAL_VARS.log_context = {}


def get_extra_context() -> Any:
    """Get log context for pid."""
    if not hasattr(_THREAD_LOCAL_VARS, "log_context"):
        _THREAD_LOCAL_VARS.log_context = {}
    return _THREAD_LOCAL_VARS.log_context


def set_extra_context(context: Dict[str, Any]) -> None:
    """Set log context for pid.

    Erases previous context.
    """
    if not hasattr(_THREAD_LOCAL_VARS, "log_context"):
        _THREAD_LOCAL_VARS.log_context = {}
    _THREAD_LOCAL_VARS.log_context = context


class ContextAwareLogAdapter(logging.LoggerAdapter):
    """Log adapter for adding additional context to a log line.

    See
    https://docs.python.org/3/howto/logging-cookbook.html#using-loggeradapters-to-impart-contextual-information    # noqa
    for more information
    """

    def process(
        self, msg: str, kwargs: MutableMapping[str, Any]  # noqa: C812
    ) -> Tuple[str, Any]:  # noqa
        """Modify the message that gets passed to a logging call.

        kwargs has the `extra` values. We choose to not modify that here, so
        just pass it through as-is to the logging function
        """
        extra = get_extra_context()
        if len(extra) == 0:
            return msg, kwargs

        extra_context = " ".join(
            ["{}={}".format(k, v) for k, v in extra.items()],
        )
        return "{} {}".format(extra_context, msg), kwargs


class LoggingContext:
    """Add context into a logging block."""

    def __init__(self, context: Dict[str, Any]) -> None:
        """Pass context to the logging."""
        self._new_context = context
        self._old_context: Dict[str, Any] = {}

    def __enter__(self) -> Any:
        """Enter into the context block."""
        self._old_context = get_extra_context()
        set_extra_context({**self._old_context, **self._new_context})
        return self

    def __exit__(self, *exc: Tuple[Any, ...]) -> None:
        """Exit the context block."""
        set_extra_context(self._old_context)


def get_logger(
    name: str, level: Optional[int | str] = None  # noqa: C812
) -> ContextAwareLogAdapter:  # noqa
    """
    Return basic logger with specified name and level.

    Args:
        name: The unique name of the logger
        level: The logger level; defaults to logging.INFO
    """
    if level is None:
        level = 20

    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    logger.addHandler(ch)
    ch.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
    return ContextAwareLogAdapter(logger, {})
