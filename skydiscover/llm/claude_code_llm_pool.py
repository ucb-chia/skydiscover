"""Typed errors and rate-limit parsers for the Claude Code CLI backend.

This module is a non-Ray subset of chia's ``claude_code_llm_pool.py``,
lifted verbatim for use by :class:`ClaudeCodeLLM`.  Only the exception
hierarchy and the two rate-limit parsers are ported — the Ray-backed
pool, placement groups, and probe task are dropped because agentic-flow
runs single-process.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger("llm_pool")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ClaudeCodeError(Exception):
    """Base for all Claude Code CLI errors.

    Every subclass must implement ``__reduce__`` for Ray serialization.
    """

    def __init__(
        self,
        node_id: str,
        error_type: str,
        exit_code: int = -1,
        raw_message: str = "",
    ):
        self.node_id = node_id
        self.error_type = error_type
        self.exit_code = exit_code
        self.raw_message = raw_message
        super().__init__(f"{error_type} on {node_id}: {raw_message[:200]}")

    def __reduce__(self):
        return (
            self.__class__,
            (self.node_id, self.error_type, self.exit_code, self.raw_message),
        )


class RateLimitError(ClaudeCodeError):
    """Raised when the Claude CLI response indicates a usage-limit hit."""

    def __init__(
        self,
        node_id: str,
        reset_time: datetime,
        raw_message: str = "",
        exit_code: int = -1,
    ):
        self.reset_time = reset_time
        super().__init__(
            node_id=node_id,
            error_type="rate_limit",
            exit_code=exit_code,
            raw_message=raw_message,
        )

    def __reduce__(self):
        return (
            self.__class__,
            (self.node_id, self.reset_time, self.raw_message, self.exit_code),
        )


class AuthenticationError(ClaudeCodeError):
    """Raised when the CLI's auth token/API key is invalid or expired."""

    def __init__(self, node_id: str, exit_code: int = -1, raw_message: str = ""):
        super().__init__(node_id, "authentication_failed", exit_code, raw_message)

    def __reduce__(self):
        return (self.__class__, (self.node_id, self.exit_code, self.raw_message))


class BillingError(ClaudeCodeError):
    """Raised when the billing account has payment issues."""

    def __init__(self, node_id: str, exit_code: int = -1, raw_message: str = ""):
        super().__init__(node_id, "billing_error", exit_code, raw_message)

    def __reduce__(self):
        return (self.__class__, (self.node_id, self.exit_code, self.raw_message))


class InvalidRequestError(ClaudeCodeError):
    """Raised when the request is malformed (bad prompt, unsupported params, etc.)."""

    def __init__(self, node_id: str, exit_code: int = -1, raw_message: str = ""):
        super().__init__(node_id, "invalid_request", exit_code, raw_message)

    def __reduce__(self):
        return (self.__class__, (self.node_id, self.exit_code, self.raw_message))


class ServerError(ClaudeCodeError):
    """Raised when Anthropic's API returns a server-side error (500/503)."""

    def __init__(
        self,
        node_id: str,
        exit_code: int = -1,
        raw_message: str = "",
        retry_after: Optional[int] = None,
    ):
        self.retry_after = retry_after
        super().__init__(node_id, "server_error", exit_code, raw_message)

    def __reduce__(self):
        return (
            self.__class__,
            (self.node_id, self.exit_code, self.raw_message, self.retry_after),
        )


class MaxOutputTokensError(ClaudeCodeError):
    """Raised when the LLM's response was truncated by the output token limit."""

    def __init__(
        self,
        node_id: str,
        exit_code: int = -1,
        raw_message: str = "",
        partial_text: str = "",
    ):
        self.partial_text = partial_text
        super().__init__(node_id, "max_output_tokens", exit_code, raw_message)

    def __reduce__(self):
        return (
            self.__class__,
            (self.node_id, self.exit_code, self.raw_message, self.partial_text),
        )


class UnknownClaudeError(ClaudeCodeError):
    """Raised for unclassified CLI errors."""

    def __init__(
        self,
        node_id: str,
        exit_code: int = -1,
        raw_message: str = "",
        stderr: str = "",
    ):
        self.stderr = stderr
        super().__init__(node_id, "unknown", exit_code, raw_message)

    def __reduce__(self):
        return (
            self.__class__,
            (self.node_id, self.exit_code, self.raw_message, self.stderr),
        )


class AllMachinesRateLimitedError(Exception):
    """Raised when no LLM machines are available within the timeout."""

    def __init__(self, message: str = "", earliest_reset: Optional[datetime] = None):
        self.earliest_reset = earliest_reset
        super().__init__(message)

    def __reduce__(self):
        return (self.__class__, (str(self), self.earliest_reset))


# ---------------------------------------------------------------------------
# Rate-limit text parser
# ---------------------------------------------------------------------------

_RATE_LIMIT_RE = re.compile(
    r"You've hit your limit\s*[·•\-—]\s*resets?\s+(\d{1,2})\s*(am|pm)\s*\(([^)]+)\)",
    re.IGNORECASE,
)


def parse_rate_limit_reset(text: str) -> Optional[datetime]:
    """Parse a Claude rate-limit message and return the UTC reset time.

    Expected format: ``"You've hit your limit · resets 4pm (America/Los_Angeles)"``

    Returns ``None`` when no rate-limit message is found.
    """
    m = _RATE_LIMIT_RE.search(text)
    if m is None:
        return None

    hour = int(m.group(1))
    ampm = m.group(2).lower()
    tz_str = m.group(3).strip()

    # Convert 12-hour → 24-hour
    if ampm == "pm" and hour != 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0

    # Resolve timezone
    try:
        import zoneinfo
        tz = zoneinfo.ZoneInfo(tz_str)
    except Exception:
        # Fallback: common abbreviations
        _abbrev = {
            "PST": -8, "PDT": -7, "MST": -7, "MDT": -6,
            "CST": -6, "CDT": -5, "EST": -5, "EDT": -4,
            "UTC": 0, "GMT": 0,
        }
        offset_hours = _abbrev.get(tz_str.upper(), 0)
        tz = timezone(timedelta(hours=offset_hours))

    now_in_tz = datetime.now(tz)
    reset_local = now_in_tz.replace(hour=hour, minute=0, second=0, microsecond=0)

    # If reset hour is in the past, it means tomorrow
    if reset_local <= now_in_tz:
        reset_local += timedelta(days=1)

    return reset_local.astimezone(timezone.utc)


def parse_rate_limit_event(event: dict) -> Optional[datetime]:
    """Parse a ``rate_limit_event`` JSON object and return the UTC reset time.

    Only triggers when ``rate_limit_info.status`` is ``"rejected"`` — the event
    is also emitted with other statuses as an informational notice, which should
    NOT be treated as a rate limit.
    """
    info = event.get("rate_limit_info", {})
    if info.get("status") != "rejected":
        return None
    resets_at = info.get("resetsAt")
    if resets_at is None:
        return None
    try:
        return datetime.fromtimestamp(int(resets_at), tz=timezone.utc)
    except (ValueError, TypeError, OSError):
        return None
