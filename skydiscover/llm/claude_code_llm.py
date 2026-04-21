"""Chia-style Claude Code CLI backend, ported into SkyDiscover.

This is a mechanical port of ``chia/chia/base/claude_code_llm.py`` with only
the surgical edits required to run outside chia's Ray-backed infrastructure:

1. ``claude_path`` and ``claude_config_dir`` are accepted as constructor
   params so the binary and auth dir can be overridden without modifying
   the class defaults.
2. ``_get_node_id`` returns ``"local"`` — there is no Ray runtime context.
3. Ray-coupled methods (``prompt_pooled``, ``_serializable_config``,
   ``_pooled_prompt_task``, ``_session_transcript_path``) are dropped.
4. ``ChiaTool`` type hints are replaced with ``Any`` — tools are unused by
   AdaEvolve.

Everything else — ``_classify_error``, typed-error retry semantics,
``_run_claude_streaming``, and NDJSON event parsing — is preserved verbatim.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import subprocess
import tempfile
from datetime import datetime
import threading
from typing import Any, List, Optional, Tuple
from uuid import uuid4


@dataclasses.dataclass
class CLIResult:
    """Structured result from a ``claude`` CLI invocation."""

    text: str
    returncode: int
    stderr: str


class ClaudeCodeLLM:
    """Wraps the Claude Code CLI (``claude --print``) as an LLM backend.

    Each call to :meth:`prompt` spawns a ``claude`` subprocess that can
    optionally connect to MCP tool servers.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        system_message: str = "",
        timeout_seconds: int = 600,
        retries: int = 3,
        logging_name: str = "claude_code",
        log_dir: Optional[str] = None,
        resume_session: bool = False,
        extra_cli_args: Optional[List[str]] = None,
        claude_path: str = "claude",
        claude_config_dir: Optional[str] = None,
    ):
        self.model = model
        self.system_message = system_message
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.extra_cli_args = extra_cli_args or []
        self.logger = logging.getLogger(logging_name)
        self.claude_path = claude_path
        self.claude_config_dir = claude_config_dir

        self._call_counter = 0
        self._session_id = str(uuid4()) if resume_session else None
        self._last_metadata: dict = {}  # populated by _process_event_line
        self._rate_limit_event: Optional[dict] = None  # populated by _process_event_line

        self._log_dir = log_dir
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            session_tag = f"_{self._session_id[:8]}" if self._session_id else ""
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._log_prefix = os.path.join(log_dir, f"{logging_name}_{run_id}{session_tag}")
        else:
            self._log_prefix = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prompt(
        self,
        user_message: str,
        tools: Optional[List[Any]] = [],
    ) -> Tuple[str, bool]:
        """Send *user_message* to Claude Code CLI and return the response.

        Returns:
            ``(response_text, success)`` — *success* is ``False`` when all
            retry attempts fail.

        Raises:
            RateLimitError: Usage limit hit — propagates immediately.
            AuthenticationError: Auth failure — propagates immediately.
            BillingError: Billing/payment issue — propagates immediately.
            InvalidRequestError: Malformed request — propagates immediately.
            ServerError: After all retries with exponential backoff.
            MaxOutputTokensError: After one retry attempt.
        """
        import time as _time

        from skydiscover.llm.claude_code_llm_pool import (
            AuthenticationError,
            BillingError,
            InvalidRequestError,
            MaxOutputTokensError,
            RateLimitError,
            ServerError,
            UnknownClaudeError,
        )

        for attempt in range(self.retries):
            try:
                self._last_metadata = {}
                self._rate_limit_event = None
                cli = self._run_claude(user_message, tools)
                self._call_counter += 1
                self._last_metadata["model"] = self.model
                self._last_metadata["tools"] = [
                    {"name": t.name, "hostname": getattr(t, "hostname", None),
                     "port": getattr(t, "port", None),
                     "node_id": getattr(t, "node_id", None)}
                    for t in tools
                ]

                # Classify and raise typed errors
                self._classify_error(cli)

                return cli.text, True

            # -- Never retry: propagate immediately --
            except (RateLimitError, AuthenticationError, BillingError, InvalidRequestError):
                raise

            # -- Retry once: stochastic generation may produce shorter output --
            except MaxOutputTokensError:
                if attempt == 0:
                    self.logger.warning(
                        "Max output tokens on attempt %d/%d, retrying once",
                        attempt + 1, self.retries,
                    )
                    continue
                raise

            # -- Retry with exponential backoff: transient API issue --
            except ServerError:
                backoff = min(5 * 2 ** attempt, 60)
                self.logger.warning(
                    "Server error on attempt %d/%d, backing off %ds",
                    attempt + 1, self.retries, backoff,
                )
                _time.sleep(backoff)

            # -- Standard retry for unknown errors --
            except UnknownClaudeError as exc:
                self.logger.warning(
                    "Unknown error on attempt %d/%d: %s",
                    attempt + 1, self.retries, exc,
                )

            except subprocess.TimeoutExpired:
                # A timeout means the session was likely created;
                # switch to --resume for subsequent attempts.
                if self._session_id is not None and self._call_counter == 0:
                    self._call_counter = 1
                self.logger.warning(
                    "Timeout on attempt %d/%d", attempt + 1, self.retries,
                )

            except Exception as exc:
                self.logger.warning(
                    "Unexpected error on attempt %d/%d: %s",
                    attempt + 1, self.retries, exc,
                )
        return "", False

    def _get_node_id(self) -> str:
        return "local"

    def _classify_error(self, cli: CLIResult) -> None:
        """Inspect *cli* and raise a typed error if something went wrong.

        Check order:
        1. Rate limit (text regex + streaming event) — highest priority
        2. Non-zero exit code — classify by stderr patterns
        3. Success — return without raising
        """
        from skydiscover.llm.claude_code_llm_pool import (
            AuthenticationError,
            BillingError,
            InvalidRequestError,
            MaxOutputTokensError,
            RateLimitError,
            ServerError,
            UnknownClaudeError,
            parse_rate_limit_event,
            parse_rate_limit_reset,
        )

        # --- 1. Rate limit (can appear even with exit code 0) ---
        reset_time = parse_rate_limit_reset(cli.text)
        if reset_time is None and self._rate_limit_event is not None:
            reset_time = parse_rate_limit_event(self._rate_limit_event)

        if reset_time is not None:
            node_id = self._get_node_id()
            self.logger.warning(
                "Rate limit detected on %s (resets %s). Response text:\n%s",
                node_id, reset_time.isoformat(), cli.text,
            )
            raise RateLimitError(
                node_id=node_id,
                reset_time=reset_time,
                raw_message=cli.text[:300],
                exit_code=cli.returncode,
            )

        # --- 2. Non-zero exit code → classify by stderr ---
        if cli.returncode != 0:
            node_id = self._get_node_id()
            stderr_lower = cli.stderr.lower()

            if any(kw in stderr_lower for kw in (
                "authentication", "unauthorized", "401", "not authenticated",
                "login", "auth token",
            )):
                raise AuthenticationError(
                    node_id=node_id,
                    exit_code=cli.returncode,
                    raw_message=cli.stderr[:300],
                )

            if any(kw in stderr_lower for kw in (
                "billing", "payment", "402", "overdue", "subscription",
                "plan expired",
            )):
                raise BillingError(
                    node_id=node_id,
                    exit_code=cli.returncode,
                    raw_message=cli.stderr[:300],
                )

            if any(kw in stderr_lower for kw in (
                "invalid request", "malformed", "400", "bad request",
                "invalid model",
            )):
                raise InvalidRequestError(
                    node_id=node_id,
                    exit_code=cli.returncode,
                    raw_message=cli.stderr[:300],
                )

            if any(kw in stderr_lower for kw in (
                "500", "503", "server error", "overloaded",
                "internal error", "service unavailable",
            )):
                raise ServerError(
                    node_id=node_id,
                    exit_code=cli.returncode,
                    raw_message=cli.stderr[:300],
                )

            if any(kw in stderr_lower for kw in (
                "max_output_tokens", "output token limit", "maximum output",
                "response too long",
            )):
                raise MaxOutputTokensError(
                    node_id=node_id,
                    exit_code=cli.returncode,
                    raw_message=cli.stderr[:300],
                    partial_text=cli.text,
                )

            # Fallback: unknown error
            raise UnknownClaudeError(
                node_id=node_id,
                exit_code=cli.returncode,
                raw_message=cli.stderr[:300],
                stderr=cli.stderr,
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_mcp_config(self, tools: List[Any]) -> dict:
        """Build the JSON object expected by ``--mcp-config``."""
        servers = {}
        for tool in tools:
            port = getattr(tool, "port", 8000)
            servers[tool.name] = {
                "type": "http",
                "url": f"http://{tool.hostname}:{port}/{tool.name}/mcp",
            }
        return {"mcpServers": servers}

    def _build_allowed_tools(self, tools: List[Any]) -> List[str]:
        """Return ``--allowedTools`` entries for every registered MCP tool."""
        allowed: list[str] = []
        for tool in tools:
            # FastMCP registers tools under server_name; the MCP tool ID
            # that Claude Code recognises is  mcp__<server>__<tool_name>.
            for fn_info in tool.mcp._tool_manager.list_tools():
                allowed.append(f"mcp__{tool.name}__{fn_info.name}")
        return allowed

    def _build_cmd(self, tools: Optional[List[Any]] = None) -> list:
        """Build the ``claude`` CLI command list.

        The user message is piped via stdin (``-p -``) to avoid OS
        argument-length limits with long prompts.
        """
        cmd = [
            self.claude_path,
            "--print",
            "--model", self.model,
            "--dangerously-skip-permissions",
        ]

        if self.extra_cli_args:
            cmd += self.extra_cli_args

        if self._log_prefix is not None:
            cmd += ["--output-format", "stream-json", "--verbose"]

        if self.system_message:
            cmd += ["--system-prompt", self.system_message]

        if self._session_id is not None:
            if self._call_counter > 0:
                cmd += ["--resume", self._session_id]
            else:
                cmd += ["--session-id", self._session_id]

        if tools:
            cfg = self._build_mcp_config(tools)
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            )
            json.dump(cfg, tmp)
            tmp.close()
            cmd += ["--mcp-config", tmp.name]

            allowed = self._build_allowed_tools(tools)
            if allowed:
                cmd += ["--allowedTools", ",".join(allowed)]

        cmd += ["-p", "-"]
        return cmd

    def _run_claude(
        self,
        user_message: str,
        tools: Optional[List[Any]] = None,
    ) -> CLIResult:
        cmd = self._build_cmd(tools)
        self.logger.info("Running: %s", " ".join(cmd[:6]) + " ...")
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

        if self.claude_config_dir is not None:
            env["CLAUDE_CONFIG_DIR"] = os.path.expanduser(self.claude_config_dir)

        if self._log_prefix is not None:
            return self._run_claude_streaming(cmd, env, user_message)

        result = subprocess.run(
            cmd,
            input=user_message,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            env=env,
        )

        if result.returncode != 0:
            self.logger.warning("claude exited %d: %s", result.returncode, result.stderr[:500])

        return CLIResult(
            text=result.stdout,
            returncode=result.returncode,
            stderr=result.stderr,
        )

    # ------------------------------------------------------------------
    # Streaming log implementation
    # ------------------------------------------------------------------

    def _run_claude_streaming(self, cmd: list, env: dict, user_message: str) -> CLIResult:
        """Run claude with ``--output-format stream-json``.

        Writes a single log file at ``<prefix>.log`` under ``log_dir``.
        Stderr (NDJSON events) is parsed by ``_process_event_line`` into
        structured entries.  Stdout lines are written with a ``[stdout]``
        prefix.  A lock serialises writes from the two drain threads.
        """
        result_text_parts: list[str] = []
        stderr_parts: list[str] = []
        log_path = f"{self._log_prefix}.log"
        lock = threading.Lock()

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        proc.stdin.write(user_message)
        proc.stdin.close()

        log_file = open(log_path, "a")

        log_file.write("=" * 80 + "\n")
        log_file.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Prompt #{self._call_counter}\n")
        log_file.write("=" * 80 + "\n\n")
        truncated = user_message[:500] + ("..." if len(user_message) > 500 else "")
        log_file.write(f"[User Message]\n{truncated}\n\n")
        log_file.flush()

        def drain_stdout():
            for line in proc.stdout:
                line = line.strip()
                if line:
                    with lock:
                        self._process_event_line(line, log_file, result_text_parts)
                        log_file.flush()

        def drain_stderr():
            for line in proc.stderr:
                with lock:
                    stderr_parts.append(line)
                    log_file.write(f"[stderr] {line}")
                    log_file.flush()

        t1 = threading.Thread(target=drain_stderr)
        t2 = threading.Thread(target=drain_stdout)
        t1.start()
        t2.start()

        try:
            proc.wait(timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            t1.join()
            t2.join()
            log_file.write(
                f"[TIMEOUT] claude exceeded {self.timeout_seconds}s; killed.\n"
            )
            log_file.flush()
            log_file.close()
            from skydiscover.llm.claude_code_llm_pool import UnknownClaudeError
            raise UnknownClaudeError(
                f"claude subprocess exceeded timeout of {self.timeout_seconds}s"
            )
        t1.join()
        t2.join()

        if not result_text_parts:
            log_file.write("[DEBUG] No events parsed.\n")

        log_file.write("-" * 80 + "\n\n")
        log_file.flush()
        log_file.close()

        if proc.returncode != 0:
            self.logger.warning("claude exited %d", proc.returncode)

        return CLIResult(
            text="".join(result_text_parts),
            returncode=proc.returncode,
            stderr="".join(stderr_parts),
        )

    def _process_event_line(self, line: str, f, result_text_parts: list) -> None:
        """Parse a single NDJSON event line and write to the log file."""
        if not line:
            return

        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            f.write(f"[UNPARSED] {line[:200]}\n")
            f.flush()
            return

        event_type = event.get("type", "")

        if event_type == "assistant":
            msg = event.get("message", {})
            for block in msg.get("content", []):
                block_type = block.get("type", "")
                if block_type == "thinking":
                    f.write("[Thinking]\n")
                    f.write(block.get("thinking", ""))
                    f.write("\n\n")
                    f.flush()
                elif block_type == "text":
                    text = block.get("text", "")
                    result_text_parts.append(text)
                    f.write("[Response]\n")
                    f.write(text)
                    f.write("\n\n")
                    f.flush()
                elif block_type == "tool_use":
                    tool_name = block.get("name", "unknown")
                    tool_input = json.dumps(block.get("input", {}))
                    if len(tool_input) > 2000:
                        tool_input = tool_input[:2000] + "\n... [truncated]"
                    f.write(f"[Tool Call: {tool_name}]\n")
                    f.write(f"Args: {tool_input}\n\n")
                    f.flush()
                elif block_type == "tool_result":
                    content = block.get("content", "")
                    if isinstance(content, list):
                        content = "\n".join(
                            c.get("text", "") for c in content
                            if isinstance(c, dict)
                        )
                    if len(content) > 2000:
                        content = content[:2000] + "\n... [truncated]"
                    f.write(f"[Tool Result]\n{content}\n\n")
                    f.flush()

        elif event_type == "user":
            # Tool results ride on user events — the CLI echoes each tool
            # reply back to the assistant as a user-turn ``tool_result``
            # block.  Capturing them here is the only way to see what the
            # assistant actually saw when it chose its next action.
            msg = event.get("message", {})
            content_blocks = msg.get("content", [])
            if isinstance(content_blocks, str):
                # Plain-text user message (initial prompt echo); skip.
                return
            for block in content_blocks:
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "tool_result":
                    continue
                content = block.get("content", "")
                if isinstance(content, list):
                    content = "\n".join(
                        c.get("text", "") for c in content
                        if isinstance(c, dict)
                    )
                if len(content) > 2000:
                    content = content[:2000] + "\n... [truncated]"
                label = "Tool Result (error)" if block.get("is_error") else "Tool Result"
                f.write(f"[{label}]\n{content}\n\n")
                f.flush()

        elif event_type == "result":
            result_text = event.get("result", "")
            if result_text and not result_text_parts:
                result_text_parts.append(result_text)

            parts = []
            meta: dict = {}
            cost = event.get("total_cost_usd")
            if cost is not None:
                parts.append(f"Cost: ${cost:.4f}")
                meta["cost_usd"] = cost
            duration = event.get("duration_ms")
            if duration is not None:
                parts.append(f"Duration: {duration / 1000:.1f}s")
                meta["duration_s"] = round(duration / 1000, 2)
            turns = event.get("num_turns")
            if turns is not None:
                parts.append(f"Turns: {turns}")
                meta["num_turns"] = turns
            usage = event.get("usage", {})
            in_tok = usage.get("input_tokens", 0)
            out_tok = usage.get("output_tokens", 0)
            if in_tok:
                parts.append(f"Input tokens: {in_tok}")
                meta["input_tokens"] = in_tok
            if out_tok:
                parts.append(f"Output tokens: {out_tok}")
                meta["output_tokens"] = out_tok
            if parts:
                f.write(f"[Metadata]\n{' | '.join(parts)}\n")
                f.flush()
            if meta:
                self._last_metadata = meta

        elif event_type == "rate_limit_event":
            self._rate_limit_event = event

        # system — skip silently
