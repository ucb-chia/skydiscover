"""SkyDiscover adapter around the chia-style :class:`ClaudeCodeLLM`.

SkyDiscover's :class:`LLMInterface` is async and exposes ``system_message``
per call; chia's ``ClaudeCodeLLM.prompt`` is sync and reads ``system_message``
from the instance.  This adapter bridges the two by running the subprocess
on a worker thread (``asyncio.to_thread``) and mutating
``self._llm.system_message`` before each call — the minimum divergence from
chia forced by SkyDiscover's active paradigm-breakthrough path
(``search/adaevolve/paradigm/generator.py`` varies the system prompt per call).
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List, Optional

from skydiscover.config import LLMModelConfig
from skydiscover.llm.base import LLMInterface, LLMResponse
from skydiscover.llm.claude_code_llm import ClaudeCodeLLM


class ClaudeCodeLLMAdapter(LLMInterface):
    """Wraps :class:`ClaudeCodeLLM` as a SkyDiscover ``LLMInterface``."""

    def __init__(
        self,
        model_cfg: LLMModelConfig,
        *,
        claude_path: str = "claude",
        claude_config_dir: Optional[str] = None,
        log_dir: Optional[str] = None,
        extra_cli_args: Optional[List[str]] = None,
    ):
        self._llm = ClaudeCodeLLM(
            model=model_cfg.name or "claude-sonnet-4-6",
            system_message=model_cfg.system_message or "",
            timeout_seconds=model_cfg.timeout or 600,
            retries=model_cfg.retries or 3,
            logging_name=model_cfg.name or "claude_code",
            log_dir=log_dir,
            resume_session=False,
            extra_cli_args=extra_cli_args,
            claude_path=claude_path,
            claude_config_dir=claude_config_dir,
        )

    async def generate(
        self,
        system_message: str,
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> LLMResponse:
        if kwargs.get("image_output"):
            raise NotImplementedError(
                "ClaudeCodeLLMAdapter does not support image_output"
            )

        if len(messages) != 1 or messages[0].get("role") != "user":
            raise ValueError(
                "ClaudeCodeLLMAdapter only supports a single user message; "
                f"got {len(messages)} messages with roles "
                f"{[m.get('role') for m in messages]}"
            )

        prompt_text = messages[0].get("content", "")
        if not isinstance(prompt_text, str):
            raise ValueError(
                "ClaudeCodeLLMAdapter expects string user content; "
                f"got {type(prompt_text).__name__}"
            )

        # SkyDiscover's paradigm-breakthrough generator varies the system
        # prompt per call, so mutate the wrapped instance before each prompt.
        self._llm.system_message = system_message or ""

        text, _success = await asyncio.to_thread(self._llm.prompt, prompt_text)

        meta = getattr(self._llm, "_last_metadata", None) or {}
        in_tok = meta.get("input_tokens")
        out_tok = meta.get("output_tokens")
        total = (
            (in_tok or 0) + (out_tok or 0)
            if (in_tok is not None or out_tok is not None)
            else None
        )
        return LLMResponse(
            text=text,
            model_used=meta.get("model") or self._llm.model,
            prompt_tokens=in_tok,
            completion_tokens=out_tok,
            total_tokens=total,
        )


def make_claude_code_init_client(
    *,
    claude_path: str = "claude",
    claude_config_dir: Optional[str] = None,
    log_dir: Optional[str] = None,
    extra_cli_args: Optional[List[str]] = None,
) -> Callable[[LLMModelConfig], ClaudeCodeLLMAdapter]:
    """Return an ``init_client`` factory that closes over CLI paths and log dir.

    Wire the returned callable onto each ``LLMModelConfig.init_client`` field
    in ``run_adaevolve.py`` (or equivalent entry point) before handing the
    config to ``run_discovery``.
    """

    def _factory(model_cfg: LLMModelConfig) -> ClaudeCodeLLMAdapter:
        return ClaudeCodeLLMAdapter(
            model_cfg,
            claude_path=claude_path,
            claude_config_dir=claude_config_dir,
            log_dir=log_dir,
            extra_cli_args=extra_cli_args,
        )

    return _factory
