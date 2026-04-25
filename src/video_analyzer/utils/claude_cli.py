"""Claude CLI wrapper — uses `claude -p` for LLM calls via Max subscription.

Instead of managing OAuth tokens directly, this shells out to the `claude` CLI
which handles its own authentication. No more expired tokens.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


@functools.cache
def get_claude_cli_path() -> str | None:
    """Find the claude CLI binary. Cached for the process lifetime."""
    candidates = [
        str(Path.home() / ".claude" / "local" / "claude"),
        "claude",
    ]
    for cmd in candidates:
        try:
            subprocess.run(
                [cmd, "--version"],
                capture_output=True,
                timeout=3,
            )
            return cmd
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            continue
    return None


def _parse_cli_response(stdout: str) -> str:
    """Parse the JSON response from `claude -p --output-format json`."""
    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Claude CLI returned non-JSON: {exc}") from exc

    if parsed.get("is_error"):
        raise RuntimeError(f"Claude CLI error: {parsed.get('result', 'unknown error')}")

    return _strip_code_fences(parsed.get("result", ""))


def _strip_code_fences(text: str) -> str:
    """Strip markdown code fences from response."""
    text = text.strip()
    if not text.startswith("```"):
        return text
    try:
        first_nl = text.index("\n")
        last_fence = text.rfind("```")
        if last_fence > first_nl:
            return text[first_nl + 1 : last_fence].strip()
    except ValueError:
        pass
    return text


def _build_args(
    cli: str,
    *,
    model: str,
    system_prompt: str,
    tools: str,
    max_budget: float,
) -> list[str]:
    args = [
        cli,
        "-p",
        "--model", model,
        "--output-format", "json",
        "--no-session-persistence",
        "--tools", tools,
        "--max-budget-usd", str(max_budget),
    ]
    if system_prompt:
        args.extend(["--system-prompt", system_prompt])
    return args


def _require_cli() -> str:
    cli = get_claude_cli_path()
    if not cli:
        raise RuntimeError(
            "Claude CLI not found. Install: npm install -g @anthropic-ai/claude-code"
        )
    return cli


def call_claude_sync(
    user_message: str,
    *,
    system_prompt: str = "",
    model: str = "haiku",
    tools: str = "",
    timeout: int = 180,
    max_budget: float = 1.0,
) -> str:
    """Call Claude CLI synchronously. Returns the text response."""
    cli = _require_cli()
    args = _build_args(
        cli, model=model, system_prompt=system_prompt, tools=tools, max_budget=max_budget,
    )
    logger.debug("Claude CLI call: model=%s tools=%r", model, tools)

    try:
        result = subprocess.run(
            args, input=user_message, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Claude CLI call timed out") from exc

    stdout = result.stdout.strip()
    if result.returncode != 0 and not stdout:
        raise RuntimeError(
            f"Claude CLI exited {result.returncode}: {result.stderr[:300]}"
        )

    return _parse_cli_response(stdout)


async def call_claude_async(
    user_message: str,
    *,
    system_prompt: str = "",
    model: str = "haiku",
    tools: str = "",
    timeout: int = 180,
    max_budget: float = 1.0,
) -> str:
    """Call Claude CLI asynchronously. Returns the text response."""
    cli = _require_cli()
    args = _build_args(
        cli, model=model, system_prompt=system_prompt, tools=tools, max_budget=max_budget,
    )
    logger.debug("Claude CLI async call: model=%s tools=%r", model, tools)

    proc = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(user_message.encode()),
            timeout=timeout,
        )
    except asyncio.TimeoutError as exc:
        proc.kill()
        await proc.wait()
        raise RuntimeError("Claude CLI call timed out") from exc

    stdout = stdout_bytes.decode().strip()
    if proc.returncode != 0 and not stdout:
        raise RuntimeError(
            f"Claude CLI exited {proc.returncode}: {stderr_bytes.decode()[:300]}"
        )

    return _parse_cli_response(stdout)
