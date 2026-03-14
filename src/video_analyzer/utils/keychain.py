"""Read Claude OAuth token from macOS Keychain (Claude Max subscription)."""

from __future__ import annotations

import json
import re
import subprocess

SETUP_TOKEN_PREFIX = "sk-ant-oat01-"
SETUP_TOKEN_MIN_LENGTH = 80
KEYCHAIN_SERVICE = "Claude Code-credentials"

# Beta headers required for OAuth token usage
OAUTH_BETA_HEADERS = {
    "anthropic-beta": "claude-code-20250219,oauth-2025-04-20",
    "user-agent": "claude-cli/2.1.2 (external, cli)",
    "x-app": "cli",
}


def read_keychain_token() -> str | None:
    """Read the Claude OAuth setup-token from macOS Keychain.

    Returns the token string or None if not found/invalid.
    """
    try:
        raw = subprocess.run(
            ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-w"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if raw.returncode != 0:
            return None

        credential_data = raw.stdout.strip()
        if not credential_data:
            return None

        return _extract_token(credential_data)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def _extract_token(raw: str) -> str | None:
    """Extract the OAuth access token from keychain JSON data."""
    # Try normal JSON parse first
    try:
        parsed = json.loads(raw)
        token = parsed.get("claudeAiOauth", {}).get("accessToken")
        if isinstance(token, str) and _is_valid_token(token):
            return token
    except json.JSONDecodeError:
        pass

    # JSON may be truncated (multi-provider storage) — regex fallback
    match = re.search(r'"accessToken"\s*:\s*"(sk-ant-oat01-[^"]+)"', raw)
    if match and _is_valid_token(match.group(1)):
        return match.group(1)

    return None


def _is_valid_token(token: str) -> bool:
    return token.startswith(SETUP_TOKEN_PREFIX) and len(token) >= SETUP_TOKEN_MIN_LENGTH


def refresh_cli_token() -> bool:
    """Trigger Claude CLI to refresh the OAuth token."""
    try:
        subprocess.run(
            ["claude", "auth", "status"],
            capture_output=True,
            timeout=15,
            env={"CLAUDECODE": "", "PATH": "/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin"},
        )
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
