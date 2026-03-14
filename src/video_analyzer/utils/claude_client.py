"""Claude client factory — uses OAuth token from macOS Keychain (Max subscription)."""

from __future__ import annotations

import anthropic

from video_analyzer.utils.keychain import OAUTH_BETA_HEADERS, read_keychain_token


def create_claude_client() -> anthropic.Anthropic:
    """Create an Anthropic client using OAuth token from macOS Keychain."""
    token = read_keychain_token()
    if not token:
        raise ValueError(
            "No Claude OAuth token found in macOS Keychain.\n"
            "Log in to Claude Code CLI first: `claude auth login`"
        )

    return anthropic.Anthropic(
        api_key=None,
        auth_token=token,
        default_headers=OAUTH_BETA_HEADERS,
    )


def create_async_claude_client() -> anthropic.AsyncAnthropic:
    """Async version — uses OAuth token from macOS Keychain."""
    token = read_keychain_token()
    if not token:
        raise ValueError(
            "No Claude OAuth token found in macOS Keychain.\n"
            "Log in to Claude Code CLI first: `claude auth login`"
        )

    return anthropic.AsyncAnthropic(
        api_key=None,
        auth_token=token,
        default_headers=OAUTH_BETA_HEADERS,
    )
