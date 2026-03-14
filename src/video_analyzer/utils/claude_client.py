"""Claude client factory — supports OAuth (Max subscription) and API key."""

from __future__ import annotations

import anthropic

from video_analyzer.utils.keychain import OAUTH_BETA_HEADERS, SETUP_TOKEN_PREFIX, read_keychain_token


def create_claude_client(api_key: str = "") -> tuple[anthropic.Anthropic, str]:
    """Create an Anthropic client with the best available auth method.

    Priority:
    1. Keychain OAuth token (Claude Max subscription — free)
    2. Explicit API key (paid per-token)

    Returns (client, auth_method) tuple.
    """
    # Try keychain first (Max subscription)
    keychain_token = read_keychain_token()
    if keychain_token:
        client = anthropic.Anthropic(
            api_key=None,
            auth_token=keychain_token,
            default_headers=OAUTH_BETA_HEADERS,
        )
        return client, "oauth-max"

    # Fall back to API key
    if api_key:
        client = anthropic.Anthropic(api_key=api_key)
        return client, "api-key"

    raise ValueError(
        "No Claude credentials found. Either:\n"
        "  1. Log in to Claude Code CLI (`claude auth login`) for Max subscription access\n"
        "  2. Set ANTHROPIC_API_KEY in .env for paid API access"
    )


def create_async_claude_client(api_key: str = "") -> tuple[anthropic.AsyncAnthropic, str]:
    """Async version of create_claude_client."""
    keychain_token = read_keychain_token()
    if keychain_token:
        client = anthropic.AsyncAnthropic(
            api_key=None,
            auth_token=keychain_token,
            default_headers=OAUTH_BETA_HEADERS,
        )
        return client, "oauth-max"

    if api_key:
        client = anthropic.AsyncAnthropic(api_key=api_key)
        return client, "api-key"

    raise ValueError(
        "No Claude credentials found. Either:\n"
        "  1. Log in to Claude Code CLI (`claude auth login`) for Max subscription access\n"
        "  2. Set ANTHROPIC_API_KEY in .env for paid API access"
    )
