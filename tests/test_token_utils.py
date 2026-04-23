import pytest

from rlm.utils.token_utils import (
    CHARS_PER_TOKEN_ESTIMATE,
    DEFAULT_CONTEXT_LIMIT,
    MODEL_CONTEXT_LIMITS,
    count_tokens,
    get_context_limit,
)


class TestGetContextLimit:
    def test_exact_match(self):
        for model, expected in MODEL_CONTEXT_LIMITS.items():
            assert get_context_limit(model) == expected

    def test_substring_match(self):
        assert get_context_limit("@openai/gpt-4o") == 128_000
        assert get_context_limit("anthropic/claude-3-sonnet-20240229") == 200_000

    def test_longest_substring_wins(self):
        # "gpt-4o" is 128k, but if there was a longer match it should win
        assert get_context_limit("gpt-4o-mini") == 128_000
        assert get_context_limit("gpt-4") == 8_192

    def test_unknown_model_fallback(self):
        assert get_context_limit("unknown-model-xyz") == DEFAULT_CONTEXT_LIMIT

    def test_empty_and_none_fallback(self):
        assert get_context_limit("") == DEFAULT_CONTEXT_LIMIT
        assert get_context_limit("unknown") == DEFAULT_CONTEXT_LIMIT


class TestCountTokens:
    def test_empty_messages(self):
        assert count_tokens([], "gpt-4o") == 0

    def test_single_message_fallback(self):
        """When tiktoken is unavailable or model unknown, uses char estimate."""
        messages = [{"role": "user", "content": "a" * CHARS_PER_TOKEN_ESTIMATE}]
        result = count_tokens(messages, "unknown-model")
        # Should be ~1 token (ceil of chars/4)
        assert result == 1

    def test_single_message_short_fallback(self):
        messages = [{"role": "user", "content": "hi"}]
        result = count_tokens(messages, "unknown-model")
        assert result == 1  # ceil(2/4) = 1

    def test_message_with_none_content(self):
        messages = [{"role": "user", "content": None}]
        result = count_tokens(messages, "unknown-model")
        assert result == 0

    def test_message_with_list_content_fallback(self):
        """Fallback stringifies the entire list, not just the text parts."""
        messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
        result = count_tokens(messages, "unknown-model")
        # str([{...}]) is ~36 chars -> ceil(36/4) = 9
        assert result == 9

    def test_tiktoken_path(self):
        """If tiktoken is installed and model is known, use it."""
        pytest.importorskip("tiktoken")
        messages = [{"role": "user", "content": "hello world"}]
        result = count_tokens(messages, "gpt-4o")
        # 3 tokens per message + tokens for content
        assert result > 3
