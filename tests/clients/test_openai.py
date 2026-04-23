"""Tests for the OpenAI client."""

from unittest.mock import MagicMock, patch

import pytest

from rlm.clients.openai import OpenAIClient
from rlm.core.types import ModelUsageSummary, UsageSummary


class TestOpenAIClientUnit:
    """Unit tests that don't require API calls."""

    def test_init_with_api_key(self):
        with patch("rlm.clients.openai.openai.OpenAI"):
            client = OpenAIClient(api_key="test-key", model_name="gpt-4o")
            assert client.model_name == "gpt-4o"

    def test_init_default_model(self):
        with patch("rlm.clients.openai.openai.OpenAI"):
            client = OpenAIClient(api_key="test-key")
            assert client.model_name is None

    def test_completion_string_prompt(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello!"
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 3
        mock_response.usage.total_tokens = 8
        mock_response.usage.cost = None
        mock_response.usage.model_extra = None

        with patch("rlm.clients.openai.openai.OpenAI") as mock_client_class:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = OpenAIClient(api_key="test-key", model_name="gpt-4o")
            result = client.completion("Hello")

            assert result == "Hello!"
            mock_client.chat.completions.create.assert_called_once()
            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert call_kwargs["model"] == "gpt-4o"
            assert call_kwargs["messages"] == [{"role": "user", "content": "Hello"}]

    def test_completion_message_list_prompt(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hi there!"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.usage.cost = None
        mock_response.usage.model_extra = None

        with patch("rlm.clients.openai.openai.OpenAI") as mock_client_class:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = OpenAIClient(api_key="test-key", model_name="gpt-4o")
            messages = [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
            ]
            result = client.completion(messages)

            assert result == "Hi there!"
            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert call_kwargs["messages"] == messages

    def test_completion_requires_model(self):
        with patch("rlm.clients.openai.openai.OpenAI"):
            client = OpenAIClient(api_key="test-key", model_name=None)
            with pytest.raises(ValueError, match="Model name is required"):
                client.completion("Hello")

    def test_completion_invalid_prompt_type(self):
        with patch("rlm.clients.openai.openai.OpenAI"):
            client = OpenAIClient(api_key="test-key", model_name="gpt-4o")
            with pytest.raises(ValueError, match="Invalid prompt type"):
                client.completion(12345)

    def test_track_cost_raises_on_missing_usage(self):
        mock_response = MagicMock()
        mock_response.usage = None

        with patch("rlm.clients.openai.openai.OpenAI"):
            client = OpenAIClient(api_key="test-key", model_name="gpt-4o")
            with pytest.raises(ValueError, match="No usage data received"):
                client._track_cost(mock_response, "gpt-4o")

    def test_usage_tracking(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "OK"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.usage.cost = None
        mock_response.usage.model_extra = None

        with patch("rlm.clients.openai.openai.OpenAI") as mock_client_class:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = OpenAIClient(api_key="test-key", model_name="gpt-4o")
            client.completion("Hello")
            client.completion("World")

            summary = client.get_usage_summary()
            assert isinstance(summary, UsageSummary)
            assert summary.model_usage_summaries["gpt-4o"].total_calls == 2
            assert summary.model_usage_summaries["gpt-4o"].total_input_tokens == 20
            assert summary.model_usage_summaries["gpt-4o"].total_output_tokens == 10

    def test_get_last_usage(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "OK"
        mock_response.usage.prompt_tokens = 7
        mock_response.usage.completion_tokens = 3
        mock_response.usage.total_tokens = 10
        mock_response.usage.cost = None
        mock_response.usage.model_extra = None

        with patch("rlm.clients.openai.openai.OpenAI") as mock_client_class:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = OpenAIClient(api_key="test-key", model_name="gpt-4o")
            client.completion("Hello")

            last = client.get_last_usage()
            assert isinstance(last, ModelUsageSummary)
            assert last.total_calls == 1
            assert last.total_input_tokens == 7
            assert last.total_output_tokens == 3

    def test_extra_body_for_prime(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "OK"
        mock_response.usage.prompt_tokens = 1
        mock_response.usage.completion_tokens = 1
        mock_response.usage.total_tokens = 2
        mock_response.usage.cost = None
        mock_response.usage.model_extra = None

        with patch("rlm.clients.openai.openai.OpenAI") as mock_client_class:
            mock_client = MagicMock()
            mock_client.base_url = "https://api.pinference.ai/api/v1/"
            mock_client.chat.completions.create.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = OpenAIClient(
                api_key="test-key",
                model_name="some-model",
                base_url="https://api.pinference.ai/api/v1/",
            )
            client.completion("Hello")

            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert call_kwargs["extra_body"] == {"usage": {"include": True}}

    def test_cost_extraction_from_openrouter(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "OK"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.usage.cost = 0.00123

        with patch("rlm.clients.openai.openai.OpenAI") as mock_client_class:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = OpenAIClient(api_key="test-key", model_name="gpt-4o")
            client.completion("Hello")

            assert client.last_cost == 0.00123
            assert client.model_costs["gpt-4o"] == 0.00123

    def test_cost_extraction_from_model_extra(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "OK"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.usage.cost = None
        mock_response.usage.model_extra = {"cost": 0.00045}

        with patch("rlm.clients.openai.openai.OpenAI") as mock_client_class:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = OpenAIClient(api_key="test-key", model_name="gpt-4o")
            client.completion("Hello")

            assert client.last_cost == 0.00045


