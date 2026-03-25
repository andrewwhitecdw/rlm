import os
from typing import Any

from openai import AsyncOpenAI, OpenAI

from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary

DEFAULT_ZAI_API_KEY = os.getenv("ZAI_API_KEY")
DEFAULT_ZAI_BASE_URL = "https://api.z.ai/api/paas/v4/"


class ZaiClient(BaseLM):
    """
    Client for Z.ai API (z.ai). Uses the OpenAI-compatible SDK with a custom base URL.

    Environment Variables:
        ZAI_API_KEY: Z.ai API key (required if not passed explicitly)

    Example:
        from rlm import RLM

        # Using backend="openai" with Z.ai base URL (recommended)
        rlm = RLM(
            backend="openai",
            backend_kwargs={
                "model_name": "glm-5",
                "api_key": os.getenv("ZAI_API_KEY"),
                "base_url": "https://api.z.ai/api/paas/v4/",
            },
        )
        result = rlm.completion("What is 2+2?")
        print(result.response)

        # Or use the dedicated ZaiClient directly
        from rlm.clients.zai import ZaiClient
        client = ZaiClient(api_key="your-key", model_name="glm-5")
        response = client.completion("Hello!")
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "glm-5",
        base_url: str = DEFAULT_ZAI_BASE_URL,
        **kwargs: Any,
    ):
        """
        Initialize Z.ai client.

        Args:
            api_key: Z.ai API key (defaults to ZAI_API_KEY env var)
            model_name: Model name (default: glm-5)
            base_url: API base URL (default: https://api.z.ai/api/paas/v4/)
            **kwargs: Additional arguments passed to OpenAI client
        """
        super().__init__(model_name=model_name, **kwargs)

        if api_key is None:
            api_key = DEFAULT_ZAI_API_KEY
        if api_key is None:
            raise ValueError("Z.ai API key is required. Set ZAI_API_KEY env var or pass api_key.")

        self.api_key = api_key
        self.base_url = base_url
        self._client = OpenAI(api_key=api_key, base_url=base_url, **kwargs)
        self._async_client = AsyncOpenAI(api_key=api_key, base_url=base_url, **kwargs)

        self._usage: dict[str, ModelUsageSummary] = {}
        self._last_usage: ModelUsageSummary | None = None

    def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        """
        Make a synchronous completion request to Z.ai.

        Args:
            prompt: String prompt or list of messages (OpenAI format)
            model: Override model name (defaults to self.model_name)

        Returns:
            Completion text response
        """
        model = model or self.model_name
        messages = self._prepare_messages(prompt)

        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
        )

        self._track_cost(model, response.usage.prompt_tokens, response.usage.completion_tokens)

        return response.choices[0].message.content

    async def acompletion(
        self, prompt: str | list[dict[str, Any]], model: str | None = None
    ) -> str:
        """
        Make an asynchronous completion request to Z.ai.

        Args:
            prompt: String prompt or list of messages (OpenAI format)
            model: Override model name (defaults to self.model_name)

        Returns:
            Completion text response
        """
        model = model or self.model_name

        messages = self._prepare_messages(prompt)

        response = await self._async_client.chat.completions.create(
            model=model,
            messages=messages,
        )

        self._track_cost(model, response.usage.prompt_tokens, response.usage.completion_tokens)

        return response.choices[0].message.content

    def get_usage_summary(self) -> UsageSummary:
        """
        Get aggregated usage summary across all models.

        Returns:
            UsageSummary with per-model usage summaries
        """
        return UsageSummary(model_usage_summaries=self._usage.copy())

    def get_last_usage(self) -> ModelUsageSummary:
        """
        Get usage summary for the most recent call.

        Returns:
            ModelUsageSummary for the last call
        """
        if self._last_usage is None:
            return ModelUsageSummary(
                model=self.model_name, total_calls=0, total_input_tokens=0, total_output_tokens=0
            )
        return self._last_usage

    def _prepare_messages(self, prompt: str | list[dict[str, Any]]) -> list[dict[str, str]]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        return prompt

    def _track_cost(self, model: str, input_tokens: int, output_tokens: int) -> None:
        if model not in self._usage:
            self._usage[model] = ModelUsageSummary(
                total_calls=0,
                total_input_tokens=0,
                total_output_tokens=0,
            )

        usage = self._usage[model]
        usage.total_calls += 1
        usage.total_input_tokens += input_tokens
        usage.total_output_tokens += output_tokens

        self._last_usage = ModelUsageSummary(
            total_calls=usage.total_calls,
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
        )
