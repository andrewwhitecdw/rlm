"""Test script for RLM with Z.ai backend."""

import os

from dotenv import load_dotenv

from rlm import RLM

load_dotenv()

api_key = os.getenv("ZAI_API_KEY")
if not api_key:
    raise ValueError("ZAI_API_KEY environment variable is required")

rlm = RLM(
    backend="openai",
    backend_kwargs={
        "model_name": "glm-5",
        "api_key": api_key,
        "base_url": "https://api.z.ai/api/paas/v4/",
    },
    environment="local",
    max_depth=1,
    verbose=True,
)

result = rlm.completion("What is 2+2? Just give me the number.")
print(f"Result: {result.response}")
