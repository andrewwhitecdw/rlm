"""Test script for ZaiClient directly."""

import os

from dotenv import load_dotenv

from rlm.clients.zai import ZaiClient

load_dotenv()

api_key = os.getenv("ZAI_API_KEY")
if not api_key:
    raise ValueError("ZAI_API_KEY environment variable is required")

client = ZaiClient(api_key=api_key)

print("Testing ZaiClient...")
print(f"Model: {client.model_name}")

result = client.completion("Say hello!")
print(f"\nResult: {result}")

usage = client.get_usage_summary()
print("\nUsage summary per model:")
for model, summary in usage.model_usage_summaries.items():
    print(f"  {model}:")
    print(f"    calls: {summary.total_calls}")
    print(f"    input tokens: {summary.total_input_tokens}")
    print(f"    output tokens: {summary.total_output_tokens}")
