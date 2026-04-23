"""Tests for LMHandler using MockLM (no real LM required)."""

from rlm.core.comms_utils import LMRequest, send_lm_request, send_lm_request_batched
from rlm.core.lm_handler import LMHandler
from tests.mock_lm import MockLM


def test_lm_handler_single_request():
    """Single prompt request returns success and echo-style content."""
    mock = MockLM(responses=["hello back"])
    with LMHandler(client=mock) as handler:
        request = LMRequest(prompt="hello")
        response = send_lm_request(handler.address, request)
    assert response.success
    assert response.chat_completion is not None
    assert response.chat_completion.response == "hello back"


def test_lm_handler_batched_request():
    """Batched prompts return one response per prompt in order."""
    responses = [f"r{i}" for i in range(5)]
    mock = MockLM(responses=responses)
    with LMHandler(client=mock, batch_max_concurrent=3) as handler:
        prompts = [f"prompt-{i}" for i in range(5)]
        result = send_lm_request_batched(handler.address, prompts)
    assert len(result) == 5
    for i, resp in enumerate(result):
        assert resp.success, resp.error
        assert resp.chat_completion is not None
        assert resp.chat_completion.response == f"r{i}"


def test_lm_handler_batched_many_prompts_semaphore_cap():
    """Many prompts complete successfully with semaphore limiting concurrency."""
    # 50 prompts, max 4 concurrent: should still all complete
    count = 50
    responses = [f"resp-{i}" for i in range(count)]
    mock = MockLM(responses=responses)
    with LMHandler(client=mock, batch_max_concurrent=4) as handler:
        prompts = [f"p-{i}" for i in range(count)]
        result = send_lm_request_batched(handler.address, prompts)
    assert len(result) == count
    for i, resp in enumerate(result):
        assert resp.success, (i, resp.error)
        assert resp.chat_completion.response == f"resp-{i}"


def test_lm_handler_usage_no_double_count_default_client():
    """Default client should not be double-counted in get_usage_summary."""
    mock = MockLM(responses=["hello"])
    with LMHandler(client=mock) as handler:
        # Make one request to bump usage
        request = LMRequest(prompt="hello")
        send_lm_request(handler.address, request)

        summary = handler.get_usage_summary()
        # Default client is registered in self.clients, so it should appear exactly once
        assert "mock-model" in summary.model_usage_summaries
        assert summary.model_usage_summaries["mock-model"].total_calls == 1


def test_lm_handler_usage_with_other_backend_client():
    """Usage should merge default, other_backend, and registered clients correctly."""
    default_mock = MockLM(model_name="default-model", responses=["r1", "r2"])
    other_mock = MockLM(model_name="other-model", responses=["r3"])

    with LMHandler(client=default_mock, other_backend_client=other_mock) as handler:
        third_mock = MockLM(model_name="third-model", responses=["r4"])
        handler.register_client("third-model", third_mock)

        # No model -> default client
        send_lm_request(handler.address, LMRequest(prompt="a"))
        # third-model is registered
        send_lm_request(handler.address, LMRequest(prompt="c", model="third-model"))

        summary = handler.get_usage_summary()
        # default client counted once (not double-counted via explicit merge + loop)
        assert summary.model_usage_summaries["default-model"].total_calls == 1
        # other_backend_client appears with 0 calls since no subcalls made
        assert summary.model_usage_summaries["other-model"].total_calls == 0
        assert summary.model_usage_summaries["third-model"].total_calls == 1


def test_lm_handler_usage_other_backend_for_subcalls():
    """other_backend_client is used when depth > 0."""
    default_mock = MockLM(model_name="default-model", responses=["r1"])
    other_mock = MockLM(model_name="other-model", responses=["r2"])

    with LMHandler(client=default_mock, other_backend_client=other_mock) as handler:
        send_lm_request(handler.address, LMRequest(prompt="a"))
        send_lm_request(handler.address, LMRequest(prompt="b", depth=1))

        summary = handler.get_usage_summary()
        assert summary.model_usage_summaries["default-model"].total_calls == 1
        assert summary.model_usage_summaries["other-model"].total_calls == 1
