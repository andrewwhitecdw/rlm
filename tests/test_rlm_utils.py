from rlm.utils.rlm_utils import filter_sensitive_keys


class TestFilterSensitiveKeys:
    def test_filters_api_key(self):
        kwargs = {"api_key": "secret", "model": "gpt-4o"}
        result = filter_sensitive_keys(kwargs)
        assert "api_key" not in result
        assert result == {"model": "gpt-4o"}

    def test_filters_api_key_variants(self):
        kwargs = {"API_KEY": "secret", "ApiKey": "secret", "my_api_key": "secret"}
        result = filter_sensitive_keys(kwargs)
        assert len(result) == 0

    def test_keeps_non_sensitive_keys(self):
        kwargs = {"model": "gpt-4o", "temperature": 0.7, "base_url": "http://localhost"}
        result = filter_sensitive_keys(kwargs)
        assert result == kwargs

    def test_empty_dict(self):
        assert filter_sensitive_keys({}) == {}

    def test_nested_dicts_are_not_traversed(self):
        """filter_sensitive_keys only operates on top-level keys."""
        kwargs = {"config": {"api_key": "nested_secret"}, "model": "gpt-4o"}
        result = filter_sensitive_keys(kwargs)
        assert "config" in result
        assert result["config"]["api_key"] == "nested_secret"
