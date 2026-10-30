from catalog_agent.config import Settings


def test_defaults_route_at_the_validated_cutoff():
    s = Settings()
    assert s.accept_threshold == 0.72
    assert s.vllm_url.startswith("http://127.0.0.1")


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("AGENT_CONCURRENCY", "8")
    assert Settings().concurrency == 8
