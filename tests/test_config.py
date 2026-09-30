from reviewsense.config import Settings


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("REVIEWSENSE_LLM_MODEL", "my-org/my-model")
    monkeypatch.setenv("REVIEWSENSE_VECTOR_INDEX", "IVF64,PQ16")
    s = Settings()
    assert s.llm_model == "my-org/my-model" and s.vector_index == "IVF64,PQ16"