"""Fast API tests: the endpoints, validation and error handling with stand-in models (no model loading)."""
import pytest
from fastapi.testclient import TestClient

from reviewsense.serving.api import Components, create_app


class StandInStars:
    def stars(self, texts):
        return [4.5 for _ in texts]


class StandInSearch:
    def search(self, query, k=5, business=None):
        return {"corrected": None, "results": [{"text": "Great tacos.", "business": "Taqueria El Sol",
                                                "review_id": "r10", "score": 0.9}][:k]}


class StandInChatbot:
    def ask(self, session_id, message):
        return {"answer": f"echo: {message}", "action": "answer", "blocked": None, "sources": []}


@pytest.fixture
def client():
    with TestClient(create_app(Components(StandInStars(), StandInSearch(), StandInChatbot()))) as test_client:
        yield test_client


def test_health_ready_and_request_ids(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json() == {"status": "ready", "chat": True}
    response = client.get("/health", headers={"X-Request-ID": "abc123"})
    assert response.headers["X-Request-ID"] == "abc123"                  # passed through, for tracing a request


def test_endpoints(client):
    assert client.post("/stars", json={"texts": ["Great", "Bad"]}).json() == {"stars": [4.5, 4.5]}
    hits = client.post("/search", json={"query": "tacos", "k": 1}).json()["results"]
    assert hits[0]["review_id"] == "r10"
    assert client.post("/chat", json={"session_id": "s1", "message": "hi"}).json()["answer"] == "echo: hi"


@pytest.mark.parametrize("path, body", [
    ("/stars", {"texts": []}),                                   # nothing to rate
    ("/stars", {"texts": ["x"] * 33}),                           # too many at once
    ("/stars", {"texts": ["x" * 6000]}),                         # too long
    ("/stars", {"text": "wrong field name"}),
    ("/search", {"query": "tacos", "k": 100}),
    ("/chat", {"session_id": "s1", "message": ""}),
])
def test_bad_input_is_rejected_with_422(client, path, body):
    assert client.post(path, json=body).status_code == 422


def test_chat_switched_off_returns_503():
    with TestClient(create_app(Components(StandInStars(), StandInSearch(), chatbot=None))) as client:
        assert client.post("/chat", json={"session_id": "s1", "message": "hi"}).status_code == 503
        assert client.get("/ready").json()["chat"] is False
