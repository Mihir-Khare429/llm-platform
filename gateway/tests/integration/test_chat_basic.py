import pytest
from fastapi.testclient import TestClient
from app.main import get_app

app = get_app()
client = TestClient(app)

def test_missing_messages_422():
    resp = client.post("/v1/chat/completions", json={"model": "x"})
    assert resp.status_code == 422

def test_stream_not_supported_400():
    body = {
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True
    }
    resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 400
