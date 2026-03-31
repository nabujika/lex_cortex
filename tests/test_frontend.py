from fastapi.testclient import TestClient

from app.main import app


def test_frontend_root_serves_html():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Legal RAG Console" in response.text

