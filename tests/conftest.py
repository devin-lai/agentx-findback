from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agentx.api.app import create_app
from agentx.config import Settings
from agentx.vision import weights


@pytest.fixture(autouse=True)
def fresh_weight_cache():
    """Tests substitute fake models; none may leak into another test through the cache."""
    weights.clear()
    yield
    weights.clear()


@pytest.fixture
def app(tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        database_url="",
        api_token="",
        stepfun_api_key="",
        stepfun_model="",
        cosmos_base_url="",
        cosmos_backend="http",
        cosmos_model_path="",
        worker_enabled=False,
        web_dist=Path("not-built"),
        _env_file=None,
    )
    return create_app(settings)


@pytest.fixture
def client(app):
    with TestClient(app) as client:
        yield client


@pytest.fixture
def indexed(client, app):
    result = client.post("/api/v1/demo")
    assert result.status_code == 201, result.text
    video = result.json()
    result = client.post(f"/api/v1/videos/{video['id']}/runs", json={})
    assert result.status_code == 202, result.text
    run = result.json()
    app.state.services.indexer.process(run["id"])
    run = client.get(f"/api/v1/runs/{run['id']}").json()
    assert run["status"] == "complete", run
    return video, run
