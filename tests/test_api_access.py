"""Browser requests cannot borrow the local workspace or a cookie's authority."""

import pytest
from fastapi.testclient import TestClient

from agentx.api.app import create_app
from agentx.config import Settings


@pytest.fixture
def access_app(tmp_path):
    app = create_app(
        Settings(
            data_dir=tmp_path, api_token="operator-fixture", worker_enabled=False, _env_file=None
        )
    )
    writes = []

    @app.post("/api/v1/access-probe")
    def write():
        writes.append(True)
        return {"written": True}

    return app, writes


def test_non_ascii_login_is_rejected_without_server_error(access_app):
    app, _ = access_app
    with TestClient(app) as client:
        result = client.post("/api/auth/login", json={"token": "invalid-秘密"})
        assert result.status_code == 401
        assert "agentx_session" not in client.cookies


@pytest.mark.parametrize("authorization", ["Basic ignored", "Digest ignored", "Bearer wrong"])
def test_invalid_authorization_cannot_fall_back_to_cookie(access_app, authorization):
    app, writes = access_app
    with TestClient(app) as client:
        assert client.post("/api/auth/login", json={"token": "operator-fixture"}).status_code == 200
        result = client.post(
            "/api/v1/access-probe",
            headers={"Authorization": authorization, "Origin": "https://untrusted.example"},
        )
        assert result.status_code == 401
        assert not writes


@pytest.mark.parametrize("token", ["", "operator-fixture"])
@pytest.mark.parametrize(
    "headers", [{"Origin": "https://untrusted.example"}, {"Sec-Fetch-Site": "cross-site"}]
)
def test_cross_site_writes_are_blocked_in_local_and_cookie_modes(access_app, token, headers):
    app, writes = access_app
    app.state.services.settings.api_token = token
    with TestClient(app) as client:
        if token:
            client.post("/api/auth/login", json={"token": token})
        assert client.post("/api/v1/access-probe", headers=headers).status_code == 403
        assert not writes
        assert (
            client.post("/api/v1/access-probe", headers={"Origin": "http://testserver"}).status_code
            == 200
        )
        assert writes == [True]


def test_valid_bearer_remains_usable_by_explicit_api_clients(access_app):
    app, writes = access_app
    with TestClient(app) as client:
        result = client.post(
            "/api/v1/access-probe",
            headers={
                "Authorization": "bearer operator-fixture",
                "Origin": "https://client.example",
            },
        )
        assert result.status_code == 200
        assert writes == [True]


def test_token_free_loopback_access_rejects_foreign_host(access_app):
    app, _ = access_app
    app.state.services.settings.api_token = ""
    with TestClient(app) as client:
        assert (
            client.get("/api/v1/videos", headers={"Host": "untrusted.example"}).status_code == 503
        )
        assert client.get("/api/v1/videos", headers={"Host": "localhost"}).status_code == 200


@pytest.mark.parametrize("path", ["/api/auth/login", "/api/auth/logout"])
def test_cross_site_requests_cannot_change_the_login_session(access_app, path):
    app, _ = access_app
    with TestClient(app) as client:
        assert (
            client.post(
                path,
                json={"token": "operator-fixture"},
                headers={"Origin": "https://untrusted.example"},
            ).status_code
            == 403
        )
        assert "agentx_session" not in client.cookies
