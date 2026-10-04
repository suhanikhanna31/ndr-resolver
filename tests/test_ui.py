from fastapi.testclient import TestClient

from app.main import app, get_service
from app.service import DecisionService


class _Stub:
    def extract(self, req):  # never called by these tests
        raise AssertionError


def _client():
    app.dependency_overrides[get_service] = lambda: DecisionService(_Stub())
    return TestClient(app)


def teardown_function():
    app.dependency_overrides.clear()


def test_ui_is_served_with_assets():
    c = _client()
    r = c.get("/")
    assert r.status_code == 200 and "NDR/RESOLVER" in r.text
    assert c.get("/app.js").status_code == 200
    assert c.get("/style.css").status_code == 200


def test_api_routes_win_over_static_mount():
    c = _client()
    assert c.get("/healthz").json() == {"ok": True}
    assert c.get("/v1/policy").json()["min_confidence_irreversible"] == 0.9


def test_ledger_and_stats_503_without_database():
    c = _client()
    assert c.get("/v1/decisions").status_code == 503
    assert c.get("/v1/decisions?limit=0").status_code == 422  # bounds enforced


def test_meta_is_public_and_reports_auth(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    with TestClient(app) as c:  # runs lifespan, so app.state is populated
        m = c.get("/meta").json()
        assert m["auth_required"] is True and m["persistence"] is False
        assert c.get("/v1/policy").status_code == 401
        assert c.get("/v1/policy", headers={"x-api-key": "secret"}).status_code == 200
