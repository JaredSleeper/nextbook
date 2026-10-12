import numpy as np
from fastapi.testclient import TestClient

import nextbook.app as app_module


def request_body():
    return {
        "seeds": [{"id": "gb:10", "title": "Book 10", "author": "Writer 9"}],
        "liked": [],
        "disliked": [],
        "read": [],
    }


def test_recommendation_rows_use_book_api_shape(fixture_data):
    with TestClient(app_module.app) as client:
        response = client.post("/api/recommend", json=request_body())
    assert response.status_code == 200
    rows = response.json()["rows"]
    assert rows[0]["key"] == "top"
    assert rows[0]["title"] == "Top picks for you"
    assert rows[0]["books"]
    assert rows[0]["books"][0]["id"].startswith("gb:")
    assert "ratings_count" in rows[0]["books"][0]


def test_picks_drops_invalid_candidate_ids(fixture_data, monkeypatch):
    def fake_choose(seeds, liked, disliked, prompt, candidates):
        picks = [
            {"id": candidate["id"], "reason": "Connects to Book 10."}
            for candidate in candidates[:6]
        ]
        picks.append({"id": 999999, "reason": "Invented"})
        return picks, 123, 45

    monkeypatch.setattr(app_module.llm, "choose_picks", fake_choose)
    with TestClient(app_module.app) as client:
        response = client.post("/api/picks", json={**request_body(), "prompt": None})
    assert response.status_code == 200
    payload = response.json()
    assert payload["fallback"] is False
    assert len(payload["books"]) == 6
    assert all(book["reason"] == "Connects to Book 10." for book in payload["books"])
    assert all(book["id"] != "gb:999999" for book in payload["books"])
    assert payload["tokens"] == {"input": 123, "output": 45}


def test_picks_falls_back_when_fewer_than_six_valid_ids(fixture_data, monkeypatch):
    def fake_choose(seeds, liked, disliked, prompt, candidates):
        return [{"id": candidates[0]["id"], "reason": "Only one."}], 10, 4

    monkeypatch.setattr(
        app_module, "embed_texts", lambda texts: np.ones((len(texts), 12), dtype=np.float32)
    )
    monkeypatch.setattr(app_module.llm, "choose_picks", fake_choose)
    with TestClient(app_module.app) as client:
        response = client.post("/api/picks", json={**request_body(), "prompt": "funny"})
    payload = response.json()
    assert payload["fallback"] is True
    assert payload["title"] == "For: funny"
    assert len(payload["books"]) >= 6
    assert all(book["reason"] is None for book in payload["books"])


def test_picks_falls_back_after_claude_failure(fixture_data, monkeypatch):
    def failed(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(app_module.llm, "choose_picks", failed)
    with TestClient(app_module.app) as client:
        response = client.post("/api/picks", json=request_body())
    payload = response.json()
    assert payload["fallback"] is True
    assert payload["books"]
    assert all(book["reason"] is None for book in payload["books"])


def test_picks_candidates_exclude_derivatives_and_adaptations(fixture_data, monkeypatch):
    original_load = app_module.load
    captured = {}

    def load_with_derivatives(data_dir):
        engine = original_load(data_dir)
        engine.catalog[2]["is_derivative"] = True
        engine.catalog[5]["title"] = "Book 10: Graphic Novel"
        return engine

    def fake_choose(seeds, liked, disliked, prompt, candidates):
        captured["candidates"] = candidates
        return (
            [
                {"id": candidate["id"], "reason": "Connects to Book 10."}
                for candidate in candidates[:6]
            ],
            123,
            45,
        )

    monkeypatch.setattr(app_module, "load", load_with_derivatives)
    monkeypatch.setattr(app_module.llm, "choose_picks", fake_choose)
    with TestClient(app_module.app) as client:
        response = client.post("/api/picks", json=request_body())
    assert response.status_code == 200
    assert captured["candidates"]
    assert all(not candidate.get("is_derivative") for candidate in captured["candidates"])
    assert all(
        not candidate["title"].startswith("Book 10: Graphic Novel")
        for candidate in captured["candidates"]
    )


def test_search_uses_local_then_dedupes_open_library(fixture_data, monkeypatch):
    async def fake_search(query, limit):
        return [
            {
                "id": "ol:/works/OL1W",
                "title": "Dune",
                "author": "Frank Herbert",
                "year": 1965,
                "cover": None,
                "in_catalog": False,
                "subjects": ["science fiction"],
            },
            {
                "id": "ol:/works/OL2W",
                "title": "Neuromancer",
                "author": "William Gibson",
                "year": 1984,
                "cover": None,
                "in_catalog": False,
                "subjects": ["science fiction"],
            },
        ]

    monkeypatch.setattr(app_module, "search_open_library", fake_search)
    with TestClient(app_module.app) as client:
        response = client.get("/api/search?q=Dune&limit=2")
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 2
    assert results[0]["id"] == "gb:1"
    assert results[1]["id"] == "ol:/works/OL2W"


def test_search_local_source_never_calls_open_library(fixture_data, monkeypatch):
    async def fail_search(*args, **kwargs):
        raise AssertionError("local search called Open Library")

    monkeypatch.setattr(app_module, "search_open_library", fail_search)
    with TestClient(app_module.app) as client:
        response = client.get("/api/search?q=Dune&limit=2&source=local")
    assert response.status_code == 200
    assert response.json()["results"][0]["id"] == "gb:1"
