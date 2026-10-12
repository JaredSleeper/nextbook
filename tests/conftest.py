import json

import numpy as np
import pytest

import nextbook.app as app_module


@pytest.fixture(autouse=True)
def no_external_keys(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    app_module._PICKS_CACHE.clear()
    app_module._SEARCH_CACHE.clear()
    app_module._EXTERNAL_SUBJECTS.clear()
    app_module._SEED_VECTOR_CACHE.clear()


@pytest.fixture
def fixture_data(tmp_path, monkeypatch):
    catalog = []
    for index in range(12):
        catalog.append(
            {
                "id": index + 1,
                "title": "Dune" if index == 0 else f"Book {index + 1}",
                "series": (
                    "Harry Potter" if index in {0, 1} else "Orbit" if index in {3, 4} else None
                ),
                "series_number": (1 if index in {0, 4} else 2 if index in {1, 3} else None),
                "author": "Frank Herbert" if index in {0, 2} else f"Writer {index}",
                "authors": [],
                "year": 2000 + index,
                "avg_rating": 4.0,
                "ratings_count": 1000 - index,
                "isbn": "9780441172719",
                "goodreads_book_id": f"gr-{index + 1}",
                "cover": None,
                "tags": ["science fiction"],
                "description": f"Description for book {index + 1}",
                "openlibrary_work_key": None,
            }
        )
    data_dir = tmp_path / "artifacts"
    data_dir.mkdir()
    (data_dir / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
    embeddings = np.eye(12, dtype=np.float16)
    np.save(data_dir / "emb.npy", embeddings)
    indices = np.empty((12, 11), dtype=np.int32)
    weights = np.empty((12, 11), dtype=np.float16)
    for row in range(12):
        indices[row] = [index for index in range(12) if index != row]
        weights[row] = np.linspace(1.0, 0.1, 11)
    for name in ("cf_ease.npz", "cf_cosine.npz"):
        np.savez_compressed(data_dir / name, indices=indices, weights=weights)
    monkeypatch.setenv("NEXTBOOK_DATA_DIR", str(data_dir))
    return data_dir
