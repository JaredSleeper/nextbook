import json
import shutil

from nextbook.recommend import Params, load


def test_recommend_excludes_seed_exclude_disliked_and_seed_series(fixture_data):
    engine = load(fixture_data)
    results = engine.recommend(seed_ids=[0], k=20, exclude={2}, disliked=[5])
    result_ids = {index for index, _ in results}
    assert 0 not in result_ids
    assert 1 not in result_ids
    assert 2 not in result_ids
    assert 5 not in result_ids


def test_recommend_keeps_lowest_numbered_book_in_series(fixture_data):
    engine = load(fixture_data)
    results = engine.recommend(seed_ids=[9], k=20)
    result_ids = {index for index, _ in results}
    assert 4 in result_ids
    assert 3 not in result_ids


def test_load_supports_two_engines_at_once(fixture_data, tmp_path):
    first = load(fixture_data)
    second_dir = tmp_path / "second"
    shutil.copytree(fixture_data, second_dir)
    catalog_path = second_dir / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog[0]["title"] = "A different catalog"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    second = load(second_dir)

    assert first is not second
    assert first.catalog[0]["title"] != second.catalog[0]["title"]
    assert second.data_dir == second_dir


def test_recommend_zero_k_returns_no_results(fixture_data):
    engine = load(fixture_data)
    assert engine.recommend(seed_ids=[9], k=0) == []


def test_recommend_filters_derivatives_and_seed_adaptations(fixture_data):
    engine = load(fixture_data)
    engine.catalog[4]["is_derivative"] = True
    engine.catalog[10]["title"] = "Book 10: Graphic Novel"
    results = engine.recommend(seed_ids=[9], k=20)
    result_ids = {index for index, _ in results}
    assert 4 not in result_ids
    assert 10 not in result_ids


def test_rows_filter_derivatives_and_external_seed_adaptations(fixture_data):
    engine = load(fixture_data)
    engine.catalog[4]["is_derivative"] = True
    engine.catalog[10]["title"] = "Book 11: Graphic Novel"
    rows = engine.build_rows(seed_ids=[9], external_seed_titles=["Book 11"])
    result_ids = {book["id"] for row in rows for book in row["books"]}
    assert engine.catalog[4]["id"] not in result_ids
    assert engine.catalog[10]["id"] not in result_ids


def test_seed_series_comparison_ignores_case_article_and_punctuation(fixture_data):
    engine = load(fixture_data)
    engine.catalog[1]["series"] = "The HARRY.POTTER"
    results = engine.recommend(seed_ids=[0], k=20)
    assert 1 not in {index for index, _ in results}


def test_default_popularity_penalty():
    assert Params().pop_beta == 0.15
