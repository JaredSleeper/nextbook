import csv

import numpy as np
import pytest

from scripts.build_data import build_cf


def test_holdout_split_is_deterministic_and_excludes_users(tmp_path):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "eval"
    raw_dir.mkdir()
    out_dir.mkdir()
    ratings = [
        (1, 1, 5),
        (1, 2, 4),
        (2, 1, 3),
        (2, 2, 5),
        (3, 1, 5),
        (3, 3, 4),
        (4, 4, 2),
    ]
    with (raw_dir / "ratings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["user_id", "book_id", "rating"])
        writer.writerows(ratings)
    catalog = [{"id": book_id} for book_id in range(1, 5)]
    build_cf(raw_dir, catalog, out_dir, 500, 0.5, 42)

    expected_holdout = np.random.default_rng(42).permutation(np.array([1, 2, 3, 4]))[:2]
    np.testing.assert_array_equal(np.load(out_dir / "holdout_users.npy"), expected_holdout)
    with np.load(out_dir / "cf_cosine.npz") as artifact:
        indices, weights = artifact["indices"], artifact["weights"]
    neighbor_position = int(np.flatnonzero(indices[0] == 1)[0])
    held_out = set(map(int, expected_holdout))
    train = [(user, book) for user, book, rating in ratings if rating >= 4 and user not in held_out]
    item_0 = sum(book == 1 for _, book in train)
    item_1 = sum(book == 2 for _, book in train)
    together = len(
        {user for user, book in train if book == 1} & {user for user, book in train if book == 2}
    )
    expected_cosine = together / (np.sqrt(item_0 * item_1) + 10)
    assert float(weights[0, neighbor_position]) == pytest.approx(expected_cosine, rel=1e-3)
