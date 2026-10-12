import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from nextbook.catalog import is_seed_adaptation, normalize_series


@dataclass
class Params:
    cf_method: str = "ease"
    w_cf: float = 0.7
    w_content: float = 0.3
    pop_beta: float = 0.15


_DEFAULT_PARAMS = Params()


def _zscore(values: np.ndarray) -> np.ndarray:
    if not len(values):
        return values.astype(np.float32)
    std = float(values.std())
    if std < 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - float(values.mean())) / std).astype(np.float32)


class Engine:
    def __init__(
        self,
        data_dir: str | Path,
        catalog: list[dict],
        embeddings: np.ndarray,
        ease_indices: np.ndarray,
        ease_weights: np.ndarray,
        cosine_indices: np.ndarray,
        cosine_weights: np.ndarray,
    ):
        self.data_dir = Path(data_dir)
        self.catalog = catalog
        self.embeddings = embeddings.astype(np.float32)
        self.ease_indices = ease_indices
        self.ease_weights = ease_weights.astype(np.float32)
        self.cosine_indices = cosine_indices
        self.cosine_weights = cosine_weights.astype(np.float32)
        self.id_to_index = {int(book["id"]): index for index, book in enumerate(catalog)}
        self.series_keys = [normalize_series(book.get("series")) for book in catalog]
        self.work_to_index = {
            book["openlibrary_work_key"]: index
            for index, book in enumerate(catalog)
            if book.get("openlibrary_work_key")
        }

    def _matrix(self, method: str) -> tuple[np.ndarray, np.ndarray]:
        if method == "ease":
            return self.ease_indices, self.ease_weights
        if method == "cosine":
            return self.cosine_indices, self.cosine_weights
        raise ValueError(f"Unknown CF method: {method}")

    def _seed_context(
        self, seed_ids: list[int], external_seed_titles: list[str]
    ) -> tuple[set[str], list[str]]:
        seed_series = {self.series_keys[index] for index in seed_ids}
        seed_series.discard("")
        seed_titles = [self.catalog[index].get("title", "") for index in seed_ids]
        seed_titles.extend(external_seed_titles)
        return seed_series, seed_titles

    def _candidate_allowed(
        self,
        index: int,
        blocked: set[int],
        seed_series: set[str],
        seed_titles: list[str],
    ) -> bool:
        book = self.catalog[index]
        return (
            index not in blocked
            and not book.get("is_derivative", False)
            and self.series_keys[index] not in seed_series
            and not is_seed_adaptation(book.get("title", ""), seed_titles)
        )

    def _taste_vector(
        self,
        seed_ids: list[int],
        extra_vectors: list[np.ndarray],
        liked: list[int],
        disliked: list[int],
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        positive_parts = [self.embeddings[index] for index in seed_ids]
        positive_weights = [1.0] * len(positive_parts)
        positive_parts.extend(
            np.asarray(vector, dtype=np.float32).reshape(-1) for vector in extra_vectors
        )
        positive_weights.extend([1.0] * len(extra_vectors))
        positive_parts.extend(self.embeddings[index] for index in liked)
        positive_weights.extend([0.5] * len(liked))
        taste = None
        if positive_parts:
            taste = np.average(np.stack(positive_parts), axis=0, weights=positive_weights)
            norm = np.linalg.norm(taste)
            if norm:
                taste /= norm
        dislike = None
        if disliked:
            dislike = self.embeddings[disliked].mean(axis=0)
            norm = np.linalg.norm(dislike)
            if norm:
                dislike /= norm
        return taste, dislike

    def score_pool(
        self,
        seed_ids: list[int],
        params: Params = _DEFAULT_PARAMS,
        extra_vectors: list[np.ndarray] = (),
        liked: list[int] = (),
        disliked: list[int] = (),
    ) -> tuple[list[tuple[int, float]], np.ndarray, np.ndarray]:
        if params.w_cf < 0 or params.w_content < 0:
            raise ValueError("Recommendation weights must be non-negative")
        neighbor_indices, neighbor_weights = self._matrix(params.cf_method)
        cf = np.zeros(len(self.catalog), dtype=np.float32)
        for seed in seed_ids:
            valid = neighbor_weights[seed] != 0
            np.add.at(cf, neighbor_indices[seed, valid], neighbor_weights[seed, valid])
        for seed in liked:
            valid = neighbor_weights[seed] != 0
            np.add.at(cf, neighbor_indices[seed, valid], 0.5 * neighbor_weights[seed, valid])
        if params.pop_beta:
            ratings = np.asarray([max(1, book.get("ratings_count", 0)) for book in self.catalog])
            cf /= np.power(ratings, params.pop_beta)
        taste, dislike_vector = self._taste_vector(
            seed_ids, list(extra_vectors), list(liked), list(disliked)
        )
        if taste is None:
            content = np.zeros(len(self.catalog), dtype=np.float32)
        else:
            content = self.embeddings @ taste
        if dislike_vector is not None:
            content -= 0.3 * (self.embeddings @ dislike_vector)
        candidates = set(np.flatnonzero(cf).tolist())
        if taste is not None:
            content_candidates = np.argpartition(content, -min(500, len(content)))[
                -min(500, len(content)) :
            ]
            candidates.update(map(int, content_candidates))
        candidate_indices = np.asarray(sorted(candidates), dtype=np.int32)
        if not len(candidate_indices):
            return [], cf, content
        cf_z = _zscore(cf[candidate_indices])
        content_z = _zscore(content[candidate_indices])
        hybrid = params.w_cf * cf_z + params.w_content * content_z
        order = np.argsort(-hybrid, kind="stable")
        ranked = [(int(candidate_indices[position]), float(hybrid[position])) for position in order]
        return ranked, cf, content

    def recommend(
        self,
        seed_ids: list[int],
        k: int = 50,
        exclude: set[int] = frozenset(),
        params: Params = _DEFAULT_PARAMS,
        extra_vectors: list[np.ndarray] = (),
        liked: list[int] = (),
        disliked: list[int] = (),
        external_seed_titles: list[str] = (),
    ) -> list[tuple[int, float]]:
        if k <= 0:
            return []
        ranked, _, _ = self.score_pool(seed_ids, params, extra_vectors, liked, disliked)
        blocked = set(seed_ids) | set(exclude) | set(disliked)
        seed_series, seed_titles = self._seed_context(seed_ids, external_seed_titles)
        eligible = []
        for index, score in ranked:
            if not self._candidate_allowed(index, blocked, seed_series, seed_titles):
                continue
            eligible.append((index, score))
        min_number: dict[str, float] = {}
        for index, _ in eligible:
            book = self.catalog[index]
            series = self.series_keys[index]
            number = book.get("series_number")
            if series and isinstance(number, (int, float)) and not isinstance(number, bool):
                min_number[series] = min(number, min_number.get(series, number))
        result = []
        seen_series = set()
        for index, score in eligible:
            book = self.catalog[index]
            series = self.series_keys[index]
            number = book.get("series_number")
            if series and series in min_number:
                if number != min_number[series] or series in seen_series:
                    continue
                seen_series.add(series)
            result.append((index, score))
            if len(result) >= k:
                break
        return result

    def build_rows(
        self,
        seed_ids: list[int],
        liked: list[int] = (),
        disliked: list[int] = (),
        read: set[int] = frozenset(),
        extra_vectors: list[np.ndarray] = (),
        external_seed_titles: list[str] = (),
    ) -> list[dict]:
        excluded = set(read) | set(seed_ids) | set(disliked)
        ranked, cf, content = self.score_pool(
            seed_ids, extra_vectors=extra_vectors, liked=liked, disliked=disliked
        )
        seed_series, seed_titles = self._seed_context(seed_ids, external_seed_titles)
        ranked = [
            (index, score)
            for index, score in ranked
            if self._candidate_allowed(index, excluded, seed_series, seed_titles)
        ]
        minimum_series_number: dict[str, float] = {}
        for index, _ in ranked:
            book = self.catalog[index]
            series = self.series_keys[index]
            number = book.get("series_number")
            if series and isinstance(number, (int, float)) and not isinstance(number, bool):
                minimum_series_number[series] = min(
                    number, minimum_series_number.get(series, number)
                )
        ranked = [
            (index, score)
            for index, score in ranked
            if not self.series_keys[index]
            or self.catalog[index].get("series_number")
            == minimum_series_number.get(self.series_keys[index])
        ]

        def books_from(values: list[tuple[int, float]], limit: int) -> list[dict]:
            eligible = [
                (index, score)
                for index, score in values
                if self._candidate_allowed(index, excluded, seed_series, seed_titles)
            ]
            min_number: dict[str, float] = {}
            for index, _ in eligible:
                book = self.catalog[index]
                series = self.series_keys[index]
                number = book.get("series_number")
                if series and isinstance(number, (int, float)) and not isinstance(number, bool):
                    min_number[series] = min(number, min_number.get(series, number))
            kept = []
            counts: dict[str, int] = {}
            seen_series = set()
            for index, _ in eligible:
                book = self.catalog[index]
                series = self.series_keys[index]
                if series and book.get("series_number") is not None and series in min_number:
                    if book["series_number"] != min_number.get(series) or series in seen_series:
                        continue
                    seen_series.add(series)
                author = book.get("author", "")
                if counts.get(author, 0) >= 2:
                    continue
                counts[author] = counts.get(author, 0) + 1
                kept.append(index)
                if len(kept) >= limit:
                    break
            return [self.catalog[index] for index in kept]

        rows = [{"key": "top", "title": "Top picks for you", "books": books_from(ranked, 20)}]
        neighbor_indices, neighbor_weights = self._matrix("ease")
        seed_strength = sorted(
            seed_ids,
            key=lambda seed: -float(neighbor_weights[seed][neighbor_weights[seed] != 0].sum()),
        )
        for seed in seed_strength[:3]:
            candidates = [
                (int(index), float(weight))
                for index, weight in zip(
                    neighbor_indices[seed], neighbor_weights[seed], strict=True
                )
                if weight != 0
                and self._candidate_allowed(int(index), excluded, seed_series, seed_titles)
            ]
            candidates.sort(key=lambda pair: pair[1], reverse=True)
            rows.append(
                {
                    "key": f"because:gb:{self.catalog[seed]['id']}",
                    "title": f"Because you loved {self.catalog[seed]['title']}",
                    "books": books_from(candidates, 15),
                }
            )
        median = float(np.median([book.get("ratings_count", 0) for book in self.catalog]))
        rows.append(
            {
                "key": "gems",
                "title": "Hidden gems",
                "books": books_from(
                    [
                        pair
                        for pair in ranked
                        if self.catalog[pair[0]].get("ratings_count", 0) < median
                    ],
                    15,
                ),
            }
        )
        top300 = ranked[:300]
        if top300:
            taste, _ = self._taste_vector(
                seed_ids, list(extra_vectors), list(liked), list(disliked)
            )
            taste_similarity = self.embeddings @ taste if taste is not None else content
            sims = np.asarray([taste_similarity[index] for index, _ in top300])
            bottom_half = np.argsort(sims, kind="stable")[: len(top300) // 2]
            outside = [
                (top300[position][0], float(cf[top300[position][0]])) for position in bottom_half
            ]
            outside.sort(key=lambda pair: pair[1], reverse=True)
        else:
            outside = []
        rows.append(
            {
                "key": "outside",
                "title": "A little outside your usual",
                "books": books_from(outside, 15),
            }
        )
        seed_authors = {
            self.catalog[index].get("author")
            for index in seed_ids
            if self.catalog[index].get("author")
        }
        authored = [
            (
                index,
                float(book.get("avg_rating", 0)) * np.log(max(1, book.get("ratings_count", 0))),
            )
            for index, book in enumerate(self.catalog)
            if book.get("author") in seed_authors
            and self._candidate_allowed(index, excluded, seed_series, seed_titles)
        ]
        authored.sort(key=lambda pair: pair[1], reverse=True)
        author_books = books_from(authored, 15)
        if len(author_books) >= 4:
            rows.append(
                {
                    "key": "authors",
                    "title": "More from authors you love",
                    "books": author_books,
                }
            )
        return rows


def load(data_dir: str | Path) -> Engine:
    directory = Path(data_dir)
    with (directory / "catalog.json").open(encoding="utf-8") as handle:
        catalog = json.load(handle)
    descriptions_path = directory / "descriptions.jsonl"
    if descriptions_path.exists():
        details = {}
        with descriptions_path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    detail = json.loads(line)
                    details[int(detail["book_id"])] = detail
                except (KeyError, ValueError, json.JSONDecodeError):
                    continue
        for book in catalog:
            detail = details.get(int(book["id"]), {})
            book["description"] = detail.get("description") or ""
            book["openlibrary_work_key"] = detail.get("work_key") or book.get(
                "openlibrary_work_key"
            )
    else:
        for book in catalog:
            book.setdefault("description", "")
    embeddings = np.load(directory / "emb.npy", mmap_mode="r")
    with np.load(directory / "cf_ease.npz") as data:
        ease_indices = data["indices"]
        ease_weights = data["weights"]
    with np.load(directory / "cf_cosine.npz") as data:
        cosine_indices = data["indices"]
        cosine_weights = data["weights"]
    return Engine(
        directory, catalog, embeddings, ease_indices, ease_weights, cosine_indices, cosine_weights
    )


@lru_cache(maxsize=1)
def _default_engine() -> Engine:
    default_dir = Path(__file__).resolve().parents[2] / "data"
    return load(os.getenv("NEXTBOOK_DATA_DIR", str(default_dir)))


def recommend(
    seed_ids: list[int],
    k: int = 50,
    exclude: set[int] = frozenset(),
    params: Params = _DEFAULT_PARAMS,
    extra_vectors: list[np.ndarray] = (),
    liked: list[int] = (),
    disliked: list[int] = (),
) -> list[tuple[int, float]]:
    return _default_engine().recommend(seed_ids, k, exclude, params, extra_vectors, liked, disliked)


def build_rows(
    seed_ids: list[int],
    liked: list[int] = (),
    disliked: list[int] = (),
    read: set[int] = frozenset(),
    extra_vectors: list[np.ndarray] = (),
) -> list[dict]:
    return _default_engine().build_rows(seed_ids, liked, disliked, read, extra_vectors)
