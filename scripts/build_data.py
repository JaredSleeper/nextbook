import argparse
import array
import csv
import json
import shutil
import urllib.request
from pathlib import Path

import numpy as np
from scipy import linalg, sparse

from nextbook.catalog import clean_isbn, clean_isbn13, clean_tag, is_derivative, parse_title

RAW_FILES = {
    name: f"https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/{name}.csv"
    for name in ("books", "ratings", "tags", "book_tags")
}
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def ensure_raw(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, url in RAW_FILES.items():
        path = raw_dir / f"{name}.csv"
        if not path.exists():
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, path)


def read_tag_map(raw_dir: Path) -> dict[int, str]:
    with (raw_dir / "tags.csv").open(newline="", encoding="utf-8") as handle:
        return {int(row["tag_id"]): row["tag_name"] for row in csv.DictReader(handle)}


def read_book_tags(raw_dir: Path, tags: dict[int, str]) -> dict[int, list[str]]:
    counts: dict[int, dict[str, int]] = {}
    with (raw_dir / "book_tags.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            tag = clean_tag(tags.get(int(row["tag_id"]), ""))
            if tag:
                book_id = int(row["goodreads_book_id"])
                counts.setdefault(book_id, {})[tag] = int(row["count"])
    return {
        book_id: [
            tag for tag, _ in sorted(values.items(), key=lambda item: (-item[1], item[0]))[:8]
        ]
        for book_id, values in counts.items()
    }


def read_descriptions(data_dir: Path) -> dict[int, dict[str, str | None]]:
    path = data_dir / "descriptions.jsonl"
    descriptions = {}
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    entry = json.loads(line)
                    descriptions[int(entry["book_id"])] = {
                        "description": entry.get("description") or "",
                        "work_key": entry.get("work_key"),
                    }
                except (KeyError, ValueError, json.JSONDecodeError):
                    continue
    return descriptions


def make_catalog(raw_dir: Path, descriptions: dict[int, dict[str, str | None]]) -> list[dict]:
    tags = read_book_tags(raw_dir, read_tag_map(raw_dir))
    catalog = []
    with (raw_dir / "books.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            book_id = int(row["book_id"])
            raw_title = row.get("title", "")
            title, series, series_number = parse_title(raw_title)
            authors = [
                author.strip() for author in row.get("authors", "").split(",") if author.strip()
            ]
            isbn = clean_isbn(row.get("isbn"))
            isbn13 = clean_isbn13(row.get("isbn13"))
            cover = row.get("image_url", "").strip()
            if "nophoto" in cover.lower() and (isbn or isbn13):
                cover = f"https://covers.openlibrary.org/b/isbn/{isbn or isbn13}-M.jpg"
            if not cover or "nophoto" in cover.lower():
                cover = None
            try:
                year_value = float(row["original_publication_year"])
                year = int(year_value) if np.isfinite(year_value) else None
            except (KeyError, ValueError):
                year = None
            try:
                avg_rating = float(row.get("average_rating") or 0)
            except ValueError:
                avg_rating = 0.0
            try:
                ratings_count = int(row.get("ratings_count") or 0)
            except ValueError:
                ratings_count = 0
            detail = descriptions.get(book_id, {})
            catalog.append(
                {
                    "id": book_id,
                    "title": title,
                    "series": series,
                    "series_number": series_number,
                    "is_derivative": is_derivative(
                        {
                            "title": title,
                            "raw_title": raw_title,
                            "author": authors[0] if authors else "",
                        }
                    ),
                    "author": authors[0] if authors else "",
                    "authors": authors,
                    "year": year,
                    "avg_rating": avg_rating,
                    "ratings_count": ratings_count,
                    "isbn": isbn or isbn13,
                    "goodreads_book_id": row.get("goodreads_book_id"),
                    "cover": cover,
                    "tags": tags.get(int(row.get("goodreads_book_id") or 0), []),
                    "description": detail.get("description", ""),
                    "openlibrary_work_key": detail.get("work_key"),
                }
            )
    return sorted(catalog, key=lambda book: book["id"])


def _top_neighbors(matrix: np.ndarray, count: int = 200) -> tuple[np.ndarray, np.ndarray]:
    size = matrix.shape[0]
    indices = np.zeros((size, count), dtype=np.int32)
    weights = np.zeros((size, count), dtype=np.float16)
    rows = np.arange(size)
    for start in range(0, size, 256):
        block = matrix[start : start + 256].copy()
        block[np.arange(len(block)), rows[start : start + len(block)]] = -np.inf
        take = min(count, max(0, size - 1))
        if take == 0:
            continue
        candidates = np.argpartition(block, -take, axis=1)[:, -take:]
        vals = np.take_along_axis(block, candidates, axis=1)
        order = np.argsort(vals, axis=1)[:, ::-1]
        candidates = np.take_along_axis(candidates, order, axis=1)
        vals = np.take_along_axis(vals, order, axis=1)
        indices[start : start + len(block), :take] = candidates
        weights[start : start + len(block), :take] = vals
    return indices, weights


def build_cf(
    raw_dir: Path,
    catalog: list[dict],
    out_dir: Path,
    regularization: float,
    holdout_frac: float,
    seed: int,
) -> None:
    index_by_id = {book["id"]: index for index, book in enumerate(catalog)}
    users = array.array("i")
    items = array.array("i")
    all_users = set()
    with (raw_dir / "ratings.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            user_id = int(row["user_id"])
            all_users.add(user_id)
            if int(row["rating"]) >= 4:
                book_index = index_by_id.get(int(row["book_id"]))
                if book_index is not None:
                    users.append(user_id)
                    items.append(book_index)
    sorted_users = np.array(sorted(all_users), dtype=np.int32)
    n_holdout = int(len(sorted_users) * holdout_frac)
    holdout_users = np.random.default_rng(seed).permutation(sorted_users)[:n_holdout]
    if holdout_frac:
        np.save(out_dir / "holdout_users.npy", holdout_users)
        excluded = set(map(int, holdout_users))
        train_users = np.fromiter((user for user in users if user not in excluded), dtype=np.int32)
        train_items = np.fromiter(
            (book for user, book in zip(users, items, strict=True) if user not in excluded),
            dtype=np.int32,
        )
    else:
        train_users = np.frombuffer(users, dtype=np.int32)
        train_items = np.frombuffer(items, dtype=np.int32)
    user_count = max(all_users, default=0) + 1
    interactions = sparse.csr_matrix(
        (np.ones(len(train_users), dtype=np.float32), (train_users, train_items)),
        shape=(user_count, len(catalog)),
        dtype=np.float32,
    )
    co = (interactions.T @ interactions).toarray().astype(np.float32)
    counts = np.asarray(interactions.sum(axis=0)).ravel()
    cosine = co / (np.sqrt(np.outer(counts, counts)) + 10.0)
    cosine_indices, cosine_weights = _top_neighbors(cosine)
    np.savez_compressed(out_dir / "cf_cosine.npz", indices=cosine_indices, weights=cosine_weights)
    del cosine, cosine_indices, cosine_weights
    co.flat[:: len(catalog) + 1] += regularization
    precision = linalg.inv(co, overwrite_a=True, check_finite=False)
    diagonal = np.diag(precision).copy()
    ease = -precision / diagonal[np.newaxis, :]
    np.fill_diagonal(ease, 0)
    ease_indices, ease_weights = _top_neighbors(ease)
    np.savez_compressed(out_dir / "cf_ease.npz", indices=ease_indices, weights=ease_weights)
    del ease, precision, co, ease_indices, ease_weights
    print(
        f"CF model built from {len(train_users):,} positive ratings; held out "
        f"{len(holdout_users):,} users"
    )


def build_embeddings(catalog: list[dict], out_dir: Path) -> None:
    from fastembed import TextEmbedding

    model = TextEmbedding(model_name=MODEL_NAME)
    texts = [
        f"{book['title']} by {book['author']}. Genres: {', '.join(book['tags'])}. "
        f"{book.get('description', '')[:600]}"
        for book in catalog
    ]
    vectors = np.vstack([np.asarray(batch) for batch in model.embed(texts, batch_size=128)])
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    np.save(out_dir / "emb.npy", vectors.astype(np.float16))


def build(args: argparse.Namespace) -> None:
    raw_dir = args.raw_dir
    out_dir = args.out
    ensure_raw(raw_dir)
    if args.download_only:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    descriptions = read_descriptions(raw_dir.parent)
    catalog = make_catalog(raw_dir, descriptions)
    with (out_dir / "catalog.json").open("w", encoding="utf-8") as handle:
        compact_catalog = [
            {key: value for key, value in book.items() if key != "description"} for book in catalog
        ]
        json.dump(compact_catalog, handle, ensure_ascii=False, separators=(",", ":"))
    if out_dir.resolve() != raw_dir.parent.resolve():
        description_path = raw_dir.parent / "descriptions.jsonl"
        if description_path.exists():
            shutil.copy2(description_path, out_dir / "descriptions.jsonl")
    build_cf(raw_dir, catalog, out_dir, args.regularization, args.holdout_frac, args.seed)
    reuse_embeddings = args.reuse_embeddings
    if reuse_embeddings is None and out_dir.resolve() != raw_dir.parent.resolve():
        source_embeddings = raw_dir.parent / "emb.npy"
        if source_embeddings.exists():
            reuse_embeddings = source_embeddings
    if reuse_embeddings and reuse_embeddings.exists():
        if reuse_embeddings.resolve() != (out_dir / "emb.npy").resolve():
            shutil.copy2(reuse_embeddings, out_dir / "emb.npy")
    else:
        build_embeddings(catalog, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--lambda", dest="regularization", type=float, default=500.0)
    parser.add_argument("--holdout-frac", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-embeddings", type=Path)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.holdout_frac < 1:
        parser.error("--holdout-frac must be in [0, 1)")
    if args.regularization <= 0:
        parser.error("--lambda must be positive")
    build(args)


if __name__ == "__main__":
    main()
