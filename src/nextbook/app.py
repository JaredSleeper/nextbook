import asyncio
import json
import logging
import os
import re
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from nextbook import llm
from nextbook.recommend import Engine, load

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = Jinja2Templates(directory=str(ROOT / "templates"))
OL_BASE = "https://openlibrary.org"
_SEARCH_CACHE: OrderedDict[tuple[str, int], list[dict]] = OrderedDict()
_EXTERNAL_SUBJECTS: OrderedDict[str, list[str]] = OrderedDict()
_SEED_VECTOR_CACHE: OrderedDict[str, np.ndarray] = OrderedDict()
_PICKS_CACHE: OrderedDict[tuple, dict] = OrderedDict()
_CACHE_LIMIT = 256
_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


class Seed(BaseModel):
    id: str | int
    title: str
    author: str = ""
    subjects: list[str] = Field(default_factory=list)


class RecommendationRequest(BaseModel):
    seeds: list[Seed] = Field(default_factory=list)
    liked: list[str | int] = Field(default_factory=list)
    disliked: list[str | int] = Field(default_factory=list)
    read: list[str | int] = Field(default_factory=list)


class PicksRequest(RecommendationRequest):
    prompt: str | None = None


def normalize(value: str) -> str:
    return " ".join(_NORMALIZE_RE.sub(" ", value.lower()).split())


def local_search(engine: Engine, query: str, limit: int) -> list[dict]:
    needle = normalize(query)
    tokens = needle.split()
    matches = []
    for index, book in enumerate(engine.catalog):
        searchable = normalize(f"{book.get('title', '')} {book.get('author', '')}")
        title = normalize(book.get("title", ""))
        if (
            not needle
            or searchable.startswith(needle)
            or all(token in searchable.split() for token in tokens)
        ):
            matches.append((index, searchable.startswith(needle), title.startswith(needle)))
    matches.sort(
        key=lambda match: (
            not (match[1] or match[2]),
            -int(engine.catalog[match[0]].get("ratings_count", 0)),
            engine.catalog[match[0]].get("title", "").lower(),
        )
    )
    return [
        {
            "id": f"gb:{engine.catalog[index]['id']}",
            "title": engine.catalog[index]["title"],
            "author": engine.catalog[index].get("author", ""),
            "year": engine.catalog[index].get("year"),
            "cover": engine.catalog[index].get("cover"),
            "in_catalog": True,
        }
        for index, _, _ in matches[:limit]
    ]


async def search_open_library(query: str, limit: int) -> list[dict]:
    cache_key = (normalize(query), limit)
    if cache_key in _SEARCH_CACHE:
        _SEARCH_CACHE.move_to_end(cache_key)
        return list(_SEARCH_CACHE[cache_key])
    try:
        async with httpx.AsyncClient(timeout=3.0, follow_redirects=True) as client:
            response = await client.get(
                f"{OL_BASE}/search.json",
                params={
                    "q": query,
                    "limit": limit,
                    "fields": "key,title,author_name,first_publish_year,cover_i,subject",
                },
            )
            response.raise_for_status()
            docs = response.json().get("docs", [])
    except (httpx.HTTPError, ValueError):
        return []
    results = []
    for doc in docs:
        key = doc.get("key")
        title = doc.get("title")
        if not key or not title:
            continue
        cover_id = doc.get("cover_i")
        authors = doc.get("author_name") or []
        results.append(
            {
                "id": f"ol:{key}",
                "title": title,
                "author": authors[0] if authors else "",
                "year": doc.get("first_publish_year"),
                "cover": f"https://covers.openlibrary.org/b/id/{cover_id}-M.jpg"
                if cover_id
                else None,
                "in_catalog": False,
                "subjects": (doc.get("subject") or [])[:8],
            }
        )
        _EXTERNAL_SUBJECTS[f"ol:{key}"] = (doc.get("subject") or [])[:8]
        _EXTERNAL_SUBJECTS.move_to_end(f"ol:{key}")
        while len(_EXTERNAL_SUBJECTS) > _CACHE_LIMIT:
            _EXTERNAL_SUBJECTS.popitem(last=False)
    _SEARCH_CACHE[cache_key] = results
    _SEARCH_CACHE.move_to_end(cache_key)
    while len(_SEARCH_CACHE) > _CACHE_LIMIT:
        _SEARCH_CACHE.popitem(last=False)
    return list(results)


@lru_cache(maxsize=1)
def _embedder():
    from fastembed import TextEmbedding

    return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")


def embed_texts(texts: list[str]) -> np.ndarray:
    vectors = np.vstack([np.asarray(batch) for batch in _embedder().embed(texts)])
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def seed_vector(seed: Seed) -> np.ndarray:
    key = str(seed.id)
    if key not in _SEED_VECTOR_CACHE:
        subjects = seed.subjects[:8] or _EXTERNAL_SUBJECTS.get(key, [])[:8]
        text = f"{seed.title} by {seed.author}. Genres: {', '.join(subjects)}"
        _SEED_VECTOR_CACHE[key] = embed_texts([text])[0]
        while len(_SEED_VECTOR_CACHE) > _CACHE_LIMIT:
            _SEED_VECTOR_CACHE.popitem(last=False)
    return _SEED_VECTOR_CACHE[key]


def resolve_catalog_ids(engine: Engine, identifiers: list[str | int]) -> list[int]:
    resolved = []
    for identifier in identifiers:
        if isinstance(identifier, int):
            book_id = identifier
        elif str(identifier).startswith("gb:"):
            try:
                book_id = int(str(identifier)[3:])
            except ValueError:
                continue
        elif str(identifier).startswith("ol:"):
            index = engine.work_to_index.get(str(identifier)[3:])
            if index is not None and index not in resolved:
                resolved.append(index)
            continue
        else:
            continue
        index = engine.id_to_index.get(book_id)
        if index is not None and index not in resolved:
            resolved.append(index)
    return resolved


def resolve_seeds(engine: Engine, seeds: list[Seed]) -> tuple[list[int], list[np.ndarray]]:
    indices = []
    vectors = []
    for seed in seeds:
        identifier = str(seed.id)
        if identifier.startswith("gb:"):
            try:
                index = engine.id_to_index.get(int(identifier[3:]))
            except ValueError:
                index = None
            if index is not None and index not in indices:
                indices.append(index)
                continue
        elif identifier.startswith("ol:"):
            work_key = identifier[3:]
            index = engine.work_to_index.get(work_key)
            if index is not None and index not in indices:
                indices.append(index)
                continue
        vectors.append(seed_vector(seed))
    return indices, vectors


def book_response(
    engine: Engine, index: int, reason: str | None = None, include_reason: bool = False
) -> dict:
    book = engine.catalog[index]
    result = {
        "id": f"gb:{book['id']}",
        "title": book["title"],
        "author": book.get("author", ""),
        "year": book.get("year"),
        "cover": book.get("cover"),
        "rating": book.get("avg_rating"),
        "ratings_count": book.get("ratings_count", 0),
        "tags": book.get("tags", [])[:5],
        "description": book.get("description") or "",
        "series": book.get("series"),
        "goodreads_url": (
            f"https://www.goodreads.com/book/show/{book['goodreads_book_id']}"
            if book.get("goodreads_book_id")
            else None
        ),
        "openlibrary_url": (
            f"{OL_BASE}{book['openlibrary_work_key']}" if book.get("openlibrary_work_key") else None
        ),
    }
    if include_reason:
        result["reason"] = reason
    return result


async def log_event(request: Request, kind: str, payload: dict) -> None:
    pool = getattr(request.app.state, "db_pool", None)
    if pool is None:
        return
    try:
        await pool.execute(
            "INSERT INTO event (kind, payload) VALUES ($1, $2::jsonb)",
            kind,
            json.dumps(payload),
        )
    except Exception:
        logger.exception("Failed to log %s event", kind)


async def favorites(engine: Engine, indices: list[int]) -> list[dict]:
    return [
        {"title": engine.catalog[index]["title"], "author": engine.catalog[index].get("author", "")}
        for index in indices
    ]


@asynccontextmanager
async def lifespan(app: FastAPI):
    data_dir = Path(os.getenv("NEXTBOOK_DATA_DIR", str(ROOT / "data")))
    app.state.engine = load(data_dir)
    app.state.db_pool = None
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        try:
            import asyncpg

            app.state.db_pool = await asyncpg.create_pool(database_url)
            async with app.state.db_pool.acquire() as connection:
                await connection.execute(
                    "CREATE TABLE IF NOT EXISTS event ("
                    "id bigserial PRIMARY KEY, ts timestamptz DEFAULT now(), kind text, "
                    "payload jsonb)"
                )
        except Exception:
            logger.exception("Event logging database unavailable")
            app.state.db_pool = None
    try:
        yield
    finally:
        if app.state.db_pool:
            await app.state.db_pool.close()


app = FastAPI(title="nextbook", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return TEMPLATES.TemplateResponse(request=request, name="index.html", context={})


@app.get("/health")
async def health(request: Request):
    return {"ok": True, "books": len(request.app.state.engine.catalog)}


@app.get("/api/search")
async def search(
    request: Request,
    q: str = Query(default=""),
    limit: int = Query(default=8, ge=1, le=25),
):
    engine: Engine = request.app.state.engine
    local = local_search(engine, q, limit)
    if len(local) >= limit or not q.strip():
        return {"results": local}
    local_keys = {
        (normalize(book.get("title", "")), normalize(book.get("author", "")))
        for book in engine.catalog
    }
    external = await search_open_library(q, limit * 2)
    results = list(local)
    seen = set(local_keys)
    for item in external:
        dedupe_key = (normalize(item["title"]), normalize(item["author"]))
        if dedupe_key in seen:
            continue
        results.append({key: value for key, value in item.items() if key != "subjects"})
        seen.add(dedupe_key)
        if len(results) >= limit:
            break
    return {"results": results}


@app.post("/api/recommend")
async def recommendations(body: RecommendationRequest, request: Request):
    started = time.perf_counter()
    engine: Engine = request.app.state.engine
    seeds, external_vectors = resolve_seeds(engine, body.seeds)
    liked = resolve_catalog_ids(engine, body.liked)
    disliked = resolve_catalog_ids(engine, body.disliked)
    read = set(resolve_catalog_ids(engine, body.read))
    rows = engine.build_rows(seeds, liked, disliked, read, external_vectors)
    output = [
        {
            "key": row["key"],
            "title": row["title"],
            "books": [
                book_response(engine, engine.id_to_index[book["id"]]) for book in row["books"]
            ],
        }
        for row in rows
    ]
    elapsed_ms = (time.perf_counter() - started) * 1000
    await log_event(
        request,
        "recommend",
        {
            "seeds": [{"title": seed.title, "author": seed.author} for seed in body.seeds],
            "prompt": None,
            "returned_ids": [book["id"] for row in output for book in row["books"]],
            "latency_ms": elapsed_ms,
            "fallback": False,
            "tokens": None,
        },
    )
    return {"rows": output}


@app.post("/api/picks")
async def picks(body: PicksRequest, request: Request):
    started = time.perf_counter()
    engine: Engine = request.app.state.engine
    seeds, external_vectors = resolve_seeds(engine, body.seeds)
    liked = resolve_catalog_ids(engine, body.liked)
    disliked = resolve_catalog_ids(engine, body.disliked)
    read = set(resolve_catalog_ids(engine, body.read))
    cache_key = (
        tuple(sorted((str(seed.id), seed.title, seed.author) for seed in body.seeds)),
        tuple(sorted(map(str, body.liked))),
        tuple(sorted(map(str, body.disliked))),
        tuple(sorted(map(str, body.read))),
        body.prompt,
    )
    if cache_key in _PICKS_CACHE:
        _PICKS_CACHE.move_to_end(cache_key)
        cached = _PICKS_CACHE[cache_key]
        await log_event(
            request,
            "picks",
            {
                "seeds": [seed.model_dump() for seed in body.seeds],
                "prompt": body.prompt,
                "returned_ids": [book["id"] for book in cached["books"]],
                "fallback": cached["fallback"],
                "latency_ms": (time.perf_counter() - started) * 1000,
                "tokens": cached["tokens"],
            },
        )
        return cached

    ranked = engine.recommend(
        seeds,
        k=len(engine.catalog),
        exclude=read,
        extra_vectors=external_vectors,
        liked=liked,
        disliked=disliked,
    )
    top_candidates = ranked[:100]
    candidate_indices = [index for index, _ in top_candidates]
    if body.prompt:
        prompt_vector = embed_texts([body.prompt])[0]
        prompt_scores = engine.embeddings @ prompt_vector
        hybrid_indices = np.fromiter((index for index, _ in ranked), dtype=np.int32)
        if len(hybrid_indices):
            hybrid_values = np.asarray([score for _, score in ranked])
            mean = float(hybrid_values.mean())
            std = float(hybrid_values.std())
            hybrid_z = np.zeros_like(hybrid_values) if std < 1e-12 else (hybrid_values - mean) / std
            prompt_rank = [
                (int(index), float(0.5 * prompt_scores[index] + 0.5 * score))
                for index, score in zip(hybrid_indices, hybrid_z, strict=True)
            ]
            prompt_rank.sort(key=lambda pair: pair[1], reverse=True)
            for index, _ in prompt_rank[:60]:
                if index not in candidate_indices:
                    candidate_indices.append(index)
                if len(candidate_indices) == 150:
                    break
    candidate_books = [engine.catalog[index] for index in candidate_indices]
    candidate_favorites = [{"title": seed.title, "author": seed.author} for seed in body.seeds]
    liked_favorites = await favorites(engine, liked)
    disliked_favorites = await favorites(engine, disliked)
    tokens = {"input": 0, "output": 0}
    fallback = False
    response_picks = []
    try:
        llm_picks, input_tokens, output_tokens = await asyncio.to_thread(
            llm.choose_picks,
            candidate_favorites,
            liked_favorites,
            disliked_favorites,
            body.prompt,
            candidate_books,
        )
        tokens = {"input": input_tokens, "output": output_tokens}
        valid_ids = {book["id"] for book in candidate_books}
        seen_ids = set()
        for pick in llm_picks:
            book_id = pick.get("id")
            if (
                not isinstance(book_id, int)
                or isinstance(book_id, bool)
                or book_id not in valid_ids
                or book_id in seen_ids
            ):
                continue
            response_picks.append(pick)
            seen_ids.add(book_id)
            if len(response_picks) == 12:
                break
        if len(response_picks) < 6:
            fallback = True
    except Exception:
        logger.exception("Claude picks failed; using hybrid fallback")
        fallback = True
    if fallback:
        response_picks = [
            {"id": engine.catalog[index]["id"], "reason": None} for index, _ in ranked[:12]
        ]
    books_by_id = {book["id"]: index for index, book in enumerate(engine.catalog)}
    output = {
        "title": f"For: {body.prompt}" if body.prompt else "Picked for you",
        "books": [
            book_response(
                engine,
                books_by_id[pick["id"]],
                pick.get("reason"),
                include_reason=True,
            )
            for pick in response_picks
            if pick["id"] in books_by_id
        ],
        "fallback": fallback,
    }
    output["tokens"] = tokens
    elapsed_ms = (time.perf_counter() - started) * 1000
    output["latency_ms"] = elapsed_ms
    _PICKS_CACHE[cache_key] = output
    _PICKS_CACHE.move_to_end(cache_key)
    while len(_PICKS_CACHE) > _CACHE_LIMIT:
        _PICKS_CACHE.popitem(last=False)
    logger.info(
        "Picks completed in %.1f ms; fallback=%s; tokens=%s",
        elapsed_ms,
        fallback,
        tokens,
    )
    await log_event(
        request,
        "picks",
        {
            "seeds": candidate_favorites,
            "prompt": body.prompt,
            "returned_ids": [book["id"] for book in output["books"]],
            "fallback": fallback,
            "latency_ms": elapsed_ms,
            "tokens": tokens,
        },
    )
    return output


@app.get("/api/book/{book_id}")
async def get_book(book_id: str, request: Request):
    engine: Engine = request.app.state.engine
    if not book_id.startswith("gb:"):
        raise HTTPException(status_code=404, detail="Book not found")
    try:
        index = engine.id_to_index[int(book_id[3:])]
    except (ValueError, KeyError):
        raise HTTPException(status_code=404, detail="Book not found") from None
    return book_response(engine, index)
