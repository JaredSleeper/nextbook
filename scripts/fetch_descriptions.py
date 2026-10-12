import argparse
import asyncio
import csv
import json
from pathlib import Path
from typing import Any

import httpx

USER_AGENT = "nextbook/0.1 (jaredsleeper@gmail.com)"
BASE_URL = "https://openlibrary.org"


def catalog_books(raw_dir: Path) -> list[dict[str, str]]:
    with (raw_dir / "books.csv").open(newline="", encoding="utf-8") as handle:
        return [
            {
                "book_id": row["book_id"],
                "title": row.get("title", ""),
                "author": row.get("authors", "").split(",")[0].strip(),
                "isbn": row.get("isbn", "").strip(),
            }
            for row in csv.DictReader(handle)
        ]


async def fetch_json(client: httpx.AsyncClient, url: str, **params: Any) -> Any | None:
    for attempt in range(5):
        try:
            response = await client.get(url, params=params)
            if response.status_code == 404:
                return None
            if response.status_code == 429 or response.status_code >= 500:
                await asyncio.sleep(min(2**attempt, 12))
                continue
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            if attempt == 4:
                return None
            await asyncio.sleep(min(2**attempt, 12))
    return None


def description_value(value: Any) -> str | None:
    if isinstance(value, dict):
        value = value.get("value")
    return value.strip() if isinstance(value, str) and value.strip() else None


async def fetch_one(
    client: httpx.AsyncClient, book: dict[str, str], semaphore: asyncio.Semaphore
) -> dict[str, Any] | None:
    async with semaphore:
        work_key = None
        description = None
        if book["isbn"]:
            edition = await fetch_json(client, f"{BASE_URL}/isbn/{book['isbn']}.json")
            works = edition.get("works", []) if isinstance(edition, dict) else []
            if works and isinstance(works[0], dict):
                work_key = works[0].get("key")
                if work_key:
                    work = await fetch_json(client, f"{BASE_URL}{work_key}.json")
                    description = (
                        description_value(work.get("description"))
                        if isinstance(work, dict)
                        else None
                    )
        if not description:
            search = await fetch_json(
                client,
                f"{BASE_URL}/search.json",
                title=book["title"],
                author=book["author"],
                limit=1,
                fields="key,first_sentence",
            )
            docs = search.get("docs", []) if isinstance(search, dict) else []
            if docs:
                doc = docs[0]
                work_key = work_key or doc.get("key")
                first_sentence = doc.get("first_sentence")
                if isinstance(first_sentence, list):
                    first_sentence = first_sentence[0] if first_sentence else None
                description = description_value(first_sentence)
        if not description and not work_key:
            return None
        return {
            "book_id": int(book["book_id"]),
            "description": description or "",
            "work_key": work_key,
        }


async def run(raw_dir: Path, output: Path) -> None:
    books = catalog_books(raw_dir)
    cached: dict[int, dict[str, Any]] = {}
    if output.exists():
        with output.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    entry = json.loads(line)
                    cached[int(entry["book_id"])] = entry
                except (KeyError, ValueError, json.JSONDecodeError):
                    continue
    remaining = [book for book in books if int(book["book_id"]) not in cached]
    semaphore = asyncio.Semaphore(8)
    timeout = httpx.Timeout(15.0)
    headers = {"User-Agent": USER_AGENT}
    output.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(
        timeout=timeout,
        headers=headers,
        follow_redirects=True,
    ) as client:
        with output.open("a", encoding="utf-8") as handle:
            for offset in range(0, len(remaining), 32):
                batch = remaining[offset : offset + 32]
                results = await asyncio.gather(
                    *(fetch_one(client, book, semaphore) for book in batch)
                )
                for result in results:
                    if result:
                        cached[result["book_id"]] = result
                        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"{min(offset + len(batch), len(remaining))}/{len(remaining)} fetched; "
                    f"{len(cached)} cached"
                )
    described = sum(bool(entry.get("description")) for entry in cached.values())
    print(f"Description coverage: {described}/{len(books)} ({described / max(len(books), 1):.1%})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/descriptions.jsonl"))
    args = parser.parse_args()
    asyncio.run(run(args.raw_dir, args.out))


if __name__ == "__main__":
    main()
