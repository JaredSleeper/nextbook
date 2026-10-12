import logging
import os

import anthropic

logger = logging.getLogger(__name__)
MODEL = "claude-sonnet-4-6"
MAX_REASON_WORDS = 25

TOOL = {
    "name": "select_books",
    "description": "Select and explain recommendations from the supplied candidate list.",
    "input_schema": {
        "type": "object",
        "properties": {
            "picks": {
                "type": "array",
                "maxItems": 12,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "reason": {"type": "string"},
                    },
                    "required": ["id", "reason"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["picks"],
        "additionalProperties": False,
    },
}


def choose_picks(
    seeds: list[dict],
    liked: list[dict],
    disliked: list[dict],
    prompt: str | None,
    candidates: list[dict],
) -> tuple[list[dict], int, int]:
    candidate_ids = {candidate["id"] for candidate in candidates}
    lines = [
        f"{book['id']} | {book['title']} | {book['author']} | {book.get('year') or ''} | "
        f"{', '.join(book.get('tags', []))} | {book.get('description', '')[:200]}"
        for book in candidates
    ]
    user_message = (
        f"Favorites: {', '.join(f'{book["title"]} by {book["author"]}' for book in seeds)}\n"
        f"Liked: {', '.join(book['title'] for book in liked) or 'none'}\n"
        f"Disliked: {', '.join(book['title'] for book in disliked) or 'none'}\n"
        f"User prompt: {prompt or 'none'}\n\n"
        "Candidates (id | title | author | year | tags | description):\n" + "\n".join(lines)
    )
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        timeout=60.0,
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=1600,
        system=(
            "Choose exactly 12 books when enough candidates exist, and otherwise choose every "
            "available candidate. You may only choose integer ids from the supplied candidate "
            "list. Never invent or alter a title. Honor the user's prompt over general taste when "
            "they conflict. Give each reason in plain language, no more than 25 words, concrete, "
            "and naming which favorite book it connects to when applicable. Do not use em dashes."
        ),
        messages=[{"role": "user", "content": user_message}],
        tools=[TOOL],
        tool_choice={"type": "tool", "name": "select_books"},
    )
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    logger.info("Claude picks used %d input and %d output tokens", input_tokens, output_tokens)
    picks = []
    seen = set()
    for block in response.content:
        if getattr(block, "type", None) != "tool_use":
            continue
        for pick in block.input.get("picks", []):
            book_id = pick.get("id")
            if (
                not isinstance(book_id, int)
                or isinstance(book_id, bool)
                or book_id not in candidate_ids
                or book_id in seen
            ):
                continue
            reason = str(pick.get("reason", "")).replace("—", "-").replace("–", "-")
            reason = " ".join(reason.split()[:MAX_REASON_WORDS])
            picks.append({"id": book_id, "reason": reason})
            seen.add(book_id)
    return picks, input_tokens, output_tokens
