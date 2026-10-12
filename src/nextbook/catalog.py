import re

TAG_STOPLIST = {
    "to-read",
    "currently-reading",
    "favorites",
    "favourites",
    "owned",
    "books-i-own",
    "my-books",
    "owned-books",
    "library",
    "kindle",
    "ebook",
    "ebooks",
    "audiobook",
    "audiobooks",
    "default",
    "wish-list",
    "i-own",
    "to-buy",
    "re-read",
}
_NORMALIZED_STOPLIST = {tag.replace("-", " ") for tag in TAG_STOPLIST}
_NON_GENRE_PREFIX = re.compile(
    r"^(to-|my-|own|read|book|fav|have|wish|tbr|default|e-?book|kindle|audio)"
)
_YEAR_TAG = re.compile(r"\d{4}")
_SERIES_SUFFIX = re.compile(r"\s+\(([^()]+?)(?:,\s*|\s*)#(\d+)\)$")


def clean_tag(tag: str) -> str | None:
    value = " ".join(tag.strip().lower().replace("-", " ").split())
    if len(value) < 3 or not value.isascii() or _YEAR_TAG.search(value):
        return None
    if value in _NORMALIZED_STOPLIST or _NON_GENRE_PREFIX.match(value.replace(" ", "-")):
        return None
    return value


def parse_title(title: str) -> tuple[str, str | None, int | None]:
    match = _SERIES_SUFFIX.search(title)
    if not match:
        return title.strip(), None, None
    return title[: match.start()].strip(), match.group(1).strip(), int(match.group(2))


def clean_isbn13(value: str | int | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if re.fullmatch(r"\d{13}", normalized):
        return normalized
    return None


def clean_isbn(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized or normalized.lower() == "nan":
        return None
    return normalized
