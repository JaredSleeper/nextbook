import re
from collections.abc import Iterable

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
_SERIES_SUFFIX = re.compile(
    r"\s+\((?P<series>[^()]+?)(?:,\s*|\s*)#(?P<number>\d+(?:\.\d+)?)"
    r"(?P<variant>-\d+(?:\.\d+)?|:\s*part\s+\d+\s+of\s+\d+)?\)$",
    re.IGNORECASE,
)
_DERIVATIVE_TITLE = re.compile(
    r"^(?:the\s+)?(?:complete|ultimate|essential)\s|"
    r"box(?:ed)?\s*set|omnibus|collection|complete\s+(?:novels|series|works)|"
    r"trilogy\s*$|quartet|summary|sparknotes|cliffsnotes|study\s+guide|bookrags|"
    r"part\s+\d+\s+of\s+\d+|,\s*part\s+\d\b|\b\d+-book\b",
    re.IGNORECASE,
)
_DERIVATIVE_AUTHOR = re.compile(r"\bbookrags\b|\bsparknotes\b", re.IGNORECASE)
_SERIES_PUNCTUATION = re.compile(r"[^a-z0-9]+")
_TITLE_PUNCTUATION = re.compile(r"[^a-z0-9:]+")


def clean_tag(tag: str) -> str | None:
    value = " ".join(tag.strip().lower().replace("-", " ").split())
    if len(value) < 3 or not value.isascii() or _YEAR_TAG.search(value):
        return None
    if value in _NORMALIZED_STOPLIST or _NON_GENRE_PREFIX.match(value.replace(" ", "-")):
        return None
    return value


def parse_title(title: str) -> tuple[str, str | None, float | None]:
    match = _SERIES_SUFFIX.search(title)
    if not match:
        return title.strip(), None, None
    number = None if match.group("variant") else float(match.group("number"))
    return title[: match.start()].strip(), match.group("series").strip(), number


def normalize_series(series: str | None) -> str:
    if not series:
        return ""
    normalized = " ".join(_SERIES_PUNCTUATION.sub(" ", series.lower()).split())
    return normalized.removeprefix("the ")


def normalize_title(title: str) -> str:
    return " ".join(_TITLE_PUNCTUATION.sub(" ", title.lower()).split())


def is_seed_adaptation(title: str, seed_titles: Iterable[str]) -> bool:
    normalized_title = normalize_title(title)
    for seed_title in seed_titles:
        normalized_seed = normalize_title(seed_title)
        if not normalized_title.startswith(normalized_seed):
            continue
        suffix = normalized_title[len(normalized_seed) :]
        if suffix.startswith(":") or suffix.startswith((" graphic novel", " part")):
            return True
    return False


def is_derivative(book: dict) -> bool:
    title = str(book.get("title") or "")
    raw_title = str(book.get("raw_title") or title)
    suffix = _SERIES_SUFFIX.search(raw_title)
    if suffix and (suffix.group("variant") or suffix.group("number").endswith(".1")):
        return True
    if _DERIVATIVE_TITLE.search(title) or _DERIVATIVE_TITLE.search(raw_title):
        return True
    authors = book.get("authors") or []
    author = str(book.get("author") or "")
    if isinstance(authors, list):
        author = " ".join([author, *(str(name) for name in authors)])
    return bool(_DERIVATIVE_AUTHOR.search(author))


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
