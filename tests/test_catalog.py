import pytest

from nextbook.catalog import (
    TAG_STOPLIST,
    clean_isbn13,
    clean_tag,
    is_derivative,
    normalize_series,
    parse_title,
)


def test_tag_cleaning_stoplist_and_year_tags():
    for tag in TAG_STOPLIST:
        assert clean_tag(tag) is None
    assert clean_tag("currently reading") is None
    assert clean_tag("read-in-2009") is None
    assert clean_tag("fantasy") == "fantasy"
    assert clean_tag("romance-novel") == "romance novel"
    assert clean_tag("猫") is None
    assert clean_tag("sf") is None


def test_series_title_parsing():
    assert parse_title("The Hunger Games (The Hunger Games, #1)") == (
        "The Hunger Games",
        "The Hunger Games",
        1,
    )
    assert parse_title("Dune (Dune Chronicles #1)") == ("Dune", "Dune Chronicles", 1)
    assert parse_title(
        "One Thousand White Women: The Journals of May Dodd (One Thousand White Women#1)"
    ) == (
        "One Thousand White Women: The Journals of May Dodd",
        "One Thousand White Women",
        1,
    )
    assert parse_title("Standalone title") == ("Standalone title", None, None)


@pytest.mark.parametrize(
    ("title", "expected_title", "expected_series", "expected_number"),
    [
        (
            "A Game of Thrones (A Song of Ice and Fire, #1)",
            "A Game of Thrones",
            "A Song of Ice and Fire",
            1,
        ),
        ("Dune (Dune Chronicles #1)", "Dune", "Dune Chronicles", 1),
        (
            "The Slow Regard of Silent Things (The Kingkiller Chronicle #2.5)",
            "The Slow Regard of Silent Things",
            "The Kingkiller Chronicle",
            2.5,
        ),
        (
            "Mistborn Trilogy Boxed Set (Mistborn, #1-3)",
            "Mistborn Trilogy Boxed Set",
            "Mistborn",
            None,
        ),
        (
            "A Storm of Swords: Blood and Gold (A Song of Ice and Fire, #3: Part 2 of 2)",
            "A Storm of Swords: Blood and Gold",
            "A Song of Ice and Fire",
            None,
        ),
        (
            "The Wheel of Time Box (Wheel of Time, #1-1)",
            "The Wheel of Time Box",
            "Wheel of Time",
            None,
        ),
    ],
)
def test_series_suffix_variants(title, expected_title, expected_series, expected_number):
    assert parse_title(title) == (expected_title, expected_series, expected_number)


def test_series_names_are_normalized_for_comparison():
    assert normalize_series("The Wheel of Time") == normalize_series("Wheel of time")


@pytest.mark.parametrize(
    "title",
    [
        "Mistborn Trilogy Boxed Set (Mistborn, #1-3)",
        "The Wheel of Time: Boxed Set #1 (Wheel of Time, #1-3)",
        "A Storm of Swords: Blood and Gold (A Song of Ice and Fire, #3: Part 2 of 2)",
        "The Way of Kings, Part 1 (The Stormlight Archive #1.1)",
        "BookRags Summary:  A Storm of Swords",
        "The Earthsea Trilogy",
    ],
)
def test_derivative_editions_are_flagged(title):
    display_title, _, _ = parse_title(title)
    assert is_derivative({"title": display_title, "raw_title": title})


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("The Ultimate Hitchhiker's Guide to the Galaxy", True),
        ("The Complete Sherlock Holmes", False),
        ("Complete Me", False),
        ("Ultimate Spider-Man, Volume 1: Power and Responsibility", False),
    ],
)
def test_complete_or_ultimate_title_prefixes(title, expected):
    assert is_derivative({"title": title}) is expected


@pytest.mark.parametrize(
    "title",
    [
        "The Hunger Games (The Hunger Games, #1)",
        "Mockingjay (The Hunger Games, #3)",
    ],
)
def test_regular_series_books_are_not_flagged_as_derivatives(title):
    display_title, _, _ = parse_title(title)
    assert not is_derivative({"title": display_title, "raw_title": title})


@pytest.mark.parametrize("author", ["BookRags", "SparkNotes"])
def test_derivative_authors_are_flagged(author):
    assert is_derivative({"title": "A Storm of Swords", "author": author})


def test_isbn13_float_formatted_value_is_unusable():
    assert clean_isbn13("9.78044e+12") is None
    assert clean_isbn13("9780441172719.0") is None
    assert clean_isbn13("9780441172719") == "9780441172719"
    assert clean_isbn13(9780441172719) == "9780441172719"
