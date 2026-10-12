from nextbook.catalog import TAG_STOPLIST, clean_isbn13, clean_tag, parse_title


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


def test_isbn13_float_formatted_value_is_unusable():
    assert clean_isbn13("9.78044e+12") is None
    assert clean_isbn13("9780441172719.0") is None
    assert clean_isbn13("9780441172719") == "9780441172719"
    assert clean_isbn13(9780441172719) == "9780441172719"
