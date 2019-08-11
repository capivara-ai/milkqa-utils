"""Tests for src/preprocessing.py."""
from milkqa_utils import preprocessing


def test_word_tokenize():
    assert preprocessing.tokenize("abc abc abc") == ["abc", "abc", "abc"]
    assert preprocessing.tokenize("abc, abc. abc! cdd... cc; o? o: a") == [
        "abc",
        ",",
        "abc",
        ".",
        "abc",
        "!",
        "cdd",
        "...",
        "cc",
        ";",
        "o",
        "?",
        "o",
        ":",
        "a",
    ]
    assert preprocessing.tokenize("abc.abc abc") == ["abc.abc", "abc"]


def test_has_alphanum():
    assert preprocessing.has_alphanum("abc.com") == True
    assert preprocessing.has_alphanum("abc.com.,!232a") == True
    assert preprocessing.has_alphanum("!,...[]{}?") == False


def test_remove_punctuation():
    assert list(preprocessing.remove_punctuation(["abc", "abc"])) == ["abc", "abc"]
    assert list(
        preprocessing.remove_punctuation(
            [
                "abc",
                ",",
                "abc",
                ".",
                "abc",
                "!",
                "cdd",
                "...",
                "cc",
                ";",
                "o",
                "?",
                "o",
                ":",
                "a",
            ]
        )
    ) == ["abc", "abc", "abc", "cdd", "cc", "o", "o", "a"]
    assert list(preprocessing.remove_punctuation(["abc.abc", "abc"])) == [
        "abc.abc",
        "abc",
    ]


def test_remove_stopwords():
    assert list(
        preprocessing.remove_stopwords(
            ["o", "rato", "roeu", "a", "roupa", "do", "rei", "de", "roma"]
        )
    ) == ["rato", "roeu", "roupa", "rei", "roma"]
    assert list(
        preprocessing.remove_stopwords(
            ["o", "o", "roeu", "a", "de", "do", "da", "de", "da"]
        )
    ) == ["roeu"]
    assert list(preprocessing.remove_stopwords(["o", "o", "da", "de"])) == []

