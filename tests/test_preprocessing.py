"""Tests for src/preprocessing.py."""
from milkqa_utils import preprocessing


def test_truecase():
    assert preprocessing.truecase("ABC ABS") == "abc abs"
    assert preprocessing.truecase("ABC ABS", str.lower) == "abc abs"
    assert preprocessing.truecase("ABC abS", str.upper) == "ABC ABS"
    assert preprocessing.truecase("ABC abS", lambda x: x * 2) == "ABC abSABC abS"


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
    assert preprocessing.has_alphanum("ãéçêàü") == True
    assert preprocessing.has_alphanum("1234.1234") == True


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


def test_preprocess_all_steps():
    assert preprocessing.preprocess("abc abc abc") == ["abc", "abc", "abc"]
    assert preprocessing.preprocess("aBc, abc. abC! CDD... cc; o? o: a") == [
        "abc",
        "abc",
        "abc",
        "cdd",
        "cc",
    ]
    assert preprocessing.preprocess(
        "Entretanto, seria interessante saber quais os testes que foram feitos para saber se o leite estava realmente ácido. "
        + "O ideal é pedir para fazer teste da acidez titulável, em que o leite está normal se o resultado der de 14 até 18 "
        + "graus Dornic."
    ) == [
        "entretanto",
        "interessante",
        "saber",
        "quais",
        "testes",
        "feitos",
        "saber",
        "leite",
        "realmente",
        "ácido",
        "ideal",
        "pedir",
        "fazer",
        "teste",
        "acidez",
        "titulável",
        "leite",
        "normal",
        "resultado",
        "der",
        "14",
        "18",
        "graus",
        "dornic",
    ]


def test_preprocess_custom_steps():
    assert preprocessing.preprocess(
        "Entretanto, seria interessante saber quais os testes que foram feitos para saber se o leite estava realmente ácido. "
        + "O ideal é pedir para fazer teste da acidez titulável, em que o leite está normal se o resultado der de 14 até 18 "
        + "graus Dornic.",
        [preprocessing.remove_punctuation],
    ) == [
        "entretanto",
        "seria",
        "interessante",
        "saber",
        "quais",
        "os",
        "testes",
        "que",
        "foram",
        "feitos",
        "para",
        "saber",
        "se",
        "o",
        "leite",
        "estava",
        "realmente",
        "ácido",
        "o",
        "ideal",
        "é",
        "pedir",
        "para",
        "fazer",
        "teste",
        "da",
        "acidez",
        "titulável",
        "em",
        "que",
        "o",
        "leite",
        "está",
        "normal",
        "se",
        "o",
        "resultado",
        "der",
        "de",
        "14",
        "até",
        "18",
        "graus",
        "dornic",
    ]
    assert preprocessing.preprocess(
        "Entretanto, seria interessante saber quais os testes que foram feitos para saber se o leite estava realmente ácido. "
        + "O ideal é pedir para fazer teste da acidez titulável, em que o leite está normal se o resultado der de 14 até 18 "
        + "graus Dornic.",
        [preprocessing.remove_stopwords],
    ) == [
        "entretanto",
        ",",
        "interessante",
        "saber",
        "quais",
        "testes",
        "feitos",
        "saber",
        "leite",
        "realmente",
        "ácido",
        ".",
        "ideal",
        "pedir",
        "fazer",
        "teste",
        "acidez",
        "titulável",
        ",",
        "leite",
        "normal",
        "resultado",
        "der",
        "14",
        "18",
        "graus",
        "dornic",
        ".",
    ]
    assert preprocessing.preprocess(
        "Entretanto, seria interessante saber quais os testes que foram feitos para saber se o leite estava realmente ácido. "
        + "O ideal é pedir para fazer teste da acidez titulável, em que o leite está normal se o resultado der de 14 até 18 "
        + "graus Dornic.",
        [],
    ) == [
        "entretanto",
        ",",
        "seria",
        "interessante",
        "saber",
        "quais",
        "os",
        "testes",
        "que",
        "foram",
        "feitos",
        "para",
        "saber",
        "se",
        "o",
        "leite",
        "estava",
        "realmente",
        "ácido",
        ".",
        "o",
        "ideal",
        "é",
        "pedir",
        "para",
        "fazer",
        "teste",
        "da",
        "acidez",
        "titulável",
        ",",
        "em",
        "que",
        "o",
        "leite",
        "está",
        "normal",
        "se",
        "o",
        "resultado",
        "der",
        "de",
        "14",
        "até",
        "18",
        "graus",
        "dornic",
        ".",
    ]

