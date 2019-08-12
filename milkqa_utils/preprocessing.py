"""Script to preprocess documents."""
import nltk
import re
import string
from typing import Iterable
from typing import List


def tokenize(input_string: str) -> List[str]:
    """Tokenizes a string using ``nltk.word_tokenizer`` and returns the list of tokens.

    Args:
        input_string: The input string.

    Returns:
        A list of tokens.
    """
    return nltk.word_tokenize(input_string)


def has_alphanum(token: str) -> bool:
    """Checks whether a token contains alphanumeric characters.

    Args:
        token: The input token.

    Returns:
        True if ``token`` contains at least one alphanumeric character, False otherwise.
    """
    return any(re.findall(r"[a-zA-Z]|[0-9]", token))


def remove_punctuation(token_list: Iterable) -> filter:
    """Removes punctuation-only tokens from ``token_list``.

    Args:
        token_list: A list of tokens, tokenized by ``tokenize()``.

    Returns:
        A filter containing only the non-punctuation token from ``token_list``.
    """
    return filter(has_alphanum, token_list)


def remove_stopwords(token_list: Iterable) -> filter:
    """Removes stopwords from ``token_list`` according to the list given by nltk.

    Args:
        token_list: A list of tokens, tokenized by ``tokenize()``.

    Returns:
        A filter containing only non-stopword tokens from ``token_list``.
    """
    stopwords = nltk.corpus.stopwords.words("portuguese")
    return filter(lambda x: x not in stopwords, token_list)
