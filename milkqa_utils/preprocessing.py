"""Script to preprocess documents."""
import re
import logging
from typing import Iterable
from typing import List
from typing import Callable
import nltk


def truecase(input_string: str, case_func: Callable[[str], str] = str.lower) -> str:
    """Truecases a string.

    Args:
        input_string: The input string.
        case_func: The function that will be used to change the string case (by default ``str.lower``).

    Returns:
        The truecased string.
    """
    return case_func(input_string)


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
    pattern = re.compile(r"\w", re.UNICODE)
    return any(pattern.findall(token))


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


def preprocess(
    input_string: str,
    preprocessing_steps: List[Callable[[Iterable], Iterable]] = [
        remove_punctuation,
        remove_stopwords,
    ],
) -> List[str]:
    """Preprocessess a string by applying a list of preprocessing steps.

    Args:
        input_string: The input string.
        preprocessing_steps: A list of functions that will be applied to ``input_string``.

    Returns:
        The list of preprocessed tokens.
    """
    try:
        preprocessed_tokens = truecase(input_string)
        preprocessed_tokens = tokenize(preprocessed_tokens)

        for step in preprocessing_steps:
            preprocessed_tokens = step(preprocessed_tokens)
        return list(preprocessed_tokens)
    except TypeError as ex:
        logging.traceback(ex)
