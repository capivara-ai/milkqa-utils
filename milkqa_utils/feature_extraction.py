"""Script to extract features from documents."""
import math
from typing import List


def idf_single_term(term: str, documents: List[List[str]]) -> float:
    """Calculates the inverse document frequency (idf) for a single term given a list of documents.

    Args:
        term: The term to be scored.
        documents: A list of documents - each represented by a list of tokens.
    
    Returns:
        The idf value for ``term`` given ``documents``.
    """
    n = len(documents)
    n_term_in_doc = sum(1 for doc in documents if term in doc)
    return math.log(n / (1 + n_term_in_doc))
