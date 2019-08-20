"""Baseline models implementation."""
from typing import List
from milkqa_utils import feature_extraction
from milkqa_utils import preprocessing


def wwm_get_matches(question: List[str], answer: List[str]) -> List[str]:
    """Returns all non-stopword tokens from the question that also occur in the the answer.
    
    Args:
        question: The list of tokens of the question.
        answer: The list of tokens of the answer.
    
    Returns:
        The list of non-stopword tokens from the question that also occur in the answer.
    """
    question_no_stopwords = preprocessing.remove_stopwords(question)
    answer_no_stopwords = preprocessing.remove_stopwords(answer)
    # We create a set for effiency (access is O(1))
    compare_set = set(answer_no_stopwords)
    return [token for token in question_no_stopwords if token in compare_set]


def wwm_score_pair(question: List[str], answer: List[str]) -> float:
    """Returns the Weighted Word Matching (WWM) score between ``question`` and ``answer``.
    WWM is defined as \"the sum of IDF values for each non-stopword in the question that also occurs in the answer\".

    Args:
        question: The list of tokens of the question.
        answer: The list of tokens of the answer.
    
    Returns:
        The WWM score of the pair (``question``, ``answer``).
    """
    matches = wwm_get_matches(question=question, answer=answer)
    documents = [question, answer]
    print(documents)
    return sum(
        feature_extraction.idf_single_term(term=token, documents=documents)
        for token in matches
    )
