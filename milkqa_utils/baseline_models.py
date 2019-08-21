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


def wwm_score(question: List[str], answers: List[List[str]]) -> List[float]:
    """Returns a list of Weighted Word Matching (WWM) scores between ``question`` and each answer in ``answers``.
    WWM is defined as \"the sum of IDF values for each non-stopword in the question that also occurs in the answer\".

    Args:
        question: The list of tokens of the question.
        answers: The list of answers - each answer is a list of tokens.

    Returns:
        The list of WWM scores for each pair (``question``, ``answer``).
    """
    # TODO: Cache idf for terms already seen
    matches = [wwm_get_matches(question=question, answer=answer) for answer in answers]
    documents = [question] + answers
    scores = []
    for match in matches:
        scores.append(
            sum(
                feature_extraction.idf_single_term(term=token, documents=documents)
                for token in match
            )
        )
    return scores

