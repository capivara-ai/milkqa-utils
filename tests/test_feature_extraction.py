"""Tests for feature_extraction.py"""
import math
import pytest
from milkqa_utils import feature_extraction


def test_idf_single_term():
    assert feature_extraction.idf_single_term(
        "porta", [["abc", "bca", "porta"]]
    ) == math.log(1 / 2)
    assert feature_extraction.idf_single_term(
        "porta", [["abc", "bca", "porta", "porta"]]
    ) == math.log(1 / 2)
    assert feature_extraction.idf_single_term(
        "porta",
        [
            ["abc", "bca", "porta", "porta"],
            ["abc", "dd", "abc"],
            ["a", "Porta"],
            ["porta", "porteira", "portão"],
            ["abc", "bca", "dd"],
        ],
    ) == math.log(5 / 3)
    assert feature_extraction.idf_single_term(
        "abc",
        [
            ["abc", "bca", "porta", "porta"],
            ["abc", "dd", "abc"],
            ["a", "Porta"],
            ["porta", "porteira", "portão"],
            ["abc", "bca", "dd"],
        ],
    ) == math.log(5 / 4)
