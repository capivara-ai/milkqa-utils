"""Tests for baseline_models.py"""
import math
import pytest
from milkqa_utils import baseline_models


def test_wwm_get_matches():
    assert baseline_models.wwm_get_matches(
        ["o", "gato", "o", "gato", "é", "fofo"], ["o", "gato", "sim"]
    ) == ["gato", "gato"]
    assert (
        baseline_models.wwm_get_matches(
            ["o", "gato", "o", "gato", "é", "fofo"], ["o", "cachorro", "sim"]
        )
        == []
    )
    assert (
        baseline_models.wwm_get_matches(
            ["o", "cachorro", "o", "papagaio", "é", "fofo"], ["o", "gato", "sim"]
        )
        == []
    )
    assert baseline_models.wwm_get_matches(
        ["o", "gato", "o", "cachorro", "é", "fofo"],
        ["o", "gato", "sim", "gato", "sim", "cachorro", "também"],
    ) == ["gato", "cachorro"]


def test_wwm_score():
    assert baseline_models.wwm_score(
        ["você", "gosta", "de", "comer", "batata"], [["batata", "é", "bom"]]
    ) == [math.log(2 / 3)]
    assert baseline_models.wwm_score(
        ["o", "gato", "o", "gato", "é", "fofo"],
        [
            ["o", "gato", "sim"],
            ["o", "gato", "o", "gato", "é", "fofo"],
            ["o", "cachorro", "sim"],
        ],
    ) == [2 * math.log(1), 2 * math.log(1) + math.log(4 / 3), 0]

