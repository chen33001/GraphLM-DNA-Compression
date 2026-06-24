import math

import pytest

from src.probability_model import MarkovGenomicLM, ProbabilityModel


def test_probability_model_interface_is_abstract() -> None:
    with pytest.raises(TypeError):
        ProbabilityModel()


def test_markov_model_assigns_fewer_bits_to_learned_pattern() -> None:
    model = MarkovGenomicLM(order=2).fit("ACACACACACAC")

    assert model.score_sequence("ACACAC") < model.score_sequence("AAAAAA")


def test_markov_score_is_finite_before_fit_and_for_n() -> None:
    score = MarkovGenomicLM(order=3).score_sequence("ACGTN")

    assert math.isfinite(score)
    assert score > 0


@pytest.mark.parametrize("order", [-1, 0])
def test_markov_model_rejects_invalid_order(order: int) -> None:
    with pytest.raises(ValueError):
        MarkovGenomicLM(order=order)
