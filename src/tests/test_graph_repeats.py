import pytest

from src.graph import build_debruijn_graph, graph_path_frequencies, graph_to_mermaid
from src.repeats import find_graph_repeats


def test_debruijn_graph_counts_repeated_edges() -> None:
    graph = build_debruijn_graph("ATATAT", k=2)

    assert graph["AT"]["TA"] == 2
    assert graph["TA"]["AT"] == 2
    assert graph_path_frequencies(graph)[("AT", "TA")] == 2


@pytest.mark.parametrize("sequence,k", [("ACGT", 0), ("ACGT", 4), ("", 2)])
def test_debruijn_graph_rejects_invalid_k(sequence: str, k: int) -> None:
    with pytest.raises(ValueError):
        build_debruijn_graph(sequence, k)


def test_repeat_detection_returns_prior_non_overlapping_source() -> None:
    sequence = "ACGTACGTGGACGTACGT"
    graph = build_debruijn_graph(sequence, k=3)

    repeats = find_graph_repeats(sequence, graph, k=3, min_len=8)

    assert repeats == [{"start": 10, "source": 0, "length": 8}]


def test_repeat_detection_ignores_short_repeats() -> None:
    sequence = "ACGTGGACGT"
    graph = build_debruijn_graph(sequence, k=3)

    assert find_graph_repeats(sequence, graph, k=3, min_len=5) == []


def test_graph_to_mermaid_renders_weighted_edges() -> None:
    graph = build_debruijn_graph("ATATAT", k=2)

    mermaid = graph_to_mermaid(graph)

    assert mermaid.startswith("flowchart LR")
    assert 'AT["AT"]' in mermaid
    assert 'TA["TA"]' in mermaid
    assert "AT -->|2| TA" in mermaid
    assert "TA -->|2| AT" in mermaid


def test_graph_to_mermaid_can_limit_edges_and_filter_counts() -> None:
    graph = {
        "AAA": {"AAT": 5, "AAC": 1},
        "AAT": {"ATG": 3},
    }

    mermaid = graph_to_mermaid(graph, max_edges=2, min_count=2)

    assert "AAA -->|5| AAT" in mermaid
    assert "AAT -->|3| ATG" in mermaid
    assert "AAC" not in mermaid


@pytest.mark.parametrize("max_edges,min_count", [(0, 1), (1, 0)])
def test_graph_to_mermaid_validates_arguments(max_edges: int, min_count: int) -> None:
    graph = build_debruijn_graph("ATATAT", k=2)

    with pytest.raises(ValueError):
        graph_to_mermaid(graph, max_edges=max_edges, min_count=min_count)
