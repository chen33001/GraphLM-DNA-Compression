"""Repeat discovery using de Bruijn k-mer seeds."""

from collections import defaultdict

from src.graph import DeBruijnGraph


def find_graph_repeats(
    sequence: str,
    graph: DeBruijnGraph,
    k: int = 15,
    min_len: int = 40,
) -> list[dict[str, int]]:
    """Find greedy, non-overlapping copies from earlier sequence positions."""
    if k <= 0 or min_len < k:
        raise ValueError("min_len must be at least k, and k must be positive")
    if len(sequence) < k or not graph:
        return []

    prior_positions: defaultdict[str, list[int]] = defaultdict(list)
    repeats: list[dict[str, int]] = []
    start = 0
    while start <= len(sequence) - k:
        seed = sequence[start : start + k]
        best_source = -1
        best_length = 0
        for source in prior_positions[seed]:
            length = k
            max_length = min(len(sequence) - start, start - source)
            while length < max_length and sequence[source + length] == sequence[start + length]:
                length += 1
            if length > best_length:
                best_source, best_length = source, length

        if best_length >= min_len:
            repeats.append({"start": start, "source": best_source, "length": best_length})
            for position in range(start, min(start + best_length, len(sequence) - k + 1)):
                prior_positions[sequence[position : position + k]].append(position)
            start += best_length
        else:
            prior_positions[seed].append(start)
            start += 1
    return repeats
