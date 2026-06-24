"""Sequence complexity helpers inspired by the notebook analysis workflow."""

import math


def shannon_bits_per_base(sequence: str) -> float:
    if not sequence:
        return 0.0
    counts: dict[str, int] = {}
    for symbol in sequence:
        counts[symbol] = counts.get(symbol, 0) + 1
    length = len(sequence)
    entropy = 0.0
    for count in counts.values():
        probability = count / length
        entropy -= probability * math.log2(probability)
    return entropy


def lz76_complexity(sequence: str) -> int:
    if not sequence:
        return 0
    complexity = 1
    index = 0
    window = ""
    candidate = sequence[0]
    while index + len(candidate) < len(sequence):
        if candidate in window:
            candidate += sequence[index + len(candidate)]
            continue
        complexity += 1
        window += candidate
        index += len(candidate)
        candidate = sequence[index : index + 1]
    return complexity


def analyze_sequence_complexity(
    sequence: str,
    *,
    bits_per_base: float | None = None,
) -> dict[str, float | int | None]:
    gc_count = sum(symbol in {"G", "C"} for symbol in sequence)
    return {
        "length": len(sequence),
        "gc_fraction": (gc_count / len(sequence)) if sequence else 0.0,
        "shannon_bits_per_base": shannon_bits_per_base(sequence),
        "lz76_complexity": lz76_complexity(sequence),
        "bits_per_base": bits_per_base,
    }
