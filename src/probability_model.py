"""Genomic probability-model abstraction and a compact Markov backend."""

from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from math import log2

ALPHABET = "ACGTN"


class ProbabilityModel(ABC):
    @abstractmethod
    def score_sequence(self, sequence: str) -> float:
        """Return the idealized encoded length in bits."""


class MarkovGenomicLM(ProbabilityModel):
    """Order-n character Markov model with Laplace smoothing."""

    def __init__(self, order: int = 3, pseudocount: float = 1.0) -> None:
        if order <= 0:
            raise ValueError("order must be positive")
        if pseudocount <= 0:
            raise ValueError("pseudocount must be positive")
        self.order = order
        self.pseudocount = pseudocount
        self._counts: defaultdict[str, Counter[str]] = defaultdict(Counter)

    def fit(self, sequence: str) -> "MarkovGenomicLM":
        self._validate(sequence)
        self._counts.clear()
        for index, symbol in enumerate(sequence):
            context = sequence[max(0, index - self.order) : index]
            self._counts[context][symbol] += 1
        return self

    def score_sequence(self, sequence: str) -> float:
        self._validate(sequence)
        bits = 0.0
        for index, symbol in enumerate(sequence):
            context = sequence[max(0, index - self.order) : index]
            counts = self._counts[context]
            total = sum(counts.values()) + self.pseudocount * len(ALPHABET)
            probability = (counts[symbol] + self.pseudocount) / total
            bits -= log2(probability)
        return bits

    @staticmethod
    def _validate(sequence: str) -> None:
        invalid = set(sequence) - set(ALPHABET)
        if invalid:
            raise ValueError(f"unsupported sequence symbols: {''.join(sorted(invalid))}")
