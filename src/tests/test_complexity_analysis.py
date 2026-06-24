from src.complexity_analysis import (
    analyze_sequence_complexity,
    lz76_complexity,
    shannon_bits_per_base,
)


def test_complexity_metrics_distinguish_repeated_from_mixed_sequence() -> None:
    repeated = "ACGT" * 50
    mixed = "ACGTGCAATGCCGTTA" * 12 + "ACGT"

    assert shannon_bits_per_base(repeated) <= shannon_bits_per_base(mixed)
    assert lz76_complexity(repeated) < lz76_complexity(mixed)


def test_analyze_sequence_complexity_includes_optional_bpb() -> None:
    result = analyze_sequence_complexity("ACGTACGT", bits_per_base=1.25)

    assert result["length"] == 8
    assert result["bits_per_base"] == 1.25
    assert "shannon_bits_per_base" in result
    assert "lz76_complexity" in result
