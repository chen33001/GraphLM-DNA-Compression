from pathlib import Path

import pytest

from src.fasta import FastaRecord, read_fasta, write_fasta


def test_fasta_round_trip_preserves_header_and_normalizes_layout(tmp_path: Path) -> None:
    source = tmp_path / "input.fa"
    restored = tmp_path / "restored.fa"
    source.write_text(">chr1 description\nACGTN\nACGT\n", encoding="utf-8")

    record = read_fasta(source)
    write_fasta(record, restored)

    assert record == FastaRecord(header=">chr1 description", sequence="ACGTNACGT")
    assert read_fasta(restored) == record


@pytest.mark.parametrize("content", ["", "ACGT\n", ">chr1\nACGTX\n", ">chr1\n"])
def test_read_fasta_rejects_invalid_input(tmp_path: Path, content: str) -> None:
    path = tmp_path / "invalid.fa"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError):
        read_fasta(path)
