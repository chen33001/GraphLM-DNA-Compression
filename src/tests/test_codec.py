import json
from pathlib import Path

import pytest

from src.archive import read_archive, write_archive
from src.archive_v2 import BINARY_MAGIC, read_binary_archive
from src.codec import (
    _pack_dna_2bit,
    _unpack_dna_2bit,
    compress_fasta,
    decompress_ghdna,
)
from src.fasta import read_fasta
from src.residual_codec import EncodedResidual, ResidualCodec


class ChunkingResidualCodec(ResidualCodec):
    codec_id = "chunking_test"

    def split_sequence(self, sequence: str) -> list[str]:
        return [sequence[:4], sequence[4:]] if len(sequence) > 4 else [sequence]

    def encode(self, sequence: str) -> EncodedResidual:
        payload = sequence.encode("ascii")
        return EncodedResidual(payload=payload, bit_length=len(payload) * 8, symbol_count=len(sequence))

    def decode(self, encoded: EncodedResidual) -> str:
        return encoded.payload.decode("ascii")


def test_archive_json_is_deterministic_and_validated(tmp_path: Path) -> None:
    path = tmp_path / "sample.ghdna"
    archive = {"magic": "GHDNA_1", "length": 0, "blocks": []}

    write_archive(archive, path)

    assert read_archive(path) == archive
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_archive_rejects_unknown_magic(tmp_path: Path) -> None:
    path = tmp_path / "bad.ghdna"
    path.write_text('{"magic":"OTHER","blocks":[]}', encoding="utf-8")

    with pytest.raises(ValueError, match="magic"):
        read_archive(path)


def test_codec_round_trip_uses_graph_copy_block(tmp_path: Path) -> None:
    source = tmp_path / "input.fa"
    archive_path = tmp_path / "output.ghdna"
    restored = tmp_path / "restored.fa"
    source.write_text(">chr1\nACGTACGTGGACGTACGT\n", encoding="utf-8")

    archive = compress_fasta(source, archive_path, k=3, min_repeat_len=8, lm_order=2)
    decompress_ghdna(archive_path, restored)

    assert read_fasta(restored) == read_fasta(source)
    assert any(block["type"] == "graph_copy" for block in archive["blocks"])


def test_decompression_rejects_checksum_mismatch(tmp_path: Path) -> None:
    archive_path = tmp_path / "bad.ghdna"
    restored = tmp_path / "restored.fa"
    archive = {
        "magic": "GHDNA_1",
        "header": ">bad",
        "length": 4,
        "sha256": "0" * 64,
        "parameters": {"k": 3, "lm_order": 2},
        "blocks": [{"type": "lm_raw", "start": 0, "seq": "ACGT", "lm_bits": 8.0}],
    }
    archive_path.write_text(json.dumps(archive), encoding="utf-8")

    with pytest.raises(ValueError, match="checksum"):
        decompress_ghdna(archive_path, restored)


def test_binary_v2_round_trip_separates_residual_payload(tmp_path: Path) -> None:
    source = tmp_path / "input.fa"
    archive_path = tmp_path / "output.ghdna"
    restored = tmp_path / "restored.fa"
    source.write_text(">chr1\nACGTACGTGGACGTACGT\n", encoding="utf-8")

    metadata = compress_fasta(
        source,
        archive_path,
        k=3,
        min_repeat_len=8,
        lm_order=2,
        archive_version=2,
    )
    decompress_ghdna(archive_path, restored)
    binary_archive = read_binary_archive(archive_path)

    assert archive_path.read_bytes().startswith(BINARY_MAGIC)
    assert metadata["magic"] == "GHDNA_2"
    assert binary_archive.blocks == metadata["blocks"]
    assert binary_archive.metadata["magic"] == metadata["magic"]
    assert binary_archive.metadata["residual_codec"] == metadata["residual_codec"]
    assert read_fasta(restored) == read_fasta(source)
    residuals = [block for block in metadata["blocks"] if block["type"] == "llm_residual"]
    assert residuals
    assert all("seq" not in block for block in residuals)


def test_binary_v2_chunked_residual_codec_emits_multiple_contiguous_residual_blocks(tmp_path: Path) -> None:
    source = tmp_path / "input.fa"
    archive_path = tmp_path / "output.ghdna"
    restored = tmp_path / "restored.fa"
    source.write_text(">chr1\nACGTACGTGGACGTACGT\n", encoding="utf-8")

    metadata = compress_fasta(
        source,
        archive_path,
        k=50,
        min_repeat_len=100,
        archive_version=2,
        residual_codec=ChunkingResidualCodec(),
    )
    decompress_ghdna(archive_path, restored, residual_codec=ChunkingResidualCodec())

    residuals = [block for block in metadata["blocks"] if block["type"] == "llm_residual"]
    assert len(residuals) > 1
    assert [block["start"] for block in residuals] == [0, 4]
    assert [block["length"] for block in residuals] == [4, len(read_fasta(source).sequence) - 4]
    assert read_fasta(restored) == read_fasta(source)


def test_raw_residual_uses_2bit_packing_for_acgt_only() -> None:
    packed = _pack_dna_2bit("ACGTAC")

    assert packed == bytes([0b00011011, 0b00010000])
    assert _unpack_dna_2bit(packed, 6) == "ACGTAC"


def test_raw_residual_2bit_packing_rejects_non_acgt_symbols() -> None:
    with pytest.raises(ValueError, match="A/C/G/T"):
        _pack_dna_2bit("ACGTN")
