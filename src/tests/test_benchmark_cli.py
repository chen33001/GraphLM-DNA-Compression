import json
import subprocess
import sys
from pathlib import Path

from src.benchmark import benchmark_fasta, benchmark_to_results, format_benchmark
from src.archive_v2 import BINARY_MAGIC
from src.fasta import read_fasta
from src.llm_residual_codec import DNAGPT2ResidualCodec, TokenProbabilityBackend
from src.residual_codec import EncodedResidual, PlainResidualCodec


class FakeBenchmarkBackend(TokenProbabilityBackend):
    alphabet = "ACGTN"

    @property
    def metadata(self) -> dict[str, object]:
        return {"model": "fake-benchmark", "revision": "test"}

    def tokenize(self, sequence: str) -> list[int]:
        return [self.alphabet.index(symbol) for symbol in sequence]

    def detokenize(self, token_ids: list[int]) -> str:
        return "".join(self.alphabet[token] for token in token_ids)

    def cdf_for_prefix(self, token_ids: list[int]) -> list[int]:
        return [1, 2, 3, 4, 5]


class FakeBenchmarkDNAGPT2Codec(DNAGPT2ResidualCodec):
    def __init__(self) -> None:
        super().__init__(FakeBenchmarkBackend(), state_bits=32)


def test_benchmark_reports_required_metrics(tmp_path: Path) -> None:
    source = tmp_path / "input.fa"
    archive = tmp_path / "output.ghdna"
    restored = tmp_path / "restored.fa"
    source.write_text(">sample\nACGTACGTGGACGTACGT\n", encoding="utf-8")

    result = benchmark_fasta(source, archive, restored, k=3, min_repeat_len=8)

    assert result["original_size"] > 0
    assert result["compressed_size"] > 0
    assert result["gzip_size"] > 0
    assert result["bits_per_base"] > 0
    assert result["gzip_bits_per_base"] > 0
    assert result["gzip_compression_time"] >= 0
    assert result["gzip_decompression_time"] >= 0
    assert result["raw_bits_per_base"] > 0
    assert result["graph_copy_blocks"] == 1
    formatted = format_benchmark(result)
    assert "GHDNA" in formatted
    assert f"{result['gzip_compression_time']:.6f}" in formatted
    assert f"{result['gzip_decompression_time']:.6f}" in formatted


def test_benchmark_to_results_persists_reports(tmp_path: Path) -> None:
    source = tmp_path / "sample.fa"
    source.write_text(">sample\nACGTACGTGGACGTACGT\n", encoding="utf-8")

    result, paths = benchmark_to_results(source, tmp_path / "results", k=3, min_repeat_len=8)

    assert result["graph_copy_blocks"] == 1
    assert all(path.is_file() for path in paths.values())


def test_benchmark_counts_binary_v2_residual_blocks(tmp_path: Path) -> None:
    source = tmp_path / "sample.fa"
    archive = tmp_path / "sample.ghdna"
    restored = tmp_path / "sample.restored.fa"
    source.write_text(">sample\nACGTACGTGGACGTACGT\n", encoding="utf-8")

    result = benchmark_fasta(
        source,
        archive,
        restored,
        archive_version=2,
        residual_codec=PlainResidualCodec(),
        k=3,
        min_repeat_len=8,
    )

    assert result["graph_copy_blocks"] == 1
    assert result["residual_blocks"] == 1


def test_benchmark_reports_requested_variant_methods(tmp_path: Path) -> None:
    source = tmp_path / "sample.fa"
    archive = tmp_path / "sample.ghdna"
    restored = tmp_path / "sample.restored.fa"
    source.write_text(">sample\nACGTACGTGGACGTACGT\n", encoding="utf-8")

    result = benchmark_fasta(
        source,
        archive,
        restored,
        archive_version=2,
        residual_codec=FakeBenchmarkDNAGPT2Codec(),
        benchmark_variants=("graph_only", "dnagpt2_residual_only", "graph_plus_dnagpt2"),
        k=3,
        min_repeat_len=8,
    )

    names = [variant["name"] for variant in result["variants"]]
    assert names == ["graph_only", "dnagpt2_residual_only", "graph_plus_dnagpt2"]
    assert all(variant["compressed_size"] > 0 for variant in result["variants"])


def test_cli_compresses_and_decompresses(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    source = tmp_path / "input.fa"
    archive = tmp_path / "output.ghdna"
    restored = tmp_path / "restored.fa"
    source.write_text(">sample\nACGTACGTGGACGTACGT\n", encoding="utf-8")

    compress = subprocess.run(
        [sys.executable, str(root / "main.py"), "compress", str(source), str(archive), "--k", "3", "--min-repeat-len", "8"],
        capture_output=True,
        text=True,
        check=False,
    )
    decompress = subprocess.run(
        [sys.executable, str(root / "main.py"), "decompress", str(archive), str(restored)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert compress.returncode == 0, compress.stderr
    assert decompress.returncode == 0, decompress.stderr
    assert read_fasta(restored) == read_fasta(source)


def test_cli_defaults_outputs_directory_for_compress_and_decompress(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    source = tmp_path / "input.fa"
    archive = root / "outputs" / "input.ghdna"
    restored = root / "outputs" / "input.restored.fa"
    source.write_text(">sample\nACGTACGTGGACGTACGT\n", encoding="utf-8")

    try:
        compress = subprocess.run(
            [sys.executable, str(root / "main.py"), "compress", str(source), "--k", "3", "--min-repeat-len", "8"],
            capture_output=True,
            text=True,
            check=False,
            cwd=root,
        )
        decompress = subprocess.run(
            [sys.executable, str(root / "main.py"), "decompress", str(archive)],
            capture_output=True,
            text=True,
            check=False,
            cwd=root,
        )

        assert compress.returncode == 0, compress.stderr
        assert decompress.returncode == 0, decompress.stderr
        assert archive.is_file()
        assert restored.is_file()
        assert read_fasta(restored) == read_fasta(source)
    finally:
        if archive.exists():
            archive.unlink()
        if restored.exists():
            restored.unlink()


def test_cli_benchmark_writes_all_artifacts_to_results_directory(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    source = tmp_path / "sample.fa"
    results_dir = tmp_path / "results"
    source.write_text(">sample\nACGTACGTGGACGTACGT\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(root / "main.py"),
            "benchmark",
            str(source),
            "--results-dir",
            str(results_dir),
            "--k",
            "3",
            "--min-repeat-len",
            "8",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (results_dir / "sample.ghdna").is_file()
    assert read_fasta(results_dir / "sample.restored.fa") == read_fasta(source)
    report = json.loads((results_dir / "sample.benchmark.json").read_text(encoding="utf-8"))
    assert report["graph_copy_blocks"] == 1
    assert "GHDNA" in (results_dir / "sample.benchmark.txt").read_text(encoding="utf-8")


def test_cli_binary_v2_plain_codec_round_trip(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    source = tmp_path / "input.fa"
    archive = tmp_path / "output.ghdna"
    restored = tmp_path / "restored.fa"
    source.write_text(">sample\nACGTACGTGGACGTACGT\n", encoding="utf-8")

    compress = subprocess.run(
        [
            sys.executable,
            str(root / "main.py"),
            "compress",
            str(source),
            str(archive),
            "--archive-version",
            "2",
            "--residual-codec",
            "plain",
            "--k",
            "3",
            "--min-repeat-len",
            "8",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    decompress = subprocess.run(
        [sys.executable, str(root / "main.py"), "decompress", str(archive), str(restored)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert compress.returncode == 0, compress.stderr
    assert decompress.returncode == 0, decompress.stderr
    assert archive.read_bytes().startswith(BINARY_MAGIC)
    assert read_fasta(restored) == read_fasta(source)
