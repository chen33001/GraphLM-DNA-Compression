"""Compression benchmark helpers."""

import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, Iterable

from src.codec import compress_fasta, decompress_ghdna
from src.fasta import read_fasta
from src.residual_codec import PlainResidualCodec, ResidualCodec


def _result_summary(
    archive: dict[str, Any],
    original_size: int,
    compressed_size: int,
    sequence_length: int,
    compression_time: float,
    decompression_time: float,
) -> dict[str, Any]:
    blocks = archive["blocks"]
    return {
        "compressed_size": compressed_size,
        "bits_per_base": compressed_size * 8 / sequence_length,
        "compression_time": compression_time,
        "decompression_time": decompression_time,
        "graph_copy_blocks": sum(block["type"] == "graph_copy" for block in blocks),
        "residual_blocks": sum(
            block["type"] in {"lm_raw", "llm_residual", "raw_residual"} for block in blocks
        ),
        "raw_bits_per_base": original_size * 8 / sequence_length,
    }


def benchmark_to_results(
    input_path: str | Path,
    results_dir: str | Path = "results",
    **codec_options: Any,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Run a benchmark and persist all artifacts under one results directory."""
    input_path = Path(input_path)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    paths = {
        "archive": results_dir / f"{stem}.ghdna",
        "restored": results_dir / f"{stem}.restored.fa",
        "json_report": results_dir / f"{stem}.benchmark.json",
        "text_report": results_dir / f"{stem}.benchmark.txt",
    }
    result = benchmark_fasta(
        input_path,
        paths["archive"],
        paths["restored"],
        **codec_options,
    )
    paths["json_report"].write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["text_report"].write_text(format_benchmark(result) + "\n", encoding="utf-8")
    return result, paths


def benchmark_fasta(
    input_path: str | Path,
    archive_path: str | Path,
    restored_path: str | Path,
    **codec_options: Any,
) -> dict[str, Any]:
    input_path = Path(input_path)
    archive_path = Path(archive_path)
    sequence_length = len(read_fasta(input_path).sequence)

    original_data = input_path.read_bytes()
    original_size = len(original_data)
    benchmark_variants = tuple(codec_options.pop("benchmark_variants", ()))
    archive, compression_time, decompression_time = _run_codec_round_trip(
        input_path,
        archive_path,
        restored_path,
        codec_options,
    )
    compressed_size = archive_path.stat().st_size

    started = perf_counter()
    gzip_data = gzip.compress(original_data, compresslevel=9)
    gzip_compression_time = perf_counter() - started

    started = perf_counter()
    gzip_restored = gzip.decompress(gzip_data)
    gzip_decompression_time = perf_counter() - started
    if gzip_restored != original_data:
        raise RuntimeError("gzip round-trip verification failed")

    gzip_size = len(gzip_data)
    result = {
        "original_size": original_size,
        "gzip_size": gzip_size,
        "gzip_bits_per_base": gzip_size * 8 / sequence_length,
        "gzip_compression_time": gzip_compression_time,
        "gzip_decompression_time": gzip_decompression_time,
    }
    result.update(
        _result_summary(
            archive,
            original_size,
            compressed_size,
            sequence_length,
            compression_time,
            decompression_time,
        )
    )
    if benchmark_variants:
        result["variants"] = _benchmark_variants(
            input_path,
            original_size,
            sequence_length,
            codec_options,
            benchmark_variants,
        )
    return result


def _run_codec_round_trip(
    input_path: Path,
    archive_path: Path,
    restored_path: Path,
    codec_options: dict[str, Any],
) -> tuple[dict[str, Any], float, float]:
    started = perf_counter()
    archive = compress_fasta(input_path, archive_path, **codec_options)
    compression_time = perf_counter() - started

    started = perf_counter()
    decompress_ghdna(
        archive_path,
        restored_path,
        residual_codec=codec_options.get("residual_codec"),
    )
    decompression_time = perf_counter() - started
    return archive, compression_time, decompression_time


def _benchmark_variants(
    input_path: Path,
    original_size: int,
    sequence_length: int,
    codec_options: dict[str, Any],
    benchmark_variants: Iterable[str],
) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    residual_codec = codec_options.get("residual_codec")
    for variant in benchmark_variants:
        variant_options = dict(codec_options)
        if variant == "graph_only":
            variant_options.update(
                {
                    "archive_version": 2,
                    "residual_codec": PlainResidualCodec(),
                    "use_graph_repeats": True,
                }
            )
        elif variant == "dnagpt2_residual_only":
            _require_dnagpt2_codec(residual_codec)
            variant_options.update(
                {
                    "archive_version": 2,
                    "residual_codec": residual_codec,
                    "use_graph_repeats": False,
                }
            )
        elif variant == "graph_plus_dnagpt2":
            _require_dnagpt2_codec(residual_codec)
            variant_options.update(
                {
                    "archive_version": 2,
                    "residual_codec": residual_codec,
                    "use_graph_repeats": True,
                }
            )
        else:
            raise ValueError(f"unknown benchmark variant: {variant}")

        with TemporaryDirectory(prefix="graphlm-benchmark-") as temp_dir:
            temp_root = Path(temp_dir)
            archive_path = temp_root / f"{variant}.ghdna"
            restored_path = temp_root / f"{variant}.restored.fa"
            archive, compression_time, decompression_time = _run_codec_round_trip(
                input_path,
                archive_path,
                restored_path,
                variant_options,
            )
            summary = _result_summary(
                archive,
                original_size,
                archive_path.stat().st_size,
                sequence_length,
                compression_time,
                decompression_time,
            )
            summary["name"] = variant
            variants.append(summary)
    return variants


def _require_dnagpt2_codec(residual_codec: ResidualCodec | None) -> None:
    if residual_codec is None or not residual_codec.codec_id.startswith("dnagpt2"):
        raise ValueError("DNAGPT2 benchmark variants require a DNAGPT2 residual codec")


def format_benchmark(result: dict[str, Any]) -> str:
    formatted = (
        "Method        Size(bytes)   bpb     Compress(s)   Decompress(s)\n"
        f"GHDNA         {result['compressed_size']:<13} "
        f"{result['bits_per_base']:<7.2f} "
        f"{result['compression_time']:<13.6f} {result['decompression_time']:.6f}\n"
        f"gzip -9       {result['gzip_size']:<13} {result['gzip_bits_per_base']:<7.2f} "
        f"{result['gzip_compression_time']:<13.6f} {result['gzip_decompression_time']:.6f}\n"
        f"raw FASTA     {result['original_size']:<13} {result['raw_bits_per_base']:<7.2f} -             -"
    )
    variants = result.get("variants", [])
    if not variants:
        return formatted
    lines = [formatted, "", "Variants", "Method                  Size(bytes)   bpb     Compress(s)   Decompress(s)"]
    for variant in variants:
        lines.append(
            f"{variant['name']:<23} {variant['compressed_size']:<13} "
            f"{variant['bits_per_base']:<7.2f} "
            f"{variant['compression_time']:<13.6f} {variant['decompression_time']:.6f}"
        )
    return "\n".join(lines)
