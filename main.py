"""Command-line interface for GraphLM-HDNA."""

import argparse
from pathlib import Path

from src.benchmark import benchmark_to_results, format_benchmark
from src.codec import compress_fasta, decompress_ghdna
from src.fasta import read_fasta
from src.graph import build_debruijn_graph, graph_to_mermaid
from src.residual_codec import PlainResidualCodec, ResidualCodec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GraphLM-HDNA compression prototype")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compress = subparsers.add_parser("compress", help="compress a FASTA file")
    compress.add_argument("input", type=Path)
    compress.add_argument("output", type=Path, nargs="?")
    _add_codec_options(compress)

    decompress = subparsers.add_parser("decompress", help="restore a .ghdna archive")
    decompress.add_argument("input", type=Path)
    decompress.add_argument("output", type=Path, nargs="?")
    decompress.add_argument(
        "--residual-codec",
        choices=("auto", "plain", "dnagpt2"),
        default="auto",
        help="GHDNA_2 residual decoder; DNAGPT2 loading must be requested explicitly",
    )
    decompress.add_argument("--device", default="cpu", help="DNAGPT2 torch device")

    benchmark = subparsers.add_parser("benchmark", help="benchmark a FASTA file")
    benchmark.add_argument("input", type=Path)
    benchmark.add_argument("--results-dir", type=Path, default=Path("results"))
    benchmark.add_argument(
        "--benchmark-variant",
        action="append",
        choices=("graph_only", "dnagpt2_residual_only", "graph_plus_dnagpt2"),
        default=[],
        help="additional benchmark methods to run and report",
    )
    _add_codec_options(benchmark)

    visualize = subparsers.add_parser(
        "visualize-graph", help="export the de Bruijn graph as a Mermaid markdown file"
    )
    visualize.add_argument("input", type=Path)
    visualize.add_argument("output", type=Path, nargs="?")
    visualize.add_argument("--k", type=int, default=15)
    visualize.add_argument("--max-edges", type=int, default=50)
    visualize.add_argument("--min-count", type=int, default=1)
    return parser


def _add_codec_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--min-repeat-len", type=int, default=40)
    parser.add_argument("--lm-order", type=int, default=3)
    parser.add_argument("--archive-version", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--residual-codec",
        choices=("plain", "dnagpt2"),
        default="plain",
        help="GHDNA_2 residual codec",
    )
    parser.add_argument("--device", default="cpu", help="DNAGPT2 torch device")
    parser.add_argument("--model-repository", default="vojtam/DNAGPT2_32")
    parser.add_argument(
        "--model-revision",
        default="cc28f01babc5271d96c65499682868e1fe40baaa",
    )
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--cdf-scale", type=int, default=32768)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--state-bits", type=int, default=64)


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "decompress":
        output = args.output or _default_decompress_output(args.input)
        output.parent.mkdir(parents=True, exist_ok=True)
        decompress_ghdna(args.input, output, residual_codec=_build_decoder(args))
        return 0
    if args.command == "visualize-graph":
        output = args.output or _default_graph_visualization_output(args.input)
        output.parent.mkdir(parents=True, exist_ok=True)
        _write_graph_visualization(
            args.input,
            output,
            k=args.k,
            max_edges=args.max_edges,
            min_count=args.min_count,
        )
        print(f"Graph visualization written to {output}")
        return 0
    residual_codec = _build_residual_codec(args)
    options = {
        "k": args.k,
        "min_repeat_len": args.min_repeat_len,
        "lm_order": args.lm_order,
        "archive_version": args.archive_version,
        "residual_codec": residual_codec,
    }
    if args.command == "compress":
        output = args.output or _default_compress_output(args.input)
        output.parent.mkdir(parents=True, exist_ok=True)
        compress_fasta(args.input, output, **options)
        return 0
    result, paths = benchmark_to_results(
        args.input,
        args.results_dir,
        benchmark_variants=args.benchmark_variant,
        **options,
    )
    print(format_benchmark(result))
    print(f"Results written to {paths['json_report'].parent}")
    return 0


def _default_compress_output(input_path: Path) -> Path:
    return Path("outputs") / f"{input_path.stem}.ghdna"


def _default_decompress_output(input_path: Path) -> Path:
    return Path("outputs") / f"{input_path.stem}.restored.fa"


def _default_graph_visualization_output(input_path: Path) -> Path:
    return Path("outputs") / f"{input_path.stem}.debruijn.md"


def _write_graph_visualization(
    input_path: Path,
    output_path: Path,
    *,
    k: int,
    max_edges: int,
    min_count: int,
) -> None:
    record = read_fasta(input_path)
    graph = build_debruijn_graph(record.sequence, k)
    mermaid = graph_to_mermaid(graph, max_edges=max_edges, min_count=min_count)
    output_path.write_text(
        "\n".join(
            [
                f"# de Bruijn graph for {input_path.name}",
                "",
                f"- header: `{record.header}`",
                f"- sequence length: `{len(record.sequence)}`",
                f"- k: `{k}`",
                f"- min edge count shown: `{min_count}`",
                f"- max edges shown: `{max_edges}`",
                "",
                "```mermaid",
                mermaid,
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _build_residual_codec(args: argparse.Namespace) -> ResidualCodec | None:
    if args.archive_version == 1:
        if args.residual_codec != "plain":
            raise ValueError("DNAGPT2 residual coding requires --archive-version 2")
        return None
    if args.residual_codec == "plain":
        return PlainResidualCodec()
    from src.dnagpt2_backend import HuggingFaceDNAGPT2Backend
    from src.llm_residual_codec import DNAGPT2ResidualCodec

    return _build_dnagpt2_codec(args)


def _build_decoder(args: argparse.Namespace) -> ResidualCodec | None:
    if args.residual_codec == "auto":
        return None
    if args.residual_codec == "plain":
        return PlainResidualCodec()
    from src.dnagpt2_backend import HuggingFaceDNAGPT2Backend
    from src.llm_residual_codec import DNAGPT2ResidualCodec

    return _build_dnagpt2_codec(args)


def _build_dnagpt2_codec(args: argparse.Namespace) -> ResidualCodec:
    from src.dnagpt2_backend import HuggingFaceDNAGPT2Backend
    from src.llm_residual_codec import DNAGPT2ResidualCodec

    return DNAGPT2ResidualCodec(
        HuggingFaceDNAGPT2Backend(
            model_repository=args.model_repository,
            revision=args.model_revision,
            device=args.device,
            context_length=args.context_length,
            cdf_scale=args.cdf_scale,
            stride=args.stride,
        ),
        state_bits=args.state_bits,
    )


if __name__ == "__main__":
    raise SystemExit(main())
