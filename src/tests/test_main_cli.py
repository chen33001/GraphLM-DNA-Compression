from pathlib import Path

from main import _default_graph_visualization_output, build_parser


def test_cli_parser_accepts_dnagpt2_backend_configuration() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "compress",
            "input.fa",
            "--archive-version",
            "2",
            "--residual-codec",
            "dnagpt2",
            "--model-repository",
            "vojtam/DNAGPT2_128",
            "--model-revision",
            "abc123",
            "--context-length",
            "512",
            "--cdf-scale",
            "16384",
            "--stride",
            "128",
            "--state-bits",
            "32",
        ]
    )

    assert args.input == Path("input.fa")
    assert args.model_repository == "vojtam/DNAGPT2_128"
    assert args.model_revision == "abc123"
    assert args.context_length == 512
    assert args.cdf_scale == 16384
    assert args.stride == 128
    assert args.state_bits == 32


def test_cli_parser_accepts_visualize_graph_configuration() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "visualize-graph",
            "input.fa",
            "--k",
            "9",
            "--max-edges",
            "20",
            "--min-count",
            "2",
        ]
    )

    assert args.input == Path("input.fa")
    assert args.k == 9
    assert args.max_edges == 20
    assert args.min_count == 2


def test_default_graph_visualization_output_uses_outputs_directory() -> None:
    assert _default_graph_visualization_output(Path("example.fa")) == Path(
        "outputs/example.debruijn.md"
    )
