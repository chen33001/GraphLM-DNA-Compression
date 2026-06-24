# GraphLM-DNA-Compression

Graph-theoretic and language-model-based DNA compression with benchmarking of compression ratio and decompression performance.

## Development setup

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/), then create the project environment and install the locked dependencies:

```bash
uv sync
```

Run commands inside the environment without activating it:

```bash
uv run pytest
```

Alternatively, activate the generated `.venv` using the command for your shell.

## Usage

```bash
uv run python main.py compress examples/chm13_chr1_500bp.fa --k 15 --min-repeat-len 40
uv run python main.py decompress outputs/chm13_chr1_500bp.ghdna
uv run python main.py benchmark examples/chm13_chr1_500bp.fa --k 15 --min-repeat-len 40
uv run python main.py visualize-graph examples/chm13_chr1_500bp.fa --k 15 --max-edges 50
```

When `compress` or `decompress` is called without an explicit output path, artifacts are written under `outputs/` by default:

```text
outputs/
|- chm13_chr1_500bp.ghdna
`- chm13_chr1_500bp.restored.fa
```

The benchmark command writes its artifacts to `results/` by default:

```text
results/
|- chm13_chr1_500bp.ghdna
|- chm13_chr1_500bp.restored.fa
|- chm13_chr1_500bp.benchmark.json
`- chm13_chr1_500bp.benchmark.txt
```

The benchmark measures archive size, bits per base, compression time, and decompression time for both GHDNA and `gzip -9`. gzip decompression is verified against the original FASTA bytes. Timings are single-run wall-clock measurements and can vary between runs.

The `visualize-graph` command writes a Mermaid markdown file to `outputs/` by default:

```text
outputs/
`- chm13_chr1_500bp.debruijn.md
```

Choose another output directory when needed:

```bash
uv run python main.py benchmark examples/chm13_chr1_500bp.fa --results-dir path/to/results --k 15 --min-repeat-len 40
```

When benchmarking the DNAGPT2 path, you can request additional ablation runs in the same report:

```bash
uv run python main.py benchmark examples/chm13_chr1_500bp.fa --archive-version 2 --residual-codec dnagpt2 --device cpu --k 15 --min-repeat-len 40 --benchmark-variant graph_only --benchmark-variant dnagpt2_residual_only --benchmark-variant graph_plus_dnagpt2
```

## Genomic benchmark examples

The current `examples/` directory contains:

| File | Reference window | Purpose |
|---|---|---|
| `chm13_chr1_500bp.fa` | T2T-CHM13v2.0 chr1, first 500 bp | Small reproducible smoke-test input |
| `chm13v2.0.fa` | T2T-CHM13v2.0 source FASTA | Large human source FASTA for extracting windows |
| `Arabidopsis_Chr1.fa` | TAIR10 chr1 source FASTA | Large plant source FASTA for extracting windows |

Run them with:

```bash
uv run python main.py benchmark examples/chm13_chr1_500bp.fa --k 15 --min-repeat-len 40
uv run python main.py visualize-graph examples/chm13_chr1_500bp.fa --k 15 --max-edges 50
```

For larger experiments, extract a bounded window from `chm13v2.0.fa` or `Arabidopsis_Chr1.fa` first instead of benchmarking the full source FASTA directly.

```bash
uv run python main.py benchmark examples/chm13_chr1_500bp.fa --archive-version 2 --residual-codec dnagpt2 --device cpu --k 15 --min-repeat-len 40
```

The current small example is normalized to uppercase `A/C/G/T/N` FASTA and is intended for fast validation before moving to larger extracted windows.

## Experimental hybrid DNAGPT2 mode

The default path remains `GHDNA_1`, a readable JSON archive with Markov-scored residual blocks. An experimental `GHDNA_2` path stores residual payloads out of line and can use either a plain residual codec or a pinned DNAGPT2-backed arithmetic codec.

```bash
uv run python main.py compress examples/chm13_chr1_500bp.fa hybrid.ghdna --archive-version 2 --residual-codec dnagpt2 --device cpu --k 15 --min-repeat-len 40
uv run python main.py decompress hybrid.ghdna hybrid.restored.fa --residual-codec dnagpt2 --device cpu
uv run python main.py benchmark examples/chm13_chr1_500bp.fa --archive-version 2 --residual-codec dnagpt2 --device cpu --k 15 --min-repeat-len 40
```

The current DNAGPT2 backend already loads the trained `vojtam/DNAGPT2_32` checkpoint by default, pinned to revision `cc28f01babc5271d96c65499682868e1fe40baaa`. When a complete local `model/` directory is present (`config.json`, `model.safetensors`, `tokenizer.json`), the backend prefers that local copy automatically before checking Hugging Face cache or the remote repository. Decompression requires an explicit `--residual-codec dnagpt2` selection for that path; the decoder does not auto-load external model weights from archive metadata.

DNAGPT2 runtime parameters are configurable from the CLI when needed:

```bash
uv run python main.py compress examples/chm13_chr1_500bp.fa --archive-version 2 --residual-codec dnagpt2 --model-repository vojtam/DNAGPT2_32 --model-revision cc28f01babc5271d96c65499682868e1fe40baaa --context-length 1024 --stride 512 --cdf-scale 32768 --state-bits 64
```

`GHDNA_2` is functionally lossless but still experimental. The current implementation resets LLM context per residual block, splits long residuals into context-safe chunks, and uses exact stepwise probability generation for correctness.

## Complexity analysis

The notebook-style complexity analysis has been split into a reusable module at `src/complexity_analysis.py`. It currently exposes:

- Shannon entropy in bits per base
- LZ76-style complexity
- GC fraction and length summaries

This is intended for analysis workflows and benchmark interpretation rather than archive production.

Re-running a benchmark for the same input filename overwrites that input's existing result artifacts.

See [the project specification](docs/PROJECT_SPECIFICATION.md), [implementation log](docs/IMPLEMENTATION_LOG.md), and [third-party notices](THIRD_PARTY_NOTICES.md) for architecture, verification, and Apache-2.0 attribution details.
