# GraphLM-DNA-Compression

## Goal of this project

GraphLM-DNA-Compression is a lossless DNA compression prototype for studying whether graph-based repeat detection and language-model-based residual coding can improve genomic compression.

The current project focuses on:

- using de Bruijn graph structure to find repeated regions in DNA sequences;
- using DNAGPT2-guided arithmetic coding for non-repeated residual regions;
- benchmarking `graph_only`, `dnagpt2_only`, and `graph_plus_dnagpt2` against `gzip -9`.

## Architecture

The current compression pipeline is:

1. Read one FASTA record.
2. Build a de Bruijn graph from the sequence.
3. Find repeated regions using graph-seeded repeat discovery.
4. Split the sequence into an ordered stream of block types:
   - `graph_copy`
   - `llm_residual`
   - `raw_residual`
5. Write a versioned archive and verify exact round-trip restoration.

### Where graph theory is used

Graph theory is used to find repeated sequence regions, not to serialize the graph itself.

- A de Bruijn graph is built from the DNA sequence.
- The repeat finder uses `k`-mer overlap structure and prior positions to locate greedy non-overlapping repeated regions.
- Each repeated region is encoded as a `graph_copy` back-reference:
  - `start`
  - `source`
  - `length`

So the graph stage acts like a repeat-copy detector over genomic sequence.

### How DNAGPT2 is used

DNAGPT2 is used only on residual sequence that is not covered by `graph_copy` blocks.

- The residual DNA substring is tokenized.
- DNAGPT2 provides next-token probabilities.
- Those probabilities are converted into integer CDFs.
- Arithmetic coding compresses the token stream into the final residual payload.

This means:

- graph stage: finds what can be copied;
- DNAGPT2 stage: compresses what cannot be copied;
- raw residual stage: stores short fallback fragments directly.

### Raw residual fallback

`raw_residual` blocks are now stored with strict 2-bit packing for `A/C/G/T`.

- `A/C/G/T` only
- `2 bits/base`
- no `N` fallback in the raw residual path

This matches the intended cleaned-data workflow for current benchmarks.

### Compression modes

The benchmark and implementation currently expose three important modes:

- `graph_only`
  - graph-copy compression plus plain residual storage
- `dnagpt2_only` (`dnagpt2_residual_only` in code/results)
  - DNAGPT2-guided arithmetic coding without graph repeats
- `graph_plus_dnagpt2`
  - forced graph-copy compression plus DNAGPT2 residual coding
- `adaptive_dnagpt2_hybrid`
  - planner-based graph-copy plus DNAGPT2 hybrid that may still choose `no_graph`

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

The benchmark measures:

- archive size
- bits per base
- compression time
- decompression time
- diagnostic fields for the current `GHDNA_2` planner

It always compares against `gzip -9`, which is used as a standard "best effort gzip" baseline.

Choose another output directory when needed:

```bash
uv run python main.py benchmark examples/chm13_chr1_500bp.fa --results-dir path/to/results --k 15 --min-repeat-len 40
```

When benchmarking the DNAGPT2 path, request the ablation variants explicitly:

```bash
uv run python main.py benchmark examples/chm13_chr1_2500bp.fa --results-dir results --archive-version 2 --residual-codec dnagpt2 --device cpu --k 15 --min-repeat-len 40 --benchmark-variant graph_only --benchmark-variant dnagpt2_residual_only --benchmark-variant graph_plus_dnagpt2
```

## Benchmark results

### `1000bp`

- sequence file: `examples/chm13_chr1_1000bp.fa`
- sequence length: `1000 bp`
- original FASTA size: `1052 bytes`

Latest reported results:

| Method | Size (bytes) | bpb | Compress (s) | Decompress (s) |
|---|---:|---:|---:|---:|
| `graph_only` | 1298 | 10.38 | 0.0049 | 0.0008 |
| `dnagpt2_only` | 1294 | 10.35 | 73.25 | 73.03 |
| `graph_plus_dnagpt2` | 1380 | 11.04 | 54.08 | 54.17 |
| `gzip -9` | 115 | 0.92 | 0.0001 | 0.0001 |

Diagnostics for the selected top-level archive (`adaptive_dnagpt2_hybrid` behavior):

- `selected_candidate = pruned_graph`
- `graph_copy_bases = 353`
- `residual_bases = 647`
- `llm_residual_blocks = 2`
- `raw_residual_blocks = 0`

Interpretation:

- `graph_plus_dnagpt2` is currently worse than `dnagpt2_only` in `bpb` on this input.
- It is faster than `dnagpt2_only`, but still much slower than `graph_only`.
- On this window, the hybrid graph stage is not yet justified by compression ratio.

### `2500bp`

- sequence file: `examples/chm13_chr1_2500bp.fa`
- sequence length: `2500 bp`
- original FASTA size: `2590 bytes`

Latest reported results:

| Method | Size (bytes) | bpb | Compress (s) | Decompress (s) |
|---|---:|---:|---:|---:|
| `graph_only` | 1971 | 6.31 | 0.0480 | 0.0014 |
| `dnagpt2_only` | 1397 | 4.47 | 358.80 | 361.04 |
| `graph_plus_dnagpt2` | 1884 | 6.03 | 33.06 | 32.97 |
| `gzip -9` | 229 | 0.73 | 0.0002 | 0.0001 |

Diagnostics for the selected top-level archive (`adaptive_dnagpt2_hybrid` behavior):

- `selected_candidate = no_graph`
- `graph_copy_bases = 0`
- `residual_bases = 2500`
- `llm_residual_blocks = 2`
- `raw_residual_blocks = 0`

Interpretation:

- `graph_only` is extremely fast but compresses worse than the DNAGPT2-based methods.
- `dnagpt2_only` gives the best compression ratio among the reported methods, but is expensive in time.
- `graph_plus_dnagpt2` is much faster than `dnagpt2_only` on this input, but it loses a lot in `bpb`.
- In the forced-graph ablation, `selected_candidate = force_graph`, `graph_copy_bases = 1946`, and `raw_residual_blocks = 7`.
- On `2500bp`, forcing graph copies improves runtime substantially, but not enough to justify the compression-ratio loss.

### `5000bp`

- sequence file: `examples/chm13_chr1_5000bp.fa`
- sequence length: `5000 bp`
- original FASTA size: `5152 bytes`

Latest reported results:

| Method | Size (bytes) | bpb | Compress (s) | Decompress (s) |
|---|---:|---:|---:|---:|
| `graph_only` | 3951 | 6.32 | 0.0347 | 0.0009 |
| `dnagpt2_only` | 1808 | 2.89 | 973.40 | 973.14 |
| `graph_plus_dnagpt2` | 2425 | 3.88 | 168.29 | 168.47 |
| `gzip -9` | 880 | 1.41 | 0.0004 | 0.0001 |

Diagnostics for the selected top-level archive (`adaptive_dnagpt2_hybrid` behavior):

- `selected_candidate = no_graph`
- `graph_copy_bases = 0`
- `residual_bases = 5000`
- `llm_residual_blocks = 3`
- `raw_residual_blocks = 0`

Interpretation:

- `dnagpt2_only` again gives the best compression ratio among the project methods.
- `graph_plus_dnagpt2` is much faster than `dnagpt2_only` on this longer input.
- In the forced-graph ablation, `selected_candidate = force_graph`, `graph_copy_bases = 2642`, and `raw_residual_blocks = 6`.
- The forced graph path still compresses worse than `dnagpt2_only`, but the runtime gap is now large enough to show a real speed/compression tradeoff.

### Current conclusion

- At the moment, the hybrid method should still be treated as experimental.
- The `1000bp` result shows a real graph-assisted hybrid layout, but it is still worse than `dnagpt2_only` in `bpb`.
- The planner-based top-level archive still selects `no_graph` on both `2500bp` and `5000bp`.
- The forced `graph_plus_dnagpt2` ablation is now clearly measurable: it is substantially faster than `dnagpt2_only` on `2500bp` and `5000bp`, but it still loses on compression ratio.
- The current evidence suggests the graph stage is acting more like a speed/structure tradeoff than a direct compression win when combined with DNAGPT2.

## DNAGPT2 model source

The DNAGPT2 backend used in this project is based on the `vojtam/DNAGPT2_32` model, pinned in the current implementation to revision:

`cc28f01babc5271d96c65499682868e1fe40baaa`

When a complete local `model/` directory is present (`config.json`, `model.safetensors`, `tokenizer.json`), the backend prefers that local copy automatically before checking Hugging Face cache or the remote repository.
