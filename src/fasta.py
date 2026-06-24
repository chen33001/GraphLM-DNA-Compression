"""Single-record FASTA input and output."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FastaRecord:
    header: str
    sequence: str


def read_fasta(path: str | Path) -> FastaRecord:
    """Read and validate one non-empty FASTA record."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines or not lines[0].startswith(">") or not lines[0][1:].strip():
        raise ValueError("FASTA must start with a non-empty header")
    if any(line.startswith(">") for line in lines[1:]):
        raise ValueError("only single-record FASTA files are supported")

    sequence = "".join(line.strip() for line in lines[1:]).upper()
    if not sequence:
        raise ValueError("FASTA sequence must not be empty")
    invalid = set(sequence) - set("ACGTN")
    if invalid:
        raise ValueError(f"unsupported FASTA symbols: {''.join(sorted(invalid))}")
    return FastaRecord(header=lines[0], sequence=sequence)


def write_fasta(record: FastaRecord, path: str | Path, line_width: int = 80) -> None:
    """Write a FASTA record with deterministic line wrapping."""
    if line_width <= 0:
        raise ValueError("line_width must be positive")
    content = [record.header]
    content.extend(
        record.sequence[index : index + line_width]
        for index in range(0, len(record.sequence), line_width)
    )
    Path(path).write_text("\n".join(content) + "\n", encoding="utf-8")
