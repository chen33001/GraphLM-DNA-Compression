"""GraphLM-HDNA compression and decompression pipeline."""

from pathlib import Path
from typing import Any

from src.archive import MAGIC, read_archive, write_archive
from src.archive_v2 import BINARY_MAGIC, BinaryArchive, read_binary_archive, write_binary_archive
from src.fasta import FastaRecord, read_fasta, write_fasta
from src.graph import build_debruijn_graph
from src.probability_model import MarkovGenomicLM
from src.repeats import find_graph_repeats
from src.residual_codec import EncodedResidual, PlainResidualCodec, ResidualCodec
from src.utils import sequence_checksum

_MIN_LLM_RESIDUAL_BASES = 64
_MAX_HYBRID_RESIDUAL_BLOCKS = 8
_MAX_HYBRID_RESIDUAL_FRACTION = 0.2


def compress_fasta(
    input_path: str | Path,
    output_path: str | Path,
    *,
    k: int = 15,
    min_repeat_len: int = 40,
    lm_order: int = 3,
    archive_version: int = 1,
    residual_codec: ResidualCodec | None = None,
    use_graph_repeats: bool = True,
) -> dict[str, Any]:
    record = read_fasta(input_path)
    graph = build_debruijn_graph(record.sequence, k) if len(record.sequence) > k else {}
    repeats = (
        find_graph_repeats(record.sequence, graph, k, min_repeat_len)
        if graph and use_graph_repeats
        else []
    )
    if archive_version == 2:
        return _compress_v2(
            record,
            output_path,
            repeats,
            residual_codec or PlainResidualCodec(),
            k=k,
            min_repeat_len=min_repeat_len,
            lm_order=lm_order,
        )
    if archive_version != 1:
        raise ValueError("archive_version must be 1 or 2")

    model = MarkovGenomicLM(lm_order).fit(record.sequence)
    blocks: list[dict[str, Any]] = []
    cursor = 0
    for repeat in repeats:
        if repeat["start"] > cursor:
            residual = record.sequence[cursor : repeat["start"]]
            blocks.append(_raw_block(cursor, residual, model))
        blocks.append({"type": "graph_copy", **repeat})
        cursor = repeat["start"] + repeat["length"]
    if cursor < len(record.sequence):
        blocks.append(_raw_block(cursor, record.sequence[cursor:], model))

    archive = {
        "magic": MAGIC,
        "header": record.header,
        "length": len(record.sequence),
        "sha256": sequence_checksum(record.sequence),
        "parameters": {"k": k, "lm_order": lm_order, "min_repeat_len": min_repeat_len},
        "blocks": blocks,
    }
    write_archive(archive, output_path)
    return archive


def decompress_ghdna(
    input_path: str | Path,
    output_path: str | Path,
    *,
    residual_codec: ResidualCodec | None = None,
) -> FastaRecord:
    if Path(input_path).read_bytes().startswith(BINARY_MAGIC):
        return _decompress_v2(input_path, output_path, residual_codec)
    archive = read_archive(input_path)
    sequence = ""
    for block in archive["blocks"]:
        if block.get("start") != len(sequence):
            raise ValueError("archive blocks are not contiguous")
        if block.get("type") == "lm_raw":
            sequence += block["seq"]
        elif block.get("type") == "graph_copy":
            source, length = block["source"], block["length"]
            if source < 0 or length <= 0 or source + length > len(sequence):
                raise ValueError("invalid graph-copy reference")
            sequence += sequence[source : source + length]
        else:
            raise ValueError("unknown archive block type")

    if len(sequence) != archive.get("length"):
        raise ValueError("decompressed length mismatch")
    if sequence_checksum(sequence) != archive.get("sha256"):
        raise ValueError("decompressed checksum mismatch")
    record = FastaRecord(header=archive["header"], sequence=sequence)
    write_fasta(record, output_path)
    return record


def _raw_block(start: int, sequence: str, model: MarkovGenomicLM) -> dict[str, Any]:
    return {
        "type": "lm_raw",
        "start": start,
        "seq": sequence,
        "lm_bits": round(model.score_sequence(sequence), 6),
    }


def _compress_v2(
    record: FastaRecord,
    output_path: str | Path,
    repeats: list[dict[str, int]],
    residual_codec: ResidualCodec,
    **parameters: int,
) -> dict[str, Any]:
    repeats = _select_v2_repeats(record.sequence, repeats, residual_codec)
    blocks, payload = _build_v2_blocks(record.sequence, repeats, residual_codec)

    metadata = {
        "magic": "GHDNA_2",
        "header": record.header,
        "length": len(record.sequence),
        "sha256": sequence_checksum(record.sequence),
        "parameters": {
            **parameters,
            "min_llm_residual_bases": _MIN_LLM_RESIDUAL_BASES,
            "max_hybrid_residual_blocks": _MAX_HYBRID_RESIDUAL_BLOCKS,
            "max_hybrid_residual_fraction": _MAX_HYBRID_RESIDUAL_FRACTION,
        },
        "residual_codec": residual_codec.codec_id,
        "residual_codec_metadata": residual_codec.archive_metadata(),
        "blocks": blocks,
    }
    write_binary_archive(BinaryArchive(metadata=metadata, payload=payload, blocks=blocks), output_path)
    return metadata


def _decompress_v2(
    input_path: str | Path,
    output_path: str | Path,
    residual_codec: ResidualCodec | None,
) -> FastaRecord:
    archive = read_binary_archive(input_path)
    metadata = {**archive.metadata, "blocks": archive.blocks}
    codec = residual_codec or PlainResidualCodec()
    if codec.codec_id != metadata.get("residual_codec"):
        raise ValueError("required residual codec is not available")

    sequence = ""
    archive_codec_metadata = metadata.get("residual_codec_metadata", {})
    for block in archive.blocks:
        if block.get("start") != len(sequence):
            raise ValueError("archive blocks are not contiguous")
        if block.get("type") == "graph_copy":
            source, length = block["source"], block["length"]
            if source < 0 or length <= 0 or source + length > len(sequence):
                raise ValueError("invalid graph-copy reference")
            sequence += sequence[source : source + length]
        elif block.get("type") == "llm_residual":
            offset = block["payload_offset"]
            payload_length = block["payload_length"]
            end = offset + payload_length
            if offset < 0 or payload_length < 0 or end > len(archive.payload):
                raise ValueError("invalid residual payload range")
            encoded = EncodedResidual(
                payload=archive.payload[offset:end],
                bit_length=block["bit_length"],
                symbol_count=block["symbol_count"],
                metadata=codec.decode_metadata(
                    archive_codec_metadata,
                    {"use_context": bool(block.get("use_context"))},
                ),
            )
            residual = codec.decode_with_context(encoded, sequence)
            if len(residual) != block["length"]:
                raise ValueError(
                    "decoded residual length mismatch "
                    f"at start={block['start']} expected={block['length']} "
                    f"decoded={len(residual)} symbol_count={block['symbol_count']}"
                )
            sequence += residual
        elif block.get("type") == "raw_residual":
            offset = block["payload_offset"]
            payload_length = block["payload_length"]
            end = offset + payload_length
            if offset < 0 or payload_length < 0 or end > len(archive.payload):
                raise ValueError("invalid raw residual payload range")
            residual = archive.payload[offset:end].decode("ascii")
            if len(residual) != block["length"]:
                raise ValueError("decoded raw residual length mismatch")
            sequence += residual
        else:
            raise ValueError("unknown archive block type")

    if len(sequence) != metadata.get("length"):
        raise ValueError("decompressed length mismatch")
    if sequence_checksum(sequence) != metadata.get("sha256"):
        raise ValueError("decompressed checksum mismatch")
    record = FastaRecord(header=metadata["header"], sequence=sequence)
    write_fasta(record, output_path)
    return record


def _build_v2_blocks(
    sequence: str,
    repeats: list[dict[str, int]],
    residual_codec: ResidualCodec,
) -> tuple[list[dict[str, Any]], bytes]:
    blocks: list[dict[str, Any]] = []
    payload = bytearray()
    cursor = 0

    def append_raw_residual(start: int, chunk: str) -> None:
        if not chunk:
            return
        offset = len(payload)
        raw_payload = chunk.encode("ascii")
        payload.extend(raw_payload)
        blocks.append(
            {
                "type": "raw_residual",
                "start": start,
                "length": len(chunk),
                "payload_offset": offset,
                "payload_length": len(raw_payload),
            }
        )

    def append_llm_residual(start: int, chunk: str, context: str) -> None:
        encoded = residual_codec.encode_with_context(chunk, context)
        offset = len(payload)
        payload.extend(encoded.payload)
        block = {
            "type": "llm_residual",
            "start": start,
            "length": len(chunk),
            "symbol_count": encoded.symbol_count,
            "bit_length": encoded.bit_length,
            "payload_offset": offset,
            "payload_length": len(encoded.payload),
        }
        block.update(residual_codec.block_metadata(encoded))
        blocks.append(block)

    def append_residual(start: int, residual: str) -> None:
        chunk_start = start
        for chunk in residual_codec.split_sequence(residual):
            if not chunk:
                continue
            if (
                _uses_dnagpt2_residuals(residual_codec)
                and len(chunk) < _MIN_LLM_RESIDUAL_BASES
            ):
                append_raw_residual(chunk_start, chunk)
            else:
                context = sequence[max(0, chunk_start - _context_window(residual_codec)) : chunk_start]
                append_llm_residual(chunk_start, chunk, context)
            chunk_start += len(chunk)

    for repeat in repeats:
        if repeat["start"] > cursor:
            append_residual(cursor, sequence[cursor : repeat["start"]])
        blocks.append({"type": "graph_copy", **repeat})
        cursor = repeat["start"] + repeat["length"]
    if cursor < len(sequence):
        append_residual(cursor, sequence[cursor:])
    return blocks, bytes(payload)


def _select_v2_repeats(
    sequence: str,
    repeats: list[dict[str, int]],
    residual_codec: ResidualCodec,
) -> list[dict[str, int]]:
    if not repeats or not _uses_dnagpt2_residuals(residual_codec):
        return repeats

    candidates = [
        [],
        _prune_repeats_for_hybrid(len(sequence), repeats, _MIN_LLM_RESIDUAL_BASES),
    ]
    best_repeats = candidates[0]
    best_size = None
    for candidate in candidates:
        blocks, payload = _build_v2_blocks(sequence, candidate, residual_codec)
        if candidate and _hybrid_is_too_fragmented(len(sequence), blocks):
            continue
        estimated_size = len(payload) + _estimated_block_table_size(blocks)
        if best_size is None or estimated_size < best_size:
            best_size = estimated_size
            best_repeats = candidate
    return best_repeats


def _prune_repeats_for_hybrid(
    sequence_length: int, repeats: list[dict[str, int]], min_residual_bases: int
) -> list[dict[str, int]]:
    pruned = list(repeats)
    while True:
        updated = _prune_repeats_pass(sequence_length, pruned, min_residual_bases)
        if len(updated) == len(pruned):
            return updated
        pruned = updated


def _prune_repeats_pass(
    sequence_length: int, repeats: list[dict[str, int]], min_residual_bases: int
) -> list[dict[str, int]]:
    if not repeats:
        return []
    kept: list[dict[str, int]] = []
    cursor = 0
    total = len(repeats)
    for index, repeat in enumerate(repeats):
        residual_before = repeat["start"] - cursor
        next_start = (
            repeats[index + 1]["start"] if index + 1 < total else sequence_length
        )
        residual_after = next_start - (repeat["start"] + repeat["length"])
        has_neighbor = index > 0 or index + 1 < total
        if has_neighbor and (
            0 < residual_before < min_residual_bases
            or 0 < residual_after < min_residual_bases
        ):
            continue
        kept.append(repeat)
        cursor = repeat["start"] + repeat["length"]
    return kept


def _estimated_block_table_size(blocks: list[dict[str, Any]]) -> int:
    size = 0
    for block in blocks:
        if block["type"] == "graph_copy":
            size += 13
        elif block["type"] == "llm_residual":
            size += 26
        elif block["type"] == "raw_residual":
            size += 17
    return size


def _hybrid_is_too_fragmented(sequence_length: int, blocks: list[dict[str, Any]]) -> bool:
    residual_blocks = [
        block for block in blocks if block["type"] in {"llm_residual", "raw_residual"}
    ]
    if len(residual_blocks) > _MAX_HYBRID_RESIDUAL_BLOCKS:
        return True
    residual_bases = sum(int(block["length"]) for block in residual_blocks)
    if not residual_blocks:
        return False
    short_blocks = sum(int(block["length"]) < _MIN_LLM_RESIDUAL_BASES for block in residual_blocks)
    return (
        short_blocks / len(residual_blocks) > _MAX_HYBRID_RESIDUAL_FRACTION
        and residual_bases < sequence_length
    )


def _context_window(residual_codec: ResidualCodec) -> int:
    archive_metadata = residual_codec.archive_metadata()
    backend = archive_metadata.get("backend")
    if isinstance(backend, dict):
        context_length = backend.get("context_length")
        if isinstance(context_length, int) and context_length > 0:
            return context_length
    return 0


def _uses_dnagpt2_residuals(residual_codec: ResidualCodec) -> bool:
    return residual_codec.codec_id.startswith("dnagpt2")
