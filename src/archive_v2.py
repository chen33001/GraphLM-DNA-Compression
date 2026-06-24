"""Deterministic binary container for GHDNA_2 metadata and payload bytes."""

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BINARY_MAGIC = b"GHDNA2\x00"
MAX_METADATA_BYTES = 64 * 1024 * 1024
_LENGTH_SIZE = 4
_GRAPH_COPY_TYPE = 1
_LLM_RESIDUAL_TYPE = 2
_RAW_RESIDUAL_TYPE = 3
_GRAPH_COPY_STRUCT = struct.Struct(">BIII")
_LLM_RESIDUAL_STRUCT = struct.Struct(">BBIIIIII")
_RAW_RESIDUAL_STRUCT = struct.Struct(">BIIII")


@dataclass(frozen=True)
class BinaryArchive:
    metadata: dict[str, Any]
    payload: bytes
    blocks: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.metadata.get("magic") != "GHDNA_2":
            raise ValueError("invalid archive metadata magic")


def write_binary_archive(archive: BinaryArchive, path: str | Path) -> None:
    metadata = dict(archive.metadata)
    blocks = archive.blocks or metadata.pop("blocks", [])
    block_bytes = _encode_blocks(blocks)
    metadata.pop("blocks", None)
    metadata["block_table_bytes"] = len(block_bytes)
    metadata["block_table_format"] = "binary_v1"
    metadata["block_table_count"] = len(blocks)
    metadata_bytes = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    if len(metadata_bytes) > MAX_METADATA_BYTES:
        raise ValueError("archive metadata is too large")
    Path(path).write_bytes(
        BINARY_MAGIC
        + struct.pack(">I", len(metadata_bytes))
        + metadata_bytes
        + block_bytes
        + archive.payload
    )


def read_binary_archive(path: str | Path) -> BinaryArchive:
    content = Path(path).read_bytes()
    if not content.startswith(BINARY_MAGIC):
        raise ValueError("invalid binary archive magic")
    header_end = len(BINARY_MAGIC) + _LENGTH_SIZE
    if len(content) < header_end:
        raise ValueError("truncated binary archive header")
    metadata_length = struct.unpack(">I", content[len(BINARY_MAGIC) : header_end])[0]
    if metadata_length > MAX_METADATA_BYTES:
        raise ValueError("archive metadata is too large")
    metadata_end = header_end + metadata_length
    if len(content) < metadata_end:
        raise ValueError("truncated binary archive metadata")
    try:
        metadata = json.loads(content[header_end:metadata_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid binary archive metadata") from error
    if not isinstance(metadata, dict):
        raise ValueError("binary archive metadata must be an object")
    if metadata.get("block_table_format") == "binary_v1":
        block_table_bytes = metadata.get("block_table_bytes")
        if not isinstance(block_table_bytes, int) or block_table_bytes < 0:
            raise ValueError("invalid binary archive block table length")
        blocks_end = metadata_end + block_table_bytes
        if len(content) < blocks_end:
            raise ValueError("truncated binary archive block table")
        blocks = _decode_blocks(content[metadata_end:blocks_end])
        payload = content[blocks_end:]
        return BinaryArchive(metadata=metadata, payload=payload, blocks=blocks)

    blocks = metadata.get("blocks", [])
    return BinaryArchive(metadata=metadata, payload=content[metadata_end:], blocks=blocks)


def _encode_blocks(blocks: list[dict[str, Any]]) -> bytes:
    encoded = bytearray()
    for block in blocks:
        block_type = block.get("type")
        if block_type == "graph_copy":
            encoded.extend(
                _GRAPH_COPY_STRUCT.pack(
                    _GRAPH_COPY_TYPE,
                    int(block["start"]),
                    int(block["source"]),
                    int(block["length"]),
                )
            )
            continue
        if block_type == "llm_residual":
            flags = 1 if block.get("use_context") else 0
            encoded.extend(
                _LLM_RESIDUAL_STRUCT.pack(
                    _LLM_RESIDUAL_TYPE,
                    flags,
                    int(block["start"]),
                    int(block["length"]),
                    int(block["symbol_count"]),
                    int(block["bit_length"]),
                    int(block["payload_offset"]),
                    int(block["payload_length"]),
                )
            )
            continue
        if block_type == "raw_residual":
            encoded.extend(
                _RAW_RESIDUAL_STRUCT.pack(
                    _RAW_RESIDUAL_TYPE,
                    int(block["start"]),
                    int(block["length"]),
                    int(block["payload_offset"]),
                    int(block["payload_length"]),
                )
            )
            continue
        raise ValueError(f"unknown binary archive block type: {block_type}")
    return bytes(encoded)


def _decode_blocks(content: bytes) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    offset = 0
    while offset < len(content):
        block_type = content[offset]
        if block_type == _GRAPH_COPY_TYPE:
            end = offset + _GRAPH_COPY_STRUCT.size
            if end > len(content):
                raise ValueError("truncated graph-copy block")
            _, start, source, length = _GRAPH_COPY_STRUCT.unpack(content[offset:end])
            blocks.append(
                {"type": "graph_copy", "start": start, "source": source, "length": length}
            )
            offset = end
            continue
        if block_type == _LLM_RESIDUAL_TYPE:
            end = offset + _LLM_RESIDUAL_STRUCT.size
            if end > len(content):
                raise ValueError("truncated residual block")
            (
                _,
                flags,
                start,
                length,
                symbol_count,
                bit_length,
                payload_offset,
                payload_length,
            ) = _LLM_RESIDUAL_STRUCT.unpack(content[offset:end])
            block = {
                "type": "llm_residual",
                "start": start,
                "length": length,
                "symbol_count": symbol_count,
                "bit_length": bit_length,
                "payload_offset": payload_offset,
                "payload_length": payload_length,
            }
            if flags & 1:
                block["use_context"] = True
            blocks.append(block)
            offset = end
            continue
        if block_type == _RAW_RESIDUAL_TYPE:
            end = offset + _RAW_RESIDUAL_STRUCT.size
            if end > len(content):
                raise ValueError("truncated raw residual block")
            _, start, length, payload_offset, payload_length = _RAW_RESIDUAL_STRUCT.unpack(
                content[offset:end]
            )
            blocks.append(
                {
                    "type": "raw_residual",
                    "start": start,
                    "length": length,
                    "payload_offset": payload_offset,
                    "payload_length": payload_length,
                }
            )
            offset = end
            continue
        raise ValueError("unknown binary archive block type")
    return blocks
