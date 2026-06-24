"""Model-independent contract for encoding non-graph DNA blocks."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EncodedResidual:
    payload: bytes
    bit_length: int
    symbol_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.bit_length < 0 or self.bit_length > len(self.payload) * 8:
            raise ValueError("bit_length is outside the payload capacity")
        if self.symbol_count < 0:
            raise ValueError("symbol_count must not be negative")


class ResidualCodec(ABC):
    """Encode and decode one independently framed residual DNA block."""

    @property
    @abstractmethod
    def codec_id(self) -> str:
        pass

    @abstractmethod
    def encode(self, sequence: str) -> EncodedResidual:
        pass

    @abstractmethod
    def decode(self, encoded: EncodedResidual) -> str:
        pass

    def encode_with_context(self, sequence: str, context: str = "") -> EncodedResidual:
        """Encode one block with optional preceding sequence context."""
        return self.encode(sequence)

    def decode_with_context(self, encoded: EncodedResidual, context: str = "") -> str:
        """Decode one block with optional preceding sequence context."""
        return self.decode(encoded)

    def split_sequence(self, sequence: str) -> list[str]:
        """Split a residual into independently encodable chunks."""
        return [sequence] if sequence else []

    def archive_metadata(self) -> dict[str, Any]:
        """Archive-level metadata stored once for all residual blocks."""
        return {}

    def block_metadata(self, encoded: EncodedResidual) -> dict[str, Any]:
        """Per-block metadata required for decoding one residual block."""
        return encoded.metadata

    def decode_metadata(
        self,
        archive_metadata: dict[str, Any],
        block_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Rebuild encoded metadata from archive-level and block-level fields."""
        if archive_metadata and block_metadata:
            return {**archive_metadata, **block_metadata}
        if archive_metadata:
            return dict(archive_metadata)
        return dict(block_metadata)


class PlainResidualCodec(ResidualCodec):
    """Temporary uncompressed residual framing used to validate GHDNA_2."""

    codec_id = "plain_ascii_v1"

    def encode(self, sequence: str) -> EncodedResidual:
        if set(sequence) - set("ACGTN"):
            raise ValueError("plain residual contains unsupported DNA symbols")
        payload = sequence.encode("ascii")
        return EncodedResidual(
            payload=payload,
            bit_length=len(payload) * 8,
            symbol_count=len(sequence),
        )

    def decode(self, encoded: EncodedResidual) -> str:
        if encoded.bit_length != len(encoded.payload) * 8:
            raise ValueError("plain residual must use complete bytes")
        sequence = encoded.payload.decode("ascii")
        if len(sequence) != encoded.symbol_count:
            raise ValueError("plain residual symbol count mismatch")
        if set(sequence) - set("ACGTN"):
            raise ValueError("plain residual contains unsupported DNA symbols")
        return sequence
