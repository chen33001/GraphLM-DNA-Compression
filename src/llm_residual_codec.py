"""Arithmetic-coded residual blocks driven by a token probability backend."""

from abc import ABC, abstractmethod

from src.arithmetic_coder import ArithmeticDecoder, ArithmeticEncoder
from src.bitpacking import pack_bits, unpack_bits
from src.residual_codec import EncodedResidual, ResidualCodec


class TokenProbabilityBackend(ABC):
    """Minimal boundary between deterministic token probabilities and coding."""

    @property
    @abstractmethod
    def metadata(self) -> dict[str, object]:
        pass

    @abstractmethod
    def tokenize(self, sequence: str) -> list[int]:
        pass

    @abstractmethod
    def detokenize(self, token_ids: list[int]) -> str:
        pass

    @abstractmethod
    def cdf_for_prefix(self, token_ids: list[int]) -> list[int]:
        """Return a strictly increasing integer CDF for the next token."""

    def cdfs_for_token_sequence(self, token_ids: list[int]) -> list[list[int]]:
        """Optionally precompute all next-token CDFs for one encoded sequence."""
        return [self.cdf_for_prefix(token_ids[:index]) for index in range(len(token_ids))]


class DNAGPT2ResidualCodec(ResidualCodec):
    codec_id = "dnagpt2_arithmetic_v1"

    def __init__(self, backend: TokenProbabilityBackend, state_bits: int = 64) -> None:
        self.backend = backend
        self.state_bits = state_bits

    def encode(self, sequence: str) -> EncodedResidual:
        return self.encode_with_context(sequence)

    def encode_with_context(self, sequence: str, context: str = "") -> EncodedResidual:
        prefix_token_ids, token_ids, use_context = self._tokenize_with_context(sequence, context)
        return self._encode_token_ids(token_ids, prefix_token_ids, use_context)

    def _encode_token_ids(
        self,
        token_ids: list[int],
        prefix_token_ids: list[int],
        use_context: bool,
    ) -> EncodedResidual:
        if not token_ids:
            return EncodedResidual(
                payload=b"",
                bit_length=0,
                symbol_count=0,
                metadata=self._metadata(use_context=use_context),
            )
        encoder = ArithmeticEncoder(self.state_bits)
        prefix = list(prefix_token_ids)
        for token in token_ids:
            cdf = self.backend.cdf_for_prefix(prefix)
            if token < 0 or token >= len(cdf):
                raise ValueError("token is outside the backend vocabulary")
            encoder.encode_symbol(token, cdf)
            prefix.append(token)
        payload, bit_length = pack_bits(encoder.finish())
        return EncodedResidual(
            payload=payload,
            bit_length=bit_length,
            symbol_count=len(token_ids),
            metadata=self._metadata(use_context=use_context),
        )

    def split_sequence(self, sequence: str) -> list[str]:
        if not sequence:
            return []
        max_tokens = int(self.backend.metadata.get("context_length", 0))
        if max_tokens <= 0:
            return [sequence]
        token_count_cache: dict[str, int] = {}

        def token_count(fragment: str) -> int:
            cached = token_count_cache.get(fragment)
            if cached is not None:
                return cached
            cached = len(self.backend.tokenize(fragment))
            token_count_cache[fragment] = cached
            return cached

        if token_count(sequence) <= max_tokens:
            return [sequence]

        chunks: list[str] = []
        start = 0
        while start < len(sequence):
            low = start + 1
            high = len(sequence)
            best_end = None
            while low <= high:
                mid = (low + high) // 2
                current_token_count = token_count(sequence[start:mid])
                if current_token_count <= max_tokens:
                    best_end = mid
                    low = mid + 1
                else:
                    high = mid - 1
            if best_end is None:
                raise ValueError("cannot split residual into context-safe DNAGPT2 chunks")
            chunks.append(sequence[start:best_end])
            start = best_end
        return chunks

    def decode(self, encoded: EncodedResidual) -> str:
        return self.decode_with_context(encoded)

    def decode_with_context(self, encoded: EncodedResidual, context: str = "") -> str:
        use_context = bool(encoded.metadata.get("use_context"))
        if encoded.metadata != self._metadata(use_context=use_context):
            raise ValueError("residual codec metadata does not match the decoder")
        if encoded.symbol_count == 0:
            if encoded.bit_length or encoded.payload:
                raise ValueError("empty residual must not contain a payload")
            return ""
        prefix_token_ids = self.backend.tokenize(context) if use_context and context else []
        decoder = ArithmeticDecoder(
            unpack_bits(encoded.payload, encoded.bit_length),
            self.state_bits,
        )
        token_ids = list(prefix_token_ids)
        decoded_token_ids: list[int] = []
        for _ in range(encoded.symbol_count):
            cdf = self.backend.cdf_for_prefix(token_ids)
            token = decoder.decode_symbol(cdf)
            token_ids.append(token)
            decoded_token_ids.append(token)
        return self.backend.detokenize(decoded_token_ids)

    def archive_metadata(self) -> dict[str, object]:
        return {
            "codec_id": self.codec_id,
            "state_bits": self.state_bits,
            "backend": self.backend.metadata,
        }

    def block_metadata(self, encoded: EncodedResidual) -> dict[str, object]:
        return {"use_context": bool(encoded.metadata.get("use_context"))}

    def decode_metadata(
        self,
        archive_metadata: dict[str, object],
        block_metadata: dict[str, object],
    ) -> dict[str, object]:
        metadata = dict(archive_metadata)
        metadata["use_context"] = bool(block_metadata.get("use_context"))
        return metadata

    def _metadata(self, *, use_context: bool) -> dict[str, object]:
        metadata = dict(self.archive_metadata())
        metadata["use_context"] = use_context
        return metadata

    def _tokenize_with_context(
        self, sequence: str, context: str
    ) -> tuple[list[int], list[int], bool]:
        target_token_ids = self.backend.tokenize(sequence)
        if not context:
            return [], target_token_ids, False

        prefix_token_ids = self.backend.tokenize(context)
        combined_token_ids = self.backend.tokenize(context + sequence)
        prefix_length = len(prefix_token_ids)
        if prefix_length > len(combined_token_ids):
            return [], target_token_ids, False
        if combined_token_ids[:prefix_length] != prefix_token_ids:
            return [], target_token_ids, False
        combined_suffix = combined_token_ids[prefix_length:]
        if self.backend.detokenize(combined_suffix) != sequence:
            return [], target_token_ids, False
        return prefix_token_ids, combined_suffix, True
