from src.llm_residual_codec import DNAGPT2ResidualCodec, TokenProbabilityBackend


class FakeDNABackend(TokenProbabilityBackend):
    alphabet = "ACGTN"

    def __init__(self, context_length: int = 1024) -> None:
        self.context_length = context_length

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "model": "fake-dna",
            "revision": "test",
            "context_length": self.context_length,
        }

    def tokenize(self, sequence: str) -> list[int]:
        return [self.alphabet.index(symbol) for symbol in sequence]

    def detokenize(self, token_ids: list[int]) -> str:
        return "".join(self.alphabet[token] for token in token_ids)

    def cdf_for_prefix(self, token_ids: list[int]) -> list[int]:
        weights = [1] * len(self.alphabet)
        if token_ids:
            weights[(token_ids[-1] + 1) % len(self.alphabet)] = 10
        total = 0
        cdf = []
        for weight in weights:
            total += weight
            cdf.append(total)
        return cdf


class FakePrecomputedDNABackend(FakeDNABackend):
    def __init__(self) -> None:
        super().__init__()
        self.prefix_calls = 0
        self.sequence_calls = 0

    def cdf_for_prefix(self, token_ids: list[int]) -> list[int]:
        self.prefix_calls += 1
        return super().cdf_for_prefix(token_ids)

    def cdfs_for_token_sequence(self, token_ids: list[int]) -> list[list[int]]:
        self.sequence_calls += 1
        return [super().cdf_for_prefix(token_ids[:index]) for index in range(len(token_ids))]


def test_dnagpt2_residual_codec_round_trip_with_independent_decoder() -> None:
    encoder_codec = DNAGPT2ResidualCodec(FakeDNABackend(), state_bits=32)
    encoded = encoder_codec.encode("ACGTNACGTACGT")

    decoder_codec = DNAGPT2ResidualCodec(FakeDNABackend(), state_bits=32)
    decoded = decoder_codec.decode(encoded)

    assert decoded == "ACGTNACGTACGT"
    assert encoded.symbol_count == 13
    assert encoded.metadata["backend"]["revision"] == "test"


def test_dnagpt2_residual_codec_handles_empty_block() -> None:
    codec = DNAGPT2ResidualCodec(FakeDNABackend(), state_bits=32)

    encoded = codec.encode("")

    assert encoded.payload == b""
    assert codec.decode(encoded) == ""


def test_dnagpt2_residual_codec_uses_exact_prefix_cdfs_during_encode() -> None:
    backend = FakePrecomputedDNABackend()
    codec = DNAGPT2ResidualCodec(backend, state_bits=32)

    encoded = codec.encode("ACGTNAC")

    assert encoded.symbol_count == 7
    assert backend.sequence_calls == 0
    assert backend.prefix_calls == 7


def test_dnagpt2_residual_codec_splits_long_sequence_into_safe_chunks() -> None:
    backend = FakeDNABackend(context_length=4)
    codec = DNAGPT2ResidualCodec(backend, state_bits=32)

    chunks = codec.split_sequence("ACGTNAC")

    assert chunks == ["ACGT", "NAC"]
