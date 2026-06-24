import pytest

from src.bitpacking import pack_bits, unpack_bits
from src.residual_codec import EncodedResidual, PlainResidualCodec, ResidualCodec


def test_bit_packing_round_trip_preserves_non_byte_aligned_bits() -> None:
    bits = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1]

    payload, bit_length = pack_bits(bits)

    assert payload == bytes([0b10110010, 0b11100000])
    assert bit_length == len(bits)
    assert unpack_bits(payload, bit_length) == bits


def test_bit_packing_rejects_non_binary_values() -> None:
    with pytest.raises(ValueError, match="0 or 1"):
        pack_bits([0, 1, 2])


def test_bit_unpacking_rejects_impossible_length() -> None:
    with pytest.raises(ValueError, match="bit length"):
        unpack_bits(b"\x00", 9)


def test_residual_codec_is_abstract() -> None:
    with pytest.raises(TypeError):
        ResidualCodec()


def test_encoded_residual_validates_payload_metadata() -> None:
    with pytest.raises(ValueError, match="bit_length"):
        EncodedResidual(payload=b"\x00", bit_length=9, symbol_count=1)


def test_plain_residual_codec_round_trip() -> None:
    codec = PlainResidualCodec()

    encoded = codec.encode("ACGTN")

    assert codec.codec_id == "plain_ascii_v1"
    assert encoded.payload == b"ACGTN"
    assert codec.decode(encoded) == "ACGTN"
