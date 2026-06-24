from src.arithmetic_coder import ArithmeticDecoder, ArithmeticEncoder
from src.bitpacking import pack_bits, unpack_bits


def test_arithmetic_coder_round_trip_with_position_specific_cdfs() -> None:
    symbols = [0, 1, 3, 2, 0, 3, 3, 1]
    cdfs = [
        [1, 2, 3, 4],
        [5, 7, 8, 10],
        [1, 3, 7, 12],
        [2, 3, 9, 10],
        [4, 5, 6, 7],
        [1, 2, 3, 20],
        [1, 2, 3, 20],
        [1, 8, 9, 10],
    ]
    encoder = ArithmeticEncoder(state_bits=32)
    for symbol, cdf in zip(symbols, cdfs, strict=True):
        encoder.encode_symbol(symbol, cdf)

    payload, bit_length = pack_bits(encoder.finish())
    decoder = ArithmeticDecoder(unpack_bits(payload, bit_length), state_bits=32)

    assert [decoder.decode_symbol(cdf) for cdf in cdfs] == symbols


def test_arithmetic_coder_handles_single_symbol_alphabet() -> None:
    encoder = ArithmeticEncoder(state_bits=32)
    for _ in range(5):
        encoder.encode_symbol(0, [1])

    bits = encoder.finish()
    decoder = ArithmeticDecoder(bits, state_bits=32)

    assert [decoder.decode_symbol([1]) for _ in range(5)] == [0] * 5
