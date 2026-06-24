"""Pack arithmetic-coder bits into bytes and restore them exactly."""

from collections.abc import Iterable


def pack_bits(bits: Iterable[int]) -> tuple[bytes, int]:
    """Pack most-significant-bit-first binary values into bytes."""
    packed = bytearray()
    current = 0
    count = 0
    bit_length = 0
    for bit in bits:
        if bit not in (0, 1):
            raise ValueError("bits must contain only 0 or 1")
        current = (current << 1) | bit
        count += 1
        bit_length += 1
        if count == 8:
            packed.append(current)
            current = 0
            count = 0
    if count:
        packed.append(current << (8 - count))
    return bytes(packed), bit_length


def unpack_bits(payload: bytes, bit_length: int) -> list[int]:
    """Unpack exactly ``bit_length`` most-significant-bit-first values."""
    if bit_length < 0 or bit_length > len(payload) * 8:
        raise ValueError("bit length is outside the payload capacity")
    return [
        (payload[index // 8] >> (7 - index % 8)) & 1
        for index in range(bit_length)
    ]
