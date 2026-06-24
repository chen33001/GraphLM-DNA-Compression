"""Integer arithmetic coding primitives for independently decoded payloads.

Adapted from ``llm_and_compression/arithmetic_encoder/coder.py`` at
commit 2947047e905998d45eaab340bac108d95f14aebd (Apache-2.0).
Changes: split encoder/decoder state, accept CDFs per symbol, remove model
deep-copying, and expose packed-payload-compatible bit streams. The upstream
implementation states that it was adapted from Nayuki's reference arithmetic
coding implementation.
"""

from bisect import bisect_right
from collections import deque
from collections.abc import Sequence


def _validate_cdf(cdf: Sequence[int]) -> None:
    if not cdf or cdf[0] <= 0:
        raise ValueError("CDF must contain positive cumulative frequencies")
    if any(high <= low for low, high in zip(cdf, cdf[1:])):
        raise ValueError("CDF must be strictly increasing")


class ArithmeticEncoder:
    def __init__(self, state_bits: int = 64) -> None:
        if state_bits < 16:
            raise ValueError("state_bits must be at least 16")
        self.state_bits = state_bits
        self.maximum = 1 << state_bits
        self.mask = self.maximum - 1
        self.top_bit = self.maximum >> 1
        self.low = 0
        self.high = self.mask
        self.underflow = 0
        self.bits: list[int] = []

    def encode_symbol(self, symbol: int, cdf: Sequence[int]) -> None:
        _validate_cdf(cdf)
        if symbol < 0 or symbol >= len(cdf):
            raise ValueError("symbol is outside the CDF")
        symbol_low = 0 if symbol == 0 else cdf[symbol - 1]
        self._update(cdf[-1], symbol_low, cdf[symbol])

    def _update(self, total: int, symbol_low: int, symbol_high: int) -> None:
        current_range = self.high - self.low + 1
        previous_low = self.low
        self.low = previous_low + symbol_low * current_range // total
        self.high = previous_low + symbol_high * current_range // total - 1
        while ((self.low ^ self.high) & self.top_bit) == 0:
            top = self.low >> (self.state_bits - 1)
            self.bits.append(top)
            self.bits.extend([top ^ 1] * self.underflow)
            self.underflow = 0
            self.low = (self.low << 1) & self.mask
            self.high = ((self.high << 1) & self.mask) | 1
        quarter = self.maximum >> 2
        while (self.low & ~self.high & quarter) != 0:
            self.underflow += 1
            self.low = (self.low << 1) ^ self.top_bit
            self.high = ((self.high ^ self.top_bit) << 1) | self.top_bit | 1

    def finish(self) -> list[int]:
        return [*self.bits, 1]


class ArithmeticDecoder:
    def __init__(self, bits: Sequence[int], state_bits: int = 64) -> None:
        if state_bits < 16:
            raise ValueError("state_bits must be at least 16")
        if any(bit not in (0, 1) for bit in bits):
            raise ValueError("encoded stream must contain only 0 or 1")
        self.state_bits = state_bits
        self.maximum = 1 << state_bits
        self.mask = self.maximum - 1
        self.top_bit = self.maximum >> 1
        self.low = 0
        self.high = self.mask
        self.bits = deque(bits)
        self.code = 0
        for _ in range(state_bits):
            self.code = (self.code << 1) | self._read_bit()

    def decode_symbol(self, cdf: Sequence[int]) -> int:
        _validate_cdf(cdf)
        total = cdf[-1]
        current_range = self.high - self.low + 1
        target = ((self.code - self.low + 1) * total - 1) // current_range
        symbol = bisect_right(cdf, target)
        symbol_low = 0 if symbol == 0 else cdf[symbol - 1]
        self._update(total, symbol_low, cdf[symbol])
        return symbol

    def _update(self, total: int, symbol_low: int, symbol_high: int) -> None:
        current_range = self.high - self.low + 1
        previous_low = self.low
        self.low = previous_low + symbol_low * current_range // total
        self.high = previous_low + symbol_high * current_range // total - 1
        while ((self.low ^ self.high) & self.top_bit) == 0:
            self.code = ((self.code << 1) & self.mask) | self._read_bit()
            self.low = (self.low << 1) & self.mask
            self.high = ((self.high << 1) & self.mask) | 1
        quarter = self.maximum >> 2
        while (self.low & ~self.high & quarter) != 0:
            next_bit = self._read_bit()
            previous_code = self.code
            self.code &= self.maximum >> 1
            self.code |= ((previous_code << 1) & (self.mask >> 1)) | next_bit
            self.low = (self.low << 1) ^ self.top_bit
            self.high = ((self.high ^ self.top_bit) << 1) | self.top_bit | 1

    def _read_bit(self) -> int:
        return self.bits.popleft() if self.bits else 0
