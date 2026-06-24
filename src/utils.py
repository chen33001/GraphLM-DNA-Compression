"""Shared utilities."""

from hashlib import sha256


def sequence_checksum(sequence: str) -> str:
    return sha256(sequence.encode("ascii")).hexdigest()
