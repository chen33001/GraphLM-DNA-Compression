from pathlib import Path

import pytest

from src.archive_v2 import BinaryArchive, read_binary_archive, write_binary_archive


def test_binary_archive_round_trip_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "sample.ghdna"
    archive = BinaryArchive(
        metadata={"magic": "GHDNA_2", "sha256": "abc"},
        payload=b"\x01\x02\xff",
        blocks=[],
    )

    write_binary_archive(archive, path)
    first_bytes = path.read_bytes()
    write_binary_archive(archive, path)

    assert path.read_bytes() == first_bytes
    restored = read_binary_archive(path)
    assert restored.payload == archive.payload
    assert restored.blocks == archive.blocks
    assert restored.metadata["magic"] == archive.metadata["magic"]
    assert restored.metadata["sha256"] == archive.metadata["sha256"]
    assert restored.metadata["block_table_format"] == "binary_v1"


@pytest.mark.parametrize(
    "content,error",
    [
        (b"OTHER!!", "magic"),
        (b"GHDNA2\x00\x00\x00", "truncated"),
        (b"GHDNA2\x00\x00\x00\x00\x10{}", "truncated"),
    ],
)
def test_binary_archive_rejects_invalid_or_truncated_files(
    tmp_path: Path, content: bytes, error: str
) -> None:
    path = tmp_path / "bad.ghdna"
    path.write_bytes(content)

    with pytest.raises(ValueError, match=error):
        read_binary_archive(path)
