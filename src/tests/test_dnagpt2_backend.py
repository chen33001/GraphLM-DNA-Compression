from pathlib import Path

from src.dnagpt2_backend import HuggingFaceDNAGPT2Backend, trim_past_key_values


class FakeCacheTensor:
    def __init__(self, name: str, seq_len: int) -> None:
        self.name = name
        self.seq_len = seq_len

    def __getitem__(self, item):
        _, _, seq_slice, _ = item
        if not isinstance(seq_slice, slice):
            raise TypeError("expected a slice on the sequence axis")
        start = 0 if seq_slice.start is None else seq_slice.start
        if start >= 0:
            raise AssertionError("expected negative slicing from the tail")
        return FakeCacheTensor(self.name, -start)


def test_trim_past_key_values_keeps_only_last_positions() -> None:
    past = (
        (FakeCacheTensor("k1", 10), FakeCacheTensor("v1", 10)),
        (FakeCacheTensor("k2", 10), FakeCacheTensor("v2", 10)),
    )

    trimmed = trim_past_key_values(past, 4)

    assert trimmed is not None
    assert [tensor.seq_len for pair in trimmed for tensor in pair] == [4, 4, 4, 4]


def test_resolve_model_source_prefers_complete_local_snapshot(
    tmp_path: Path, monkeypatch
) -> None:
    snapshot_path = (
        tmp_path
        / ".hf-cache"
        / "hub"
        / "models--vojtam--DNAGPT2_32"
        / "snapshots"
        / "rev123"
    )
    snapshot_path.mkdir(parents=True)
    for filename in ("config.json", "model.safetensors", "tokenizer.json"):
        (snapshot_path / filename).write_text("x", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    backend = HuggingFaceDNAGPT2Backend.__new__(HuggingFaceDNAGPT2Backend)
    backend.model_repository = "vojtam/DNAGPT2_32"
    backend.revision = "rev123"

    assert backend._resolve_model_source().resolve() == snapshot_path.resolve()


def test_resolve_model_source_prefers_default_model_directory(
    tmp_path: Path, monkeypatch
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir(parents=True)
    for filename in ("config.json", "model.safetensors", "tokenizer.json"):
        (model_path / filename).write_text("x", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    backend = HuggingFaceDNAGPT2Backend.__new__(HuggingFaceDNAGPT2Backend)
    backend.model_repository = "vojtam/DNAGPT2_32"
    backend.revision = "rev123"

    assert backend._resolve_model_source().resolve() == model_path.resolve()


def test_resolve_model_source_prefers_explicit_local_model_directory(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "custom_model"
    model_path.mkdir(parents=True)
    for filename in ("config.json", "model.safetensors", "tokenizer.json"):
        (model_path / filename).write_text("x", encoding="utf-8")

    backend = HuggingFaceDNAGPT2Backend.__new__(HuggingFaceDNAGPT2Backend)
    backend.model_repository = str(model_path)
    backend.revision = "rev123"

    assert backend._resolve_model_source().resolve() == model_path.resolve()


def test_resolve_model_source_falls_back_to_repository_name(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    backend = HuggingFaceDNAGPT2Backend.__new__(HuggingFaceDNAGPT2Backend)
    backend.model_repository = "vojtam/DNAGPT2_32"
    backend.revision = "missing"

    assert backend._resolve_model_source() == "vojtam/DNAGPT2_32"
