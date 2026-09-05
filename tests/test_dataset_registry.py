"""Offline tests for the dataset registry / dispatch layer (Phase 13).

The registered loaders are monkeypatched with fakes, so no network access
and no ``datasets`` import ever happen.
"""

import pytest

from hybridsearch.data.common import Dataset
from hybridsearch.data.registry import (
    DATASET_LOADERS,
    DATASET_NAMES,
    load_dataset_by_name,
)


def make_fake(name):
    return Dataset(
        corpus=[{"id": "1", "text": "a"}],
        queries={"q1": "x"},
        qrels={"q1": {"1": 1}},
        name=name,
        source=f"fake/{name}",
    )


def test_registered_names():
    assert DATASET_NAMES == ("scifact", "fiqa")
    assert set(DATASET_LOADERS) == set(DATASET_NAMES)


def test_dispatch_scifact(monkeypatch):
    fake = make_fake("scifact")
    monkeypatch.setitem(DATASET_LOADERS, "scifact", lambda: fake)

    assert load_dataset_by_name("scifact") is fake


def test_dispatch_fiqa(monkeypatch):
    fake = make_fake("fiqa")
    monkeypatch.setitem(DATASET_LOADERS, "fiqa", lambda: fake)

    assert load_dataset_by_name("fiqa") is fake


def test_unknown_dataset_raises_before_any_loader_runs(monkeypatch):
    calls = []

    def loader():
        calls.append(1)
        return None

    for name in DATASET_NAMES:
        monkeypatch.setitem(DATASET_LOADERS, name, loader)

    with pytest.raises(ValueError, match="unknown dataset"):
        load_dataset_by_name("nope")

    assert calls == []