"""Tiny dataset registry: name -> loader, shared by the experiment CLIs.

This is a lightweight dispatch table, not a plugin system: adding a
dataset means adding a loader module under ``hybridsearch/data/`` and one
entry here. Importing this module is offline-safe because the loaders
themselves only import ``datasets`` (and touch the network) when called.
"""

from __future__ import annotations

from typing import Callable

from hybridsearch.data.common import Dataset
from hybridsearch.data.fiqa import load_fiqa
from hybridsearch.data.scifact import load_scifact

DATASET_LOADERS: dict[str, Callable[[], Dataset]] = {
    "scifact": load_scifact,
    "fiqa": load_fiqa,
}

DATASET_NAMES: tuple[str, ...] = tuple(DATASET_LOADERS)


def load_dataset_by_name(name: str) -> Dataset:
    """Load a dataset by its registered name (``"scifact"`` or ``"fiqa"``).

    Unknown names raise ``ValueError`` before any network access.
    """
    loader = DATASET_LOADERS.get(name)
    if loader is None:
        raise ValueError(
            f"unknown dataset {name!r}; supported datasets: "
            f"{', '.join(DATASET_NAMES)}"
        )
    return loader()