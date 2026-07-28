import pytest
import torch
from omegaconf import OmegaConf

from mldft.ml.train import _configure_multiprocessing_sharing_strategy


def test_configure_multiprocessing_sharing_strategy(monkeypatch):
    selected = []
    monkeypatch.setattr(
        torch.multiprocessing,
        "get_all_sharing_strategies",
        lambda: {"file_descriptor", "file_system"},
    )
    monkeypatch.setattr(
        torch.multiprocessing,
        "set_sharing_strategy",
        selected.append,
    )

    _configure_multiprocessing_sharing_strategy(
        OmegaConf.create({"multiprocessing_sharing_strategy": "file_descriptor"})
    )

    assert selected == ["file_descriptor"]


def test_configure_multiprocessing_sharing_strategy_rejects_unknown(monkeypatch):
    monkeypatch.setattr(
        torch.multiprocessing,
        "get_all_sharing_strategies",
        lambda: {"file_descriptor", "file_system"},
    )

    with pytest.raises(ValueError, match="Unsupported multiprocessing sharing strategy"):
        _configure_multiprocessing_sharing_strategy(
            OmegaConf.create({"multiprocessing_sharing_strategy": "unknown"})
        )
