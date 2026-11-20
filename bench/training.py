"""Atomic checkpoint writes."""
from __future__ import annotations
import os
from pathlib import Path
import torch


def atomic_torch_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


