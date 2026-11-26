"""Jalankan seed 2/3 dan dua ablation ringan untuk catatan 122450130.

Arsitektur sama persis dengan models/122450130/run.py (1D-CNN 3→16→32,
BatchNorm1d, MaxPool, AdaptiveAvgPool, Dropout 0.2, Adam lr=3e-3 wd=1e-4,
50 epoch, batch 32). Menulis prediksi ke results/122450130/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))
from src.protokol import data

OUT = AKAR / "results" / "122450130"


class CNN1D(nn.Module):
    def __init__(self, use_bn: bool = True, dropout: float = 0.2):
        super().__init__()
        lapis: list[nn.Module] = [nn.Conv1d(3, 16, 3, padding=1)]
        if use_bn:
            lapis.append(nn.BatchNorm1d(16))
        lapis += [nn.ReLU(), nn.MaxPool1d(2),
                  nn.Conv1d(16, 32, 3, padding=1)]
        if use_bn:
            lapis.append(nn.BatchNorm1d(32))
        lapis += [nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten()]
        if dropout > 0:
            lapis.append(nn.Dropout(dropout))
        lapis.append(nn.Linear(32, 8))
        self.net = nn.Sequential(*lapis)

    def forward(self, x):
        return self.net(x)


def jalankan(label: str, seed: int, use_bn: bool = True, dropout: float = 0.2) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, ytr, Xva, yva, _, _ = data.muat()

    def ke_tensor(arr):
        return None if arr is None else torch.tensor(np.asarray(arr, np.float32)).permute(0, 2, 1)

    Xtr, Xva = ke_tensor(Xtr), ke_tensor(Xva)
    ytr = torch.tensor(np.asarray(ytr), dtype=torch.long)
    model = CNN1D(use_bn=use_bn, dropout=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    n = len(Xtr)
    model.train()
    for _ in range(50):
        perm = torch.randperm(n)
        for i in range(0, n, 32):
            ix = perm[i:i + 32]
            opt.zero_grad()
            ce(model(Xtr[ix]), ytr[ix]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(Xva).argmax(1).numpy()
    np.save(OUT / f"pred_val_{label}.npy", pred)
    print(f"{label}: acc={float((pred == np.asarray(yva)).mean()):.4f}", flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    jalankan("seed2", 2)
    jalankan("seed3", 3)
    jalankan("ab_no_bn", 1, use_bn=False)
    jalankan("ab_no_drop", 1, dropout=0.0)
    print("selesai ->", OUT)
