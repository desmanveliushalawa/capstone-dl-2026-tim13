"""Kerangka run.py anggota. Salin folder ini menjadi models/<nim>/ lalu isi bagian MODEL.

    python models/<nim>/run.py --seed 1 [--final]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))
from src.protokol import data  # noqa: E402  (protokol tim yang dibekukan)

NIM = Path(__file__).resolve().parent.name     # nama folder = NIM
JALUR = "tl"                                   # cnn | tl | rnn | trf | ae



def latih_dan_prediksi(seed: int, final: bool) -> tuple[np.ndarray, np.ndarray | None]:
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, ytr, Xva, _, Xte, _ = data.muat()

    # Standarisasi per koordinat memakai statistik data latih saja (sesuai PROTOKOL.md)
    Xtr_np = np.asarray(Xtr, dtype=np.float32)
    Xva_np = np.asarray(Xva, dtype=np.float32)
    mu = Xtr_np.mean(axis=0, keepdims=True)
    sd = Xtr_np.std(axis=0, keepdims=True) + 1e-6
    Xtr = (Xtr_np - mu) / sd
    Xva = (Xva_np - mu) / sd
    if Xte is not None:
        Xte = (np.asarray(Xte, dtype=np.float32) - mu) / sd

    def ke_tensor(arr):
        if arr is None:
            return None
        return torch.tensor(
            np.asarray(arr, dtype=np.float32)
        ).reshape(len(arr), -1)

    X_train = ke_tensor(Xtr)
    y_train = torch.tensor(np.asarray(ytr), dtype=torch.long)
    X_val = ke_tensor(Xva)
    X_test = ke_tensor(Xte) if final and Xte is not None else None

    class TransferMLPGesture(nn.Module):
        def __init__(self, in_features=63, num_classes=8):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(in_features, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
            )
            self.classifier = nn.Linear(64, num_classes)

        def forward(self, x):
            feat = self.backbone(x)
            return self.classifier(feat)

    model = TransferMLPGesture(in_features=63, num_classes=8)
    opt = torch.optim.AdamW(
        model.parameters(), lr=0.002, weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()

    batch_size = 32
    n_samples = len(X_train)
    model.train()

    for epoch in range(50):
        perm = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            bx, by = X_train[idx], y_train[idx]

            opt.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pred_val = model(X_val).argmax(1).numpy()
        pred_test = (
            model(X_test).argmax(1).numpy()
            if final and X_test is not None else None
        )

    return pred_val, pred_test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--final", action="store_true", help="sprint terakhir: evaluasi data uji sekali")
    ap.add_argument("--perangkat", default="cpu laptop")
    a = ap.parse_args()
    t0 = time.time()
    pred_val, pred_test = latih_dan_prediksi(a.seed, a.final)
    durasi = time.time() - t0
    out = AKAR / "results" / NIM
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_val.npy", np.asarray(pred_val))
    perintah = [sys.executable, "-m", "src.common.evaluate", "--nim", NIM, "--model", JALUR, "--seed", str(a.seed),
                "--pred-val", str(out / "pred_val.npy"), "--durasi", f"{durasi:.1f}", "--perangkat", a.perangkat]
    if pred_test is not None:
        np.save(out / "pred_test.npy", np.asarray(pred_test))
        perintah += ["--pred-test", str(out / "pred_test.npy")]
    subprocess.run(perintah, check=True, cwd=AKAR)


if __name__ == "__main__":
    main()
