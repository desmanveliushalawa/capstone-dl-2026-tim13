"""Kerangka run.py anggota. Salin folder ini menjadi models/<nim>/ lalu isi bagian MODEL.

    python models/122450130/run.py --seed 1 [--final]
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
JALUR = "cnn"                                   # cnn | tl | rnn | trf | ae

 # Input X: [N, 21, 3] -> Ubah ke format Channel First untuk Conv1D: [N, 3, 21]
def latih_dan_prediksi(seed: int, final: bool) -> tuple[np.ndarray, np.ndarray | None]:
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, ytr, Xva, _, Xte, _ = data.muat()

       # Input X: [N, 21, 3] -> Ubah ke format Channel First untuk Conv1D: [N, 3, 21]
    def ke_tensor(arr):
        if arr is None: return None
        t = torch.tensor(np.asarray(arr, dtype=np.float32))
        return t.permute(0, 2, 1) # [N, 3, 21]

    X_train = ke_tensor(Xtr)
    y_train = torch.tensor(np.asarray(ytr), dtype=torch.long)
    X_val = ke_tensor(Xva)
    X_test = ke_tensor(Xte) if final and Xte is not None else None

    class CNN1DGesture(nn.Module):
        def __init__(self, num_classes=8):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Dropout(0.2),
                nn.Linear(32, num_classes)
            )
        def forward(self, x):
            return self.net(x)

    model = CNN1DGesture(num_classes=8)
    opt = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    batch_size = 32
    n_samples = len(X_train)
    model.train()
    for epoch in range(50):
        perm = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i+batch_size]
            bx, by = X_train[idx], y_train[idx]
            opt.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pred_val = model(X_val).argmax(1).numpy()
        pred_test = model(X_test).argmax(1).numpy() if final and X_test is not None else None

    return pred_val, pred_test



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    ap.add_argument("--final", action="store_true", help="sprint terakhir: evaluasi data uji sekali")
    ap.add_argument("--perangkat", default="cpu laptop")
    a = ap.parse_args()
    t0 = time.time()
    pred_val, pred_test = latih_dan_prediksi(a.seed, a.final)
    durasi = time.time() - t0
    out = AKAR / "results" / '122450130'
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
