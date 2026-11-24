"""Model RNN / Bidirectional LSTM untuk Klasifikasi 8 Gestur Tangan Meja 5V.
NIM: 123450021 (Khoirul Muttoharoh - Tim 13)
Jalur: rnn

Penggunaan:
    python models/123450021/run.py --seed 1
    python models/123450021/run.py --seed 1 --final
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))
from src.protokol import data  # noqa: E402  (protokol tim yang dibekukan)

NIM = Path(__file__).resolve().parent.name     # nama folder = NIM (123450021)
JALUR = "rnn"                                   # cnn | tl | rnn | trf | ae


def latih_dan_prediksi(seed: int, final: bool) -> tuple[np.ndarray, np.ndarray | None]:
    """Melatih Bidirectional LSTM pada sekuens koordinat 21 titik sendi tangan."""
    import torch
    import torch.nn as nn

    # 1. Menetapkan seed untuk reproduksibilitas
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 2. Muat data melalui protokol tim
    Xtr, ytr, Xva, _, Xte, _ = data.muat()

    # Input X: [N, 21, 3] -> Sekuens panjang 21 titik, tiap titik memiliki 3 fitur koordinat (x, y, z)
    def ke_tensor(arr: np.ndarray | None) -> torch.Tensor | None:
        if arr is None:
            return None
        return torch.tensor(np.asarray(arr, dtype=np.float32))

    X_train = ke_tensor(Xtr)
    y_train = torch.tensor(np.asarray(ytr), dtype=torch.long)
    X_val = ke_tensor(Xva)
    X_test = ke_tensor(Xte) if final and Xte is not None else None

    # 3. Arsitektur Model: Bidirectional LSTM
    class BiLSTMGesture(nn.Module):
        def __init__(self, input_size: int = 3, hidden_size: int = 32, num_classes: int = 8,
                     num_layers: int = 2, dropout: float = 0.2):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            # Karena bidirectional, representasi laten berdimensi hidden_size * 2
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size * 2, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: [Batch, 21, 3]
            out, (hn, cn) = self.lstm(x)
            # Global mean pooling sepanjang 21 titik sendi tangan
            out_pool = out.mean(dim=1)
            out_pool = self.dropout(out_pool)
            return self.fc(out_pool)

    model = BiLSTMGesture(input_size=3, hidden_size=32, num_classes=8, num_layers=2, dropout=0.2)

    # 4. Optimizer & Loss Function
    opt = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 5. Training Loop
    batch_size = 32
    n_samples = len(X_train)
    epochs = 50

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_size):
            b_idx = perm[i : i + batch_size]
            bx, by = X_train[b_idx], y_train[b_idx]

            opt.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            opt.step()

    # 6. Evaluasi / Prediksi
    model.eval()
    with torch.no_grad():
        pred_val = model(X_val).argmax(dim=1).numpy()
        pred_test = None
        if final and X_test is not None:
            pred_test = model(X_test).argmax(dim=1).numpy()

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

    perintah = [
        sys.executable,
        "-m",
        "src.common.evaluate",
        "--nim",
        NIM,
        "--model",
        JALUR,
        "--seed",
        str(a.seed),
        "--pred-val",
        str(out / "pred_val.npy"),
        "--durasi",
        f"{durasi:.1f}",
        "--perangkat",
        a.perangkat,
    ]
    if pred_test is not None:
        np.save(out / "pred_test.npy", np.asarray(pred_test))
        perintah += ["--pred-test", str(out / "pred_test.npy")]

    subprocess.run(perintah, check=True, cwd=AKAR, env=dict(os.environ, PYTHONIOENCODING="utf-8"))


if __name__ == "__main__":
    main()
