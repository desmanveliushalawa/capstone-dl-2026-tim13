"""Transformer encoder kecil untuk klasifikasi 8 gestur tangan (Tim 13, jalur trf).

    python models/123450032/run.py --seed 1 [--final]

Masukan  : X [N, 21, 3]  -> 21 titik MediaPipe (x, y, z) sebagai 21 token, tiap token 3 fitur.
Keluaran : logit [N, 8]  -> argmax = label gestur 0..7 (klasifikasi, lihat src/protokol/PROTOKOL.md).

Potongan tangan (citra) belum ada di src.protokol.data.muat() (protokol beku, hanya landmark),
jadi model ini baru memakai token landmark. Potongan citra untuk MobileNet ada di jalur TL.
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

NIM = Path(__file__).resolve().parent.name     # nama folder = NIM
JALUR = "trf"                                   # cnn | tl | rnn | trf | ae

N_TITIK, N_FITUR, N_KELAS = 21, 3, 8
HP = dict(d_model=32, nhead=4, layers=2, ff=64, dropout=0.1, lr=2e-3, wd=1e-2, epoch=50, batch=32)


def standarisasi(Xtr: np.ndarray, *lain: np.ndarray | None):
    """Standarisasi per koordinat (21x3) dengan statistik data latih saja (praproses protokol)."""
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + 1e-6
    return [(None if X is None else ((X - mu) / sd).astype(np.float32)) for X in (Xtr, *lain)]


def latih_dan_prediksi(seed: int, final: bool) -> tuple[np.ndarray, np.ndarray | None]:
    """Kembalikan (pred_val, pred_test|None)."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, ytr, Xva, _, Xte, _ = data.muat()
    assert Xtr.shape[1:] == (N_TITIK, N_FITUR), f"masukan harus [N, 21, 3], dapat {Xtr.shape}"
    Xtr, Xva, Xte = standarisasi(np.asarray(Xtr, np.float32), np.asarray(Xva, np.float32),
                                 None if Xte is None else np.asarray(Xte, np.float32))
    Xtr, Xva = torch.from_numpy(Xtr), torch.from_numpy(Xva)
    ytr = torch.as_tensor(np.asarray(ytr), dtype=torch.long)

    class TransformerGestur(nn.Module):
        def __init__(self, d, nhead, layers, ff, dropout):
            super().__init__()
            self.embed = nn.Linear(N_FITUR, d)                                # token = satu titik
            self.pos = nn.Parameter(torch.zeros(1, N_TITIK + 1, d))           # posisi (identitas sendi) + CLS
            nn.init.normal_(self.pos, std=0.02)
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
            lap = nn.TransformerEncoderLayer(d, nhead, ff, dropout, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(lap, layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d)
            self.kepala = nn.Linear(d, N_KELAS)

        def forward(self, x):                                                 # x: [B, 21, 3]
            h = torch.cat([self.cls.expand(len(x), -1, -1), self.embed(x)], dim=1) + self.pos
            return self.kepala(self.norm(self.enc(h)[:, 0]))                  # logit [B, 8]

    m = TransformerGestur(HP["d_model"], HP["nhead"], HP["layers"], HP["ff"], HP["dropout"])
    # AdamW: adaptif per parameter (Transformer peka terhadap skala gradien) + weight decay terpisah
    opt = torch.optim.AdamW(m.parameters(), lr=HP["lr"], weight_decay=HP["wd"])
    langkah = HP["epoch"] * int(np.ceil(len(Xtr) / HP["batch"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=HP["lr"], total_steps=langkah, pct_start=0.1)
    gen = torch.Generator().manual_seed(seed)
    m.train()
    for _ in range(HP["epoch"]):
        urut = torch.randperm(len(Xtr), generator=gen)
        for i in range(0, len(Xtr), HP["batch"]):
            b = urut[i:i + HP["batch"]]
            opt.zero_grad()
            nn.functional.cross_entropy(m(Xtr[b]), ytr[b]).backward()
            opt.step()
            sched.step()

    m.eval()
    with torch.no_grad():
        logit_val = m(Xva)
        assert logit_val.shape == (len(Xva), N_KELAS), logit_val.shape
        pred_val = logit_val.argmax(1).numpy()
        pred_test = m(torch.from_numpy(Xte)).argmax(1).numpy() if final and Xte is not None else None
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
    subprocess.run(perintah, check=True, cwd=AKAR, env=dict(os.environ, PYTHONIOENCODING="utf-8"))


if __name__ == "__main__":
    main()
