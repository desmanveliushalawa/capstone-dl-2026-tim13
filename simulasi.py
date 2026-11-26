"""Pembuat data simulasi gestur (HANYA bila data/simulasi/gestur_sampel.npz tim belum ada di laptop).

    python simulasi.py

Menulis data/simulasi/gestur_sampel.npz berisi X [800, 21, 3] dan y [800] (label 0..7), sama bentuknya
dengan sampel Sprint 0. Isinya buatan: tiap kelas punya "bentuk tangan" prototipe + derau,
sehingga model yang benar dapat belajar. data/ tidak di-commit; data_hash akan BERBEDA dari
hash tim (148ea43c...), jadi jangan dipakai untuk hasil yang dilaporkan. Data aktual ada di
data/aktual/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parent
TUJUAN = AKAR / "data" / "simulasi" / "gestur_sampel.npz"


def buat(n: int = 800, n_kelas: int = 8, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    prototipe = rng.normal(0.0, 0.25, size=(n_kelas, 21, 3)).astype(np.float32)
    y = np.tile(np.arange(n_kelas), n // n_kelas + 1)[:n]
    rng.shuffle(y)
    geser = rng.normal(0.5, 0.05, size=(n, 1, 3))           # posisi tangan di bingkai
    derau = rng.normal(0.0, 0.08, size=(n, 21, 3))          # jitter landmark
    return (prototipe[y] + geser + derau).astype(np.float32), y.astype(np.int64)


if __name__ == "__main__":
    if TUJUAN.exists():
        raise SystemExit(f"{TUJUAN} sudah ada; tidak ditimpa.")
    X, y = buat()
    TUJUAN.parent.mkdir(exist_ok=True)
    np.savez(TUJUAN, X=X, y=y)
    print("ditulis", TUJUAN, X.shape, y.shape)
