"""Pemuat data kasus tim — MILIK TIM, dibekukan setelah Sprint 0 dengan persetujuan pembimbing."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from src.common import manifest, split

TUGAS = "klasifikasi"
BERKAS_DATA = manifest.AKAR / "data" / "gestur_sampel.npz"


def muat():
    """Memuat dataset gestur tangan 21 titik koordinat MediaPipe untuk 8 kelas gestur."""
    if not BERKAS_DATA.exists():
        raise FileNotFoundError(f"Berkas data {BERKAS_DATA} tidak ditemukan.")
    
    data = np.load(BERKAS_DATA)
    X = data["X"]  # Bentuk: [N, 21, 3]
    y = data["y"]  # Label 0..7
    
    # Split beku latih dan validasi (seed 2026, porsi validasi 10%)
    tr, va = split.indeks(len(X), porsi_val=0.1, seed=2026)
    
    # Untuk data uji terpisah (Sprint 2 / evaluasi akhir)
    X_test, y_test = None, None
    
    return X[tr], y[tr], X[va], y[va], X_test, y_test