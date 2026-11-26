"""Pemuat data kasus tim — MILIK TIM, dibekukan setelah Sprint 0 dengan persetujuan pembimbing.

Protokol Sprint 1 (lihat ``src/protokol/PROTOKOL.md``):
  * sumber bawaan adalah folder sesi simulasi hasil ``meja.py --simulasi jalankan``;
  * tiap kondisi membuang 1 detik pertama;
  * split latih/validasi/uji terpisah per partisipan;
  * normalisasi mean/std dihitung dari data latih saja.

Kontrak ``muat()`` tetap sama seperti starter:
``X_train, y_train, X_val, y_val, X_test, y_test``.

Untuk memakai dataset aktual ``data/aktual/gestur_primer.npz`` (jalur Sprint 0),
setel ``PROTOKOL_SUMBER=aktual`` atau panggil :func:`muat_aktual` langsung.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from . import sesi

TUGAS = "klasifikasi"
AKAR = Path(__file__).resolve().parents[2]
BERKAS_AKTUAL = AKAR / "data" / "aktual" / "gestur_primer.npz"


def sumber_aktif() -> str:
    """``simulasi`` (bawaan) atau ``aktual`` sesuai ``PROTOKOL_SUMBER``."""
    pilihan = os.environ.get("PROTOKOL_SUMBER", "simulasi").strip().lower()
    if pilihan not in ("simulasi", "aktual"):
        raise ValueError(f"PROTOKOL_SUMBER harus 'simulasi' atau 'aktual', bukan {pilihan!r}")
    return pilihan


def muat():
    """Kembalikan X/y latih, validasi, dan uji dari folder sesi simulasi."""
    if sumber_aktif() == "aktual":
        return muat_aktual()
    return sesi.muat()


def muat_lipat(partisipan: str):
    """Split GroupKFold: partisipan ``partisipan`` menjadi data uji."""
    return sesi.muat(lipat=partisipan)


def muat_detail(lipat: str | None = None):
    return sesi.muat_detail(lipat=lipat)


def info_split(lipat: str | None = None):
    return sesi.info_split(lipat=lipat)


def muat_aktual():
    """Jalur data aktual (Sprint 0): split acak 10 % dengan seed 2026.

    Dipertahankan agar skrip anggota tim yang masih memakai dataset aktual tetap
    berjalan. Protokol Sprint 1 memakai :func:`muat` (split per partisipan).
    """
    if not BERKAS_AKTUAL.exists():
        raise FileNotFoundError(
            f"Dataset aktual tidak ada: {BERKAS_AKTUAL}. "
            "Jalankan pemotong sesi simulasi atau sediakan data/aktual/gestur_primer.npz."
        )
    from src.common import split

    data = np.load(BERKAS_AKTUAL)
    X, y = data["X"], data["y"]
    tr, va = split.indeks(len(X), porsi_val=0.1, seed=2026)
    return X[tr], y[tr], X[va], y[va], None, None


def muat_partisipan() -> np.ndarray:
    """Label partisipan untuk seluruh sampel sebelum split.

    Pada sumber simulasi, urutannya sama dengan ``sesi.muat_mentah()["peserta"]``.
    Untuk metrik per partisipan pada split, pakai ``muat_detail()["peserta_uji"]``.
    """
    if sumber_aktif() == "aktual":
        return np.load(BERKAS_AKTUAL)["partisipan"]
    data, _ = sesi.muat_mentah()
    return data["peserta"]
