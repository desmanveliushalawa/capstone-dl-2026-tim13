"""Metrik evaluasi: F1-makro per peserta (metrik utama protokol Tim 13)."""
from __future__ import annotations

import numpy as np


def akurasi(y: np.ndarray, p: np.ndarray) -> float:
    y, p = np.asarray(y).ravel(), np.asarray(p).ravel()
    return float((y == p).mean()) if len(y) else 0.0


def f1_makro(y: np.ndarray, p: np.ndarray, label: list[int] | None = None) -> float:
    y, p = np.asarray(y).ravel().astype(int), np.asarray(p).ravel().astype(int)
    if label is None:
        label = sorted(set(y.tolist()) | set(p.tolist()))
    skor = []
    for c in label:
        tp = int(((y == c) & (p == c)).sum())
        fp = int(((y != c) & (p == c)).sum())
        fn = int(((y == c) & (p != c)).sum())
        skor.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(skor)) if skor else 0.0


def per_peserta(
    y: np.ndarray,
    p: np.ndarray,
    peserta: np.ndarray,
    label: list[int] | None = None,
) -> list[dict]:
    y, p, peserta = np.asarray(y).ravel(), np.asarray(p).ravel(), np.asarray(peserta).astype(str).ravel()
    hasil = []
    for nama in sorted(set(peserta.tolist())):
        m = peserta == nama
        hasil.append({
            "peserta": nama,
            "n": int(m.sum()),
            "akurasi": akurasi(y[m], p[m]),
            "f1_makro": f1_makro(y[m], p[m], label),
            "n_kelas_benar": int(len(set(y[m].tolist()))),
        })
    return hasil


def ringkas(baris: list[dict]) -> dict:
    f1 = np.array([b["f1_makro"] for b in baris], np.float64)
    acc = np.array([b["akurasi"] for b in baris], np.float64)
    return {
        "f1_makro_rerata": float(f1.mean()) if len(f1) else 0.0,
        "f1_makro_simpangan_baku": float(f1.std(ddof=1)) if len(f1) > 1 else 0.0,
        "akurasi_rerata": float(acc.mean()) if len(acc) else 0.0,
        "akurasi_simpangan_baku": float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
        "jumlah_peserta": int(len(baris)),
    }


def matriks_konfusi(y: np.ndarray, p: np.ndarray, label: list[int] | None = None) -> np.ndarray:
    y, p = np.asarray(y).ravel().astype(int), np.asarray(p).ravel().astype(int)
    if label is None:
        label = sorted(set(y.tolist()) | set(p.tolist()))
    pos = {c: i for i, c in enumerate(label)}
    m = np.zeros((len(label), len(label)), np.int64)
    for a, b in zip(y, p):
        m[pos[a], pos[b]] += 1
    return m
