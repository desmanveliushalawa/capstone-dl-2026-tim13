"""Model Pembanding Non-DL: Random Forest + Statistik Sudut Sendi & Kecepatan Pose.
Tim 13 — Capstone Deep Learning (Ruang Cerdas 5 Volt).
Penanggung Jawab: Khoirul Muttoharoh (123450021 - Analis Utama) / Tiket #8.

Metode Rekayasa Fitur Fisik Tangan:
    1. Koordinat dasar 21 landmark MediaPipe (63 fitur).
    2. Sudut sendi (joint angles) pada 5 jari: CMC-MCP-PIP-DIP-TIP (15 sudut).
    3. Jarak relatif ujung jari ke pergelangan tangan (wrist distance).
    4. Jarak antar-ujung jari (fingertip spans), khusus pendeteksi gestur OK (jempol bertemu telunjuk).
    5. Statistik dispersi koordinat (mean, std, bounding span).
    6. Model: RandomForestClassifier (100 estimators).

Penggunaan:
    python scripts/baseline_rf.py --seed 2026
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

AKAR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AKAR))
from src.protokol import data

DAFTAR_GESTUR = [
    "0: Terbuka",
    "1: Kepalan",
    "2: Jempol Atas",
    "3: Jempol Bawah",
    "4: Telunjuk Atas",
    "5: Dua Jari (V)",
    "6: Tanda OK",
    "7: Hadap & Turun",
]


def hitung_sudut(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Menghitung sudut (dalam radian) pada titik sendi b dari 3 titik 3D: a -> b -> c."""
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba, axis=-1, keepdims=True) + 1e-7
    norm_bc = np.linalg.norm(bc, axis=-1, keepdims=True) + 1e-7
    cosine = np.sum(ba * bc, axis=-1, keepdims=True) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.arccos(cosine).squeeze(-1)


def ekstraksi_fitur_tangan(X: np.ndarray) -> np.ndarray:
    """Mengekstraksi fitur geometri sendi, jarak, dan sudut dari koordinat [N, 21, 3]."""
    N = len(X)
    fitur_list = []

    # 1. Koordinat Mentah Diratakan (63 Fitur)
    fitur_list.append(X.reshape(N, -1))

    # Definisi Landmark MediaPipe:
    # 0: Wrist
    # Jempol: 1, 2, 3, 4
    # Telunjuk: 5, 6, 7, 8
    # Tengah: 9, 10, 11, 12
    # Manis: 13, 14, 15, 16
    # Kelingking: 17, 18, 19, 20
    rantai_jari = [
        (0, 1, 2), (1, 2, 3), (2, 3, 4),        # Jempol
        (0, 5, 6), (5, 6, 7), (6, 7, 8),        # Telunjuk
        (0, 9, 10), (9, 10, 11), (10, 11, 12),  # Tengah
        (0, 13, 14), (14, 15, 16), (15, 16, 17),# Manis
        (0, 17, 18), (18, 19, 20), (19, 20, 20),# Kelingking
    ]

    # 2. Sudut Sendi Antar Ruas Jari (15 Fitur)
    sudut_jari = []
    for p1, p2, p3 in rantai_jari:
        sudut_jari.append(hitung_sudut(X[:, p1], X[:, p2], X[:, p3]))
    fitur_list.append(np.column_stack(sudut_jari))

    # 3. Jarak Ujung Jari ke Pergelangan Tangan (5 Fitur)
    ujung_jari = [4, 8, 12, 16, 20]
    jarak_wrist = []
    for tip in ujung_jari:
        d = np.linalg.norm(X[:, tip] - X[:, 0], axis=-1)
        jarak_wrist.append(d)
    fitur_list.append(np.column_stack(jarak_wrist))

    # 4. Jarak Antar Ujung Jari (Krusial untuk Gestur OK: Jempol bertemu Telunjuk) (4 Fitur)
    jarak_antar_tip = [
        np.linalg.norm(X[:, 4] - X[:, 8], axis=-1),   # Jempol - Telunjuk (Tanda OK -> jarak mendekati 0)
        np.linalg.norm(X[:, 8] - X[:, 12], axis=-1),  # Telunjuk - Tengah
        np.linalg.norm(X[:, 12] - X[:, 16], axis=-1), # Tengah - Manis
        np.linalg.norm(X[:, 16] - X[:, 20], axis=-1), # Manis - Kelingking
    ]
    fitur_list.append(np.column_stack(jarak_antar_tip))

    # 5. Statistik Koordinat (Rentang / Bounding Span & Mean Posisi) (6 Fitur)
    span_xyz = X.max(axis=1) - X.min(axis=1)  # [N, 3]
    mean_xyz = X.mean(axis=1)                 # [N, 3]
    fitur_list.append(span_xyz)
    fitur_list.append(mean_xyz)

    hasil_fitur = np.hstack(fitur_list).astype(np.float32)
    return hasil_fitur


def main() -> None:
    ap = argparse.ArgumentParser(description="Baseline Non-DL: Random Forest + Sudut Sendi")
    ap.add_argument("--seed", type=int, default=2026, help="Random seed (default: 2026)")
    ap.add_argument("--estimators", type=int, default=100, help="Jumlah pohon di Random Forest")
    args = ap.parse_args()

    print("=" * 65)
    print(" BASELINE PEMBANDING NON-DL: RANDOM FOREST + SUDUT SENDI")
    print(f" Kasus: Ruang Cerdas 5 Volt (Tim 13) | Seed: {args.seed}")
    print("=" * 65)

    # 1. Muat Data Resmi Tim
    Xtr, ytr, Xva, yva, _, _ = data.muat()
    print(f"Data latih: {Xtr.shape[0]} sampel | Data validasi: {Xva.shape[0]} sampel")

    # 2. Rekayasa Fitur Geometri Tangan
    t0 = time.time()
    Ftr = ekstraksi_fitur_tangan(Xtr)
    Fva = ekstraksi_fitur_tangan(Xva)
    durasi_fe = time.time() - t0
    print(f"Fitur diekstraksi: {Ftr.shape[1]} fitur fisik/sampel ({durasi_fe:.3f}s)")

    # 3. Latih Random Forest
    rf = RandomForestClassifier(
        n_estimators=args.estimators,
        max_depth=12,
        random_state=args.seed,
        n_jobs=-1,
    )
    t_train = time.time()
    rf.fit(Ftr, ytr)
    durasi_train = time.time() - t_train

    # 4. Prediksi & Evaluasi
    p_val = rf.predict(Fva)
    durasi_total = time.time() - t0

    acc = float((p_val == yva).mean())
    # Macro F1 manual sesuai standar evaluasi
    f1_list = []
    for c in np.unique(np.concatenate([yva, p_val])):
        tp = int(((p_val == c) & (yva == c)).sum())
        fp = int(((p_val == c) & (yva != c)).sum())
        fn = int(((p_val != c) & (yva == c)).sum())
        f1_list.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    macro_f1 = float(np.mean(f1_list))

    print("\n" + "-" * 40)
    print(" HASIL EVALUASI BASELINE NON-DL")
    print("-" * 40)
    print(f" Akurasi Validasi (Val Acc)      : {acc * 100:.2f}% ({acc:.4f})")
    print(f" Macro-F1 Validasi               : {macro_f1:.4f}")
    print(f" Durasi Pelatihan                : {durasi_train:.2f} detik")
    print("-" * 40)

    # Simpan Prediksi dan Metrik
    out = AKAR / "results" / "baseline_rf"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_val.npy", p_val)

    metrik_json = {
        "metrik": {
            "val_acc": round(acc, 4),
            "val_macro_f1": round(macro_f1, 6),
        },
        "seed": args.seed,
        "model": "random_forest_joint_angles",
        "n_estimators": args.estimators,
        "n_features": Ftr.shape[1],
        "durasi_detik": round(durasi_total, 2),
        "perangkat": "cpu laptop",
        "catatan": "Baseline pembanding non-DL statistik sudut sendi + jarak pose",
    }
    (out / "metrics.json").write_text(json.dumps(metrik_json, indent=2), encoding="utf-8")
    print(f"-> Hasil tersimpan di: {out.relative_to(AKAR)}/metrics.json")

    # Tampilkan Confusion Matrix
    print("\nMatriks Kebingungan (Confusion Matrix 8 Kelas):")
    cm = confusion_matrix(yva, p_val)
    print(cm)


if __name__ == "__main__":
    main()

