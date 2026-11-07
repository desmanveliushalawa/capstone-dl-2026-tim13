"""Skrip Pemrosesan & Penggabungan Data Primer Meja Uji.
Tim 13 — Capstone Deep Learning (Ruang Cerdas 5 Volt).

Sesuai Roadmap Sprint 1:
    1. Membaca seluruh rekaman .npy di data/mentah/
    2. Membuang 1 detik pertama tiap kondisi (menghilangkan masa transisi gerak tangan)
    3. Menggabungkan fitur koordinat 21 titik (x, y, z) dan label 0..7
    4. Menyimpan dataset akhir ke data/gestur_primer.npz
    5. Menghitung data_hash dan memperbarui data/MANIFEST.json

Penggunaan:
    python scripts/proses_data_primer.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AKAR))
from src.common import manifest

FOLDER_MENTAH = AKAR / "data" / "mentah"
BERKAS_OUTPUT = AKAR / "data" / "gestur_primer.npz"
BERKAS_MANIFEST = AKAR / "data" / "MANIFEST.json"


def main() -> None:
    if not FOLDER_MENTAH.exists():
        print(f"Folder {FOLDER_MENTAH} belum ditemukan. Silakan lakukan perekaman terlebih dahulu.")
        return

    berkas_npy = sorted(FOLDER_MENTAH.glob("*.npy"))
    if not berkas_npy:
        print(f"Tidak ada berkas .npy di {FOLDER_MENTAH}.")
        return

    print(f"Ditemukan {len(berkas_npy)} berkas rekaman di {FOLDER_MENTAH}...")

    X_list = []
    y_list = []
    metadata = []

    # Asumsi 30 FPS, 3 detik total = 90 frame.
    # 1 detik pertama = 30 frame dibuang.
    FRAME_BUANG = 30

    for f in berkas_npy:
        # Format nama berkas: <partisipan>_g<id>_l<lampu>_u<ulangan>.npy
        bagian = f.stem.split("_")
        if len(bagian) < 4:
            continue

        partisipan = bagian[0]
        id_gestur = int(bagian[1].replace("g", ""))
        lampu = int(bagian[2].replace("l", ""))
        ulangan = int(bagian[3].replace("u", ""))

        data_frame = np.load(f)  # [N_frames, 21, 3]
        if len(data_frame) <= FRAME_BUANG:
            # Jika frame terlalu pendek, ambil paruh kedua
            frame_stabil = data_frame[len(data_frame) // 2 :]
        else:
            frame_stabil = data_frame[FRAME_BUANG:]

        # Ambil median atau rata-rata landmark pada periode stabil
        # untuk gestur statis representatif
        landmark_representatif = np.median(frame_stabil, axis=0)  # [21, 3]

        X_list.append(landmark_representatif)
        y_list.append(id_gestur)
        metadata.append({
            "partisipan": partisipan,
            "gestur": id_gestur,
            "lampu": lampu,
            "ulangan": ulangan,
            "n_frame_stabil": len(frame_stabil),
        })

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    print(f"\nDataset terkumpul:")
    print(f"  Shape X: {X.shape} (Jumlah sampel, 21 landmark, 3 koordinat)")
    print(f"  Shape y: {y.shape} (Label 0..7)")
    print(f"  Distribusi kelas: {np.bincount(y)}")

    # Simpan dataset
    np.savez_compressed(BERKAS_OUTPUT, X=X, y=y)
    print(f"\n-> Dataset berhasil disimpan ke: {BERKAS_OUTPUT.relative_to(AKAR)}")

    # Perbarui data/MANIFEST.json
    konten_manifest = {
        "sumber": "https://github.com/desmanveliushalawa/capstone-dl-2026-tim13",
        "berkas": ["data/gestur_primer.npz"],
        "catatan": f"Dataset primer rekaman meja uji Tim 13 ({len(X)} sampel, buang 1 detik pertama, 8 gestur).",
    }
    BERKAS_MANIFEST.write_text(json.dumps(konten_manifest, indent=2), encoding="utf-8")
    print(f"-> MANIFEST diperbarui di: {BERKAS_MANIFEST.relative_to(AKAR)}")

    # Tampilkan hash data baru
    h = manifest.hitung_hash()
    print(f"-> Data Hash baru: {h}")
    print("\nLangkah selanjutnya:")
    print("  1. Perbarui src/protokol/data.py jika beralih ke data/gestur_primer.npz")
    print("  2. Jalankan ulang run_all.py untuk memverifikasi seluruh 5 model.")


if __name__ == "__main__":
    main()

