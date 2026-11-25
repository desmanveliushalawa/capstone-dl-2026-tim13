"""Ekstraksi landmark MediaPipe dari rekaman primer meja uji (data/aktual).

    python scripts/ekstrak_landmark_aktual.py

Masukan:
    data/aktual/kamera/*.jpg              frame kamera
    data/aktual/kamera_index/*.csv        t_host tiap frame
    data/aktual/sesi-*/label.csv          384 kondisi (partisipan, gestur, lampu, t_mulai/t_selesai)

Keluaran:
    data/aktual/gestur_primer.npz         X [N,21,3], y [N], partisipan/lampu/ulangan/t_host/...
    data/aktual/gestur_primer_ringkasan.json

Aturan praproses protokol: buang 1 detik pertama setiap kondisi (fase transisi), ambil satu
landmark representatif per frame, dan cerminkan koordinat x bila MediaPipe mendeteksi tangan kiri.
"""
from __future__ import annotations

import csv
import glob
import json
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[1]
AKTUAL = AKAR / "data" / "aktual"
FOLDER_KAMERA = AKTUAL / "kamera"
POLA_INDEKS = str(AKTUAL / "kamera_index" / "bagian-*.csv")
BERKAS_LABEL = AKTUAL / "sesi-20261118-123649" / "label.csv"
BERKAS_OUTPUT = AKTUAL / "gestur_primer.npz"
BERKAS_RINGKASAN = AKTUAL / "gestur_primer_ringkasan.json"
MODEL_HAND = AKTUAL / "hand_landmarker.task"
URL_MODEL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")

DAFTAR_GESTUR = [
    "tangan terbuka",
    "kepalan",
    "jempol ke atas",
    "jempol ke bawah",
    "telunjuk ke atas",
    "dua jari (V)",
    "tanda OK",
    "telapak menghadap kamera lalu turun",
]
GESTUR_KE_LABEL = {g: i for i, g in enumerate(DAFTAR_GESTUR)}
FRAME_BUANG_DETIK = 1.0


def baca_indeks() -> list[tuple[str, float]]:
    baris: list[dict] = []
    for f in sorted(glob.glob(POLA_INDEKS)):
        with open(f, newline="", encoding="utf-8") as fh:
            baris.extend(csv.DictReader(fh))
    return [(r["berkas"], float(r["t_host"])) for r in baris if r.get("t_host")]


def baca_label() -> list[dict]:
    with open(BERKAS_LABEL, newline="", encoding="utf-8") as fh:
        kondisi = list(csv.DictReader(fh))
    for r in kondisi:
        r["t_mulai_host"] = float(r["t_mulai_host"])
        r["t_selesai_host"] = float(r["t_selesai_host"])
        r["lampu"] = int(r["lampu"])
        r["ulangan"] = int(r["ulangan"])
    return kondisi


def cari_kondisi(t: float, kondisi: list[dict]) -> dict | None:
    for r in kondisi:
        if r["t_mulai_host"] + FRAME_BUANG_DETIK <= t <= r["t_selesai_host"]:
            return r
    return None


def main() -> None:
    import mediapipe as mp

    vision = mp.tasks.vision
    if not MODEL_HAND.exists():
        MODEL_HAND.parent.mkdir(parents=True, exist_ok=True)
        print(f"mengunduh model MediaPipe -> {MODEL_HAND}")
        urllib.request.urlretrieve(URL_MODEL, MODEL_HAND)

    indeks = baca_indeks()
    kondisi = baca_label()
    print(f"frame kamera: {len(indeks)} | kondisi label: {len(kondisi)}")

    kandidat = []
    for nama, t in indeks:
        r = cari_kondisi(t, kondisi)
        if r is not None:
            kandidat.append((nama, t, r))
    print(f"kandidat setelah buang {FRAME_BUANG_DETIK:.0f} detik pertama: {len(kandidat)}")

    opsi = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_HAND)),
        num_hands=1,
        running_mode=vision.RunningMode.IMAGE,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    X, y = [], []
    partisipan, gestur, tangan, berkas = [], [], [], []
    lampu, ulangan, t_host = [], [], []
    hilang = Counter()
    t0 = time.time()

    with vision.HandLandmarker.create_from_options(opsi) as landmarker:
        for i, (nama, t, r) in enumerate(kandidat, 1):
            path = FOLDER_KAMERA / nama
            if not path.exists():
                hilang["berkas_tidak_ada"] += 1
                continue
            hasil = landmarker.detect(mp.Image.create_from_file(str(path)))
            if not hasil.hand_landmarks:
                hilang[f"{r['partisipan']}|{r['gestur']}"] += 1
                continue
            lm = hasil.hand_landmarks[0]
            if len(lm) != 21:
                hilang["landmark_bukan_21"] += 1
                continue
            koord = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
            sisi = hasil.handedness[0][0].category_name if hasil.handedness else "?"
            if sisi == "Left":
                koord[:, 0] = 1.0 - koord[:, 0]
            X.append(koord)
            y.append(GESTUR_KE_LABEL[r["gestur"]])
            partisipan.append(r["partisipan"])
            gestur.append(r["gestur"])
            tangan.append(sisi)
            berkas.append(nama)
            lampu.append(r["lampu"])
            ulangan.append(r["ulangan"])
            t_host.append(t)
            if i % 50 == 0:
                print(f"  {i}/{len(kandidat)} frame, {len(X)} terdeteksi, {time.time() - t0:.1f}s")

    if not X:
        raise SystemExit("Tidak ada tangan terdeteksi.")

    X = np.stack(X).astype(np.float32)
    y = np.asarray(y, dtype=np.int64)
    np.savez_compressed(
        BERKAS_OUTPUT,
        X=X,
        y=y,
        partisipan=np.asarray(partisipan),
        gestur=np.asarray(gestur),
        tangan=np.asarray(tangan),
        berkas=np.asarray(berkas),
        lampu=np.asarray(lampu, dtype=np.int64),
        ulangan=np.asarray(ulangan, dtype=np.int64),
        t_host=np.asarray(t_host, dtype=np.float64),
    )

    per_partisipan = Counter(partisipan)
    per_kelas = Counter(gestur)
    matriks = defaultdict(int)
    for p, g in zip(partisipan, gestur):
        matriks[f"{p}|{g}"] += 1
    ringkasan = {
        "sumber_kamera": str(FOLDER_KAMERA.relative_to(AKAR)),
        "sumber_label": str(BERKAS_LABEL.relative_to(AKAR)),
        "n_frame_kandidat": len(kandidat),
        "n_sampel": int(len(X)),
        "n_ditolak_tanpa_tangan": int(len(kandidat) - len(X)),
        "durasi_detik": round(time.time() - t0, 1),
        "frame_buang_detik": FRAME_BUANG_DETIK,
        "per_partisipan": dict(sorted(per_partisipan.items())),
        "per_kelas": {g: per_kelas[g] for g in DAFTAR_GESTUR},
        "kombinasi_partisipan_gestur": dict(sorted(matriks.items())),
        "frame_tanpa_tangan": dict(hilang),
    }
    BERKAS_RINGKASAN.write_text(json.dumps(ringkasan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"sampel tersimpan: {len(X)} -> {BERKAS_OUTPUT.relative_to(AKAR)}")
    print(f"per partisipan: {dict(sorted(per_partisipan.items()))}")
    print(f"per kelas: {ringkasan['per_kelas']}")
    print(f"ringkasan -> {BERKAS_RINGKASAN.relative_to(AKAR)}")


if __name__ == "__main__":
    main()
