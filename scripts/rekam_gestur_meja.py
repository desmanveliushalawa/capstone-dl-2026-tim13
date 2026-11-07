"""Skrip Perekaman Data Primer Gestur Tangan (MediaPipe & Kamera Meja Uji).
Kasus: Ruang Cerdas 5 Volt (Tim 13 - Capstone Deep Learning ITERA).

Protokol Meja Uji:
    - 8 Partisipan: P01 - P08
    - 8 Gestur Statis:
        0: Tangan terbuka
        1: Kepalan
        2: Jempol ke atas
        3: Jempol ke bawah
        4: Telunjuk ke atas
        5: Dua jari (V)
        6: Tanda OK
        7: Telapak menghadap kamera lalu turun
    - 2 Kondisi Lampu: 128 (redup) dan 255 (terang)
    - 3 Kali Ulangan per kondisi
    - Durasi: 3 detik per pengambilan (90 frame @ 30 FPS)

Penggunaan:
    python scripts/rekam_gestur_meja.py --partisipan P01 --kamera 0
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Suppress noisy logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
try:
    import mediapipe as mp
except ImportError:
    print("Error: mediapipe belum terinstal. Jalankan: pip install mediapipe opencv-python")
    sys.exit(1)

AKAR = Path(__file__).resolve().parents[1]
FOLDER_OUTPUT = AKAR / "data" / "mentah"

DAFTAR_GESTUR = [
    "0: Tangan terbuka",
    "1: Kepalan",
    "2: Jempol ke atas",
    "3: Jempol ke bawah",
    "4: Telunjuk ke atas",
    "5: Dua jari (V)",
    "6: Tanda OK",
    "7: Telapak hadap kamera lalu turun",
]


def rekam_kondisi(
    cap: cv2.VideoCapture,
    hands: any,
    mp_draw: any,
    mp_hands: any,
    partisipan: str,
    id_gestur: int,
    lampu: int,
    ulangan: int,
    fps: int = 30,
    durasi_detik: int = 3,
) -> bool:
    """Merekam satu sesi gestur: 3 detik persiapan + 3 detik perekaman aktif."""
    nama_gestur = DAFTAR_GESTUR[id_gestur]
    total_frame_rekam = fps * durasi_detik

    # 1. Fase Hitung Mundur Persiapan (3 Detik)
    t_awal = time.time()
    while True:
        elapsed = time.time() - t_awal
        sisa = 3 - int(elapsed)
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame kamera.")
            return False

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        cv2.rectangle(frame, (10, 10), (w - 10, 120), (40, 40, 40), -1)
        cv2.putText(frame, f"Partisipan: {partisipan} | Lampu: {lampu} | Ulangan: {ulangan}/3", (25, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Gestur: {nama_gestur}", (25, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        cv2.putText(frame, f"BERSIAP: {max(1, sisa)} detik... (Posisikan tangan)", (25, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        cv2.imshow("Perekam Meja Uji - Tim 13", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            return False

        if elapsed >= 3.0:
            break

    # 2. Fase Perekaman Aktif (3 Detik = ~90 Frame)
    frame_landmarks = []
    frame_citra_terakhir = None
    t_rekam = time.time()
    n_tercatat = 0

    while n_tercatat < total_frame_rekam:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hasil = hands.process(rgb)

        coords = np.zeros((21, 3), dtype=np.float32)
        if hasil.multi_hand_landmarks:
            tangan = hasil.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, tangan, mp_hands.HAND_CONNECTIONS)
            for idx, lm in enumerate(tangan.landmark):
                coords[idx] = [lm.x, lm.y, lm.z]
            frame_citra_terakhir = frame.copy()

        frame_landmarks.append(coords)
        n_tercatat += 1

        # Visualisasi Status Rekam
        cv2.rectangle(frame, (10, 10), (w - 10, 80), (0, 0, 180), -1)
        sisa_waktu = max(0.0, durasi_detik - (time.time() - t_rekam))
        cv2.putText(frame, f"MEREKAM! Sisa: {sisa_waktu:.1f}s [{n_tercatat}/{total_frame_rekam}]",
                    (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"{nama_gestur}", (25, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Perekam Meja Uji - Tim 13", frame)
        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            return False

    # 3. Simpan Berkas
    FOLDER_OUTPUT.mkdir(parents=True, exist_ok=True)
    nama_dasar = f"{partisipan}_g{id_gestur}_l{lampu}_u{ulangan}"
    path_npy = FOLDER_OUTPUT / f"{nama_dasar}.npy"
    np.save(path_npy, np.array(frame_landmarks, dtype=np.float32))

    if frame_citra_terakhir is not None:
        path_jpg = FOLDER_OUTPUT / f"{nama_dasar}.jpg"
        cv2.imwrite(str(path_jpg), frame_citra_terakhir)

    print(f"  -> Tersimpan: {path_npy.name} ({len(frame_landmarks)} frame)")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Perekaman Data Gestur Meja Uji Tim 13")
    parser.add_argument("--partisipan", default="P01", help="ID Partisipan (misal: P01, P02, dst)")
    parser.add_argument("--kamera", type=int, default=0, help="Indeks kamera laptop/webcam (default: 0)")
    parser.add_argument("--fps", type=int, default=30, help="FPS perekaman (default: 30)")
    args = parser.parse_args()

    print("=" * 60)
    print(f" PEREKAM DATA PRIMER RUANG CERDAS 5 VOLT — TIM 13")
    print(f" Partisipan: {args.partisipan}")
    print("=" * 60)

    cap = cv2.VideoCapture(args.kamera)
    if not cap.isOpened():
        print(f"Gagal membuka kamera indeks {args.kamera}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    kondisi_lampu = [128, 255]
    total_ulangan = 3

    try:
        for id_gestur in range(8):
            for lampu in kondisi_lampu:
                for u in range(1, total_ulangan + 1):
                    print(f"\n[Ambil Data] Gestur: {DAFTAR_GESTUR[id_gestur]} | Lampu: {lampu} | Ulangan: {u}/3")
                    print("Tekan Spasi di jendela kamera untuk MULAI, atau 's' untuk Lewati, 'q' untuk Keluar.")

                    # Jeda konfirmasi sebelum mulai
                    mulai = False
                    while not mulai:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame = cv2.flip(frame, 1)
                        h, w, _ = frame.shape
                        cv2.rectangle(frame, (10, 10), (w - 10, 100), (50, 50, 50), -1)
                        cv2.putText(frame, f"Tekan SPASI untuk mulai rekam:", (25, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(frame, f"{DAFTAR_GESTUR[id_gestur]} (Lampu: {lampu}, U: {u})", (25, 75),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                        cv2.imshow("Perekam Meja Uji - Tim 13", frame)

                        k = cv2.waitKey(30) & 0xFF
                        if k == ord(" "):
                            mulai = True
                        elif k == ord("s"):
                            print("  -> Dilewati.")
                            break
                        elif k == ord("q") or k == 27:
                            print("\nPerekaman dihentikan pengguna.")
                            return

                    if mulai:
                        berhasil = rekam_kondisi(
                            cap, hands, mp_draw, mp_hands,
                            args.partisipan, id_gestur, lampu, u,
                            fps=args.fps, durasi_detik=3
                        )
                        if not berhasil:
                            print("Perekaman dibatalkan.")
                            return

        print("\n" + "=" * 60)
        print(f"Perekaman untuk {args.partisipan} SELESAI!")
        print(f"Berkas tersimpan di: {FOLDER_OUTPUT}")
        print("=" * 60)

    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

