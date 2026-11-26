"""Ekstraksi 21 landmark tangan (MediaPipe Hands) dan potongan ROI.

Mendukung dua API MediaPipe:
  * Tasks API baru (mediapipe >= 1.0, modul ``mediapipe.tasks``)
  * Solutions API lama (mediapipe 0.10.x, ``mp.solutions.hands``)

Model ``hand_landmarker.task`` diunduh otomatis bila belum ada.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_BAWAAN = Path(__file__).resolve().parents[1] / "models_mediapipe" / "hand_landmarker.task"


def pastikan_model(path: str | Path | None = None) -> Path:
    p = Path(path) if path else MODEL_BAWAAN
    if p.exists() and p.stat().st_size > 1_000_000:
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    print(f"  mengunduh model MediaPipe -> {p}", flush=True)
    try:
        urllib.request.urlretrieve(MODEL_URL, p)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Gagal mengunduh model MediaPipe. Unduh manual dari {MODEL_URL} "
            f"lalu simpan sebagai {p}."
        ) from exc
    return p


def bbox_dari_landmark(landmark: np.ndarray, lebar: int, tinggi: int) -> np.ndarray:
    xs = np.clip(landmark[:, 0], 0, 1) * lebar
    ys = np.clip(landmark[:, 1], 0, 1) * tinggi
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.float32)


def potong_roi(img: np.ndarray, landmark: np.ndarray, ukuran: int = 224, margin: float = 0.35):
    """Potong kotak di sekitar tangan; aman bila kotak keluar batas citra."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = bbox_dari_landmark(landmark, w, h)
    sisi = max(x1 - x0, y1 - y0)
    sisi = max(sisi * (1.0 + 2.0 * margin), 48.0)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    kiri, atas = int(round(cx - sisi / 2)), int(round(cy - sisi / 2))
    kanan, bawah = int(round(cx + sisi / 2)), int(round(cy + sisi / 2))
    pad_kiri = max(0, -kiri)
    pad_atas = max(0, -atas)
    pad_kanan = max(0, kanan - w)
    pad_bawah = max(0, bawah - h)
    if any((pad_kiri, pad_atas, pad_kanan, pad_bawah)):
        img = cv2.copyMakeBorder(img, pad_atas, pad_bawah, pad_kiri, pad_kanan, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        kiri += pad_kiri
        kanan += pad_kiri
        atas += pad_atas
        bawah += pad_atas
    crop = img[max(0, atas):max(1, bawah), max(0, kiri):max(1, kanan)]
    if crop.size == 0:
        crop = img
    return cv2.resize(crop, (ukuran, ukuran), interpolation=cv2.INTER_AREA), (kiri, atas, kanan, bawah)


class DetektorTangan:
    """Pembungkus tipis MediaPipe yang mengembalikan landmark [21,3] atau None."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        conf: float = 0.1,
        num_hands: int = 1,
        mode: str = "VIDEO",
    ) -> None:
        import mediapipe as mp  # impor lambat agar CLI tetap ringan

        self.mp = mp
        self.num_hands = int(num_hands)
        self.mode = mode.upper()
        self._t_akhir = -1
        self.api = ""

        if hasattr(mp, "tasks"):
            from mediapipe.tasks import python as mpp
            from mediapipe.tasks.python import vision

            model = pastikan_model(model_path)
            running = vision.RunningMode.VIDEO if self.mode == "VIDEO" else vision.RunningMode.IMAGE
            opsi = vision.HandLandmarkerOptions(
                base_options=mpp.BaseOptions(model_asset_path=str(model)),
                running_mode=running,
                num_hands=self.num_hands,
                min_hand_detection_confidence=float(conf),
                min_hand_presence_confidence=float(conf),
                min_tracking_confidence=float(conf),
            )
            self._det = vision.HandLandmarker.create_from_options(opsi)
            self._vision = vision
            self.api = "tasks"
        elif hasattr(mp, "solutions"):
            self._det = mp.solutions.hands.Hands(
                static_image_mode=(self.mode != "VIDEO"),
                max_num_hands=self.num_hands,
                model_complexity=1,
                min_detection_confidence=float(conf),
                min_tracking_confidence=float(conf),
            )
            self.api = "solutions"
        else:
            raise RuntimeError("MediaPipe terpasang tetapi tidak punya API tasks maupun solutions.")

    def deteksi(self, img_bgr: np.ndarray, t_ms: int = 0):
        """Kembalikan (landmark|None, bbox|None, skor, label_tangan)."""
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if self.api == "tasks":
            citra = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
            if self.mode == "VIDEO":
                t = max(int(t_ms), self._t_akhir + 1)
                self._t_akhir = t
                hasil = self._det.detect_for_video(citra, t)
            else:
                hasil = self._det.detect(citra)
            if not hasil.hand_landmarks:
                return None, None, float("nan"), ""
            tangan = hasil.hand_landmarks[0]
            lm = np.array([(p.x, p.y, p.z) for p in tangan], np.float32)
            skor, label = float("nan"), ""
            if hasil.handedness:
                kat = hasil.handedness[0][0]
                skor, label = float(getattr(kat, "score", float("nan"))), str(getattr(kat, "category_name", ""))
        else:
            hasil = self._det.process(rgb)
            if not hasil.multi_hand_landmarks:
                return None, None, float("nan"), ""
            tangan = hasil.multi_hand_landmarks[0]
            lm = np.array([(p.x, p.y, p.z) for p in tangan.landmark], np.float32)
            skor, label = float("nan"), ""
            if hasil.multi_handedness:
                kat = hasil.multi_handedness[0].classification[0]
                skor, label = float(kat.score), str(kat.label)
        if lm.shape != (21, 3):
            return None, None, float("nan"), ""
        h, w = img_bgr.shape[:2]
        return lm, bbox_dari_landmark(lm, w, h), skor, label

    def tutup(self) -> None:
        try:
            self._det.close()
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.tutup()
