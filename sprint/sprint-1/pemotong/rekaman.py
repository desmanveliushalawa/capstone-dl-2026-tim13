"""Membaca rekaman web Meja 5 Volt (.tar) tanpa mengekstrak seluruh citra."""
from __future__ import annotations

import csv
import tarfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Frame:
    nama: str          # nama member di dalam tar, mis. kamera/f000001.jpg
    berkas: str        # f000001.jpg
    t_host: float      # detik epoch jam komputer meja


def _baca_indeks(tf: tarfile.TarFile) -> dict[str, float]:
    idx: dict[str, float] = {}
    anggota = [m for m in tf.getmembers() if m.isfile()]
    kandidat = [m for m in anggota if m.name in ("kamera_index.csv", "kamera/kamera_index.csv")]
    kandidat += [m for m in anggota if m.name.startswith("kamera_index/bagian-") and m.name.endswith(".csv")]
    for m in kandidat:
        isi = tf.extractfile(m).read().decode("utf-8-sig", errors="replace")
        for row in csv.DictReader(isi.splitlines()):
            nama = (row.get("berkas") or "").strip()
            t = row.get("t_host")
            if nama and t not in (None, ""):
                idx[nama] = float(t)
    return idx


def baca_frames(path: str | Path, maks: int | None = None) -> list[Frame]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Rekaman tidak ditemukan: {p}")
    with tarfile.open(p) as tf:
        idx = _baca_indeks(tf)
        anggota = sorted(
            (m for m in tf.getmembers() if m.isfile() and m.name.startswith("kamera/f")),
            key=lambda m: m.name,
        )
        if maks is not None:
            anggota = anggota[:maks]
        frames = []
        for m in anggota:
            berkas = m.name.split("/")[-1]
            frames.append(Frame(nama=m.name, berkas=berkas, t_host=idx.get(berkas, float("nan"))))
    if not frames:
        raise ValueError(f"Tidak ada citra kamera/f*.jpg di {p}")
    # Isi waktu yang hilang dengan interpolasi linear bila indeks terpotong.
    t = np.array([f.t_host for f in frames], dtype=np.float64)
    baik = np.isfinite(t)
    if baik.sum() >= 2:
        t[~baik] = np.interp(np.flatnonzero(~baik), np.flatnonzero(baik), t[baik])
    elif baik.sum() == 1:
        t[~baik] = t[baik][0] + np.flatnonzero(~baik) * 1.0
    for f, nilai in zip(frames, t):
        f.t_host = float(nilai)
    return frames


def baca_citra(tf: tarfile.TarFile, nama: str) -> np.ndarray:
    buf = np.frombuffer(tf.extractfile(nama).read(), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Gagal membaca citra {nama}")
    return img


def hitung_profil(path: str | Path, frames: list[Frame]) -> dict[str, np.ndarray]:
    """Kembalikan kecerahan, diff antar-frame, dan mask frame terlihat.

    Diff dihitung pada citra kecil 160x120 skala abu-abu, cukup untuk mendeteksi
    pergantian peserta tanpa memuat citra penuh berulang kali.
    """
    n = len(frames)
    rerata = np.zeros(n, np.float32)
    diff = np.zeros(n, np.float32)
    sebelumnya: np.ndarray | None = None
    with tarfile.open(path) as tf:
        for i, f in enumerate(frames):
            kecil = cv2.resize(baca_citra(tf, f.nama), (160, 120))
            abu = cv2.cvtColor(kecil, cv2.COLOR_BGR2GRAY).astype(np.float32)
            rerata[i] = abu.mean()
            diff[i] = 0.0 if sebelumnya is None else float(np.abs(abu - sebelumnya).mean())
            sebelumnya = abu
            if (i + 1) % 250 == 0:
                print(f"  profil {i + 1}/{n}", flush=True)
    terlihat = rerata >= 20.0
    diff[~terlihat] = 0.0
    return {"rerata": rerata, "diff": diff, "terlihat": terlihat}


def _rata_bergerak(x: np.ndarray, lebar: int) -> np.ndarray:
    if lebar <= 1:
        return x
    inti = np.ones(lebar, dtype=np.float64) / lebar
    return np.convolve(x.astype(np.float64), inti, mode="same")


def deteksi_batas(
    profil: dict[str, np.ndarray],
    min_jarak: int = 80,
    tepi: int = 30,
    maks_batas: int = 7,
) -> list[int]:
    """Deteksi batas blok peserta dari lonjakan diff (orang masuk/keluar).

    Strategi: haluskan diff 5 frame, ambil puncak lokal di atas median + 5*MAD,
    lalu pilih puncak terkuat dengan jarak minimum agar tidak ada dua batas
    berdekatan. Kembalikan indeks frame batas (tanpa 0 dan ujung).
    """
    terlihat = profil["terlihat"]
    idx = np.flatnonzero(terlihat)
    if len(idx) < 2 * tepi:
        return []
    awal, akhir = int(idx[0]), int(idx[-1])
    halus = _rata_bergerak(profil["diff"], 5)

    dasar = halus[idx]
    med = float(np.median(dasar))
    mad = float(np.median(np.abs(dasar - med))) + 1e-9
    ambang = med + 5.0 * mad

    kandidat: list[tuple[float, int]] = []
    for i in range(awal + tepi, akhir - tepi + 1):
        if halus[i] < ambang:
            continue
        jendela = halus[max(awal, i - 3): min(akhir + 1, i + 4)]
        if halus[i] < jendela.max():
            continue
        kiri = halus[max(awal, i - min_jarak): i + 1].min()
        kanan = halus[i: min(akhir + 1, i + min_jarak + 1)].min()
        prominence = halus[i] - max(kiri, kanan)
        if prominence > 0.25 * halus[i]:
            kandidat.append((float(prominence), i))

    kandidat.sort(reverse=True)
    dipilih: list[int] = []
    for _, i in kandidat:
        if all(abs(i - j) >= min_jarak for j in dipilih):
            dipilih.append(i)
        if len(dipilih) >= maks_batas:
            break
    return sorted(dipilih)


def interval_terukur(frames: list[Frame]) -> float:
    t = np.array([f.t_host for f in frames], dtype=np.float64)
    if len(t) < 3:
        return float("nan")
    d = np.diff(t)
    d = d[np.isfinite(d) & (d > 0)]
    return float(np.median(d)) if len(d) else float("nan")
