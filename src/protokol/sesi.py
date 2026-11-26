"""Pemuatan sesi simulasi + split per peserta + normalisasi dari data latih.

Modul ini dipakai oleh ``src/protokol/data.py``. Aturan yang dibekukan:
  * satu peserta hanya muncul di satu sisi split (latih/validasi/uji);
  * normalisasi (mean/std) dihitung dari data latih saja;
  * data uji tidak pernah dipakai untuk memilih hyperparameter.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
SESI_BAWAAN = AKAR / "sprint" / "sprint-1" / "sesi_simulasi"
DATASET = "dataset.npz"
METADATA = "sesi.json"


def folder_sesi(folder: str | Path | None = None) -> Path:
    return Path(folder) if folder is not None else SESI_BAWAAN


def muat_mentah(folder: str | Path | None = None) -> tuple[dict[str, np.ndarray], dict]:
    p = folder_sesi(folder)
    berkas = p / DATASET
    if not berkas.exists():
        raise FileNotFoundError(
            f"Dataset sesi belum ada: {berkas}. Jalankan dulu:\n"
            f"  python sprint/sprint-1/meja.py --simulasi jalankan sprint/rencana/t13-gestur-kamera.yaml"
        )
    z = np.load(berkas, allow_pickle=False)
    data = {k: z[k] for k in z.files}
    for k in ("peserta", "kondisi"):
        if k in data:
            data[k] = data[k].astype(str)
    meta = json.loads((p / METADATA).read_text(encoding="utf-8")) if (p / METADATA).exists() else {}
    return data, meta


def daftar_peserta(data: dict) -> list[str]:
    return sorted(set(data["peserta"].tolist()))


def bagi(
    data: dict,
    meta: dict | None = None,
    lipat: str | None = None,
    latih: list[str] | None = None,
    validasi: list[str] | None = None,
    uji: list[str] | None = None,
) -> dict:
    """Tentukan indeks latih/validasi/uji yang terpisah per peserta.

    lipat=Pxx  -> uji = Pxx, validasi = peserta berikutnya (siklik), latih = sisanya.
    Tanpa lipat -> pakai ``split_beku`` di sesi.json, atau 3 peserta terakhir.
    """
    semua = daftar_peserta(data)
    if len(semua) < 3:
        raise ValueError(f"Perlu minimal 3 peserta; dataset hanya berisi {semua}.")

    if latih is not None or validasi is not None or uji is not None:
        latih = list(latih or [])
        validasi = list(validasi or [])
        uji = list(uji or [])
        if not (latih and validasi and uji):
            raise ValueError("latih, validasi, dan uji harus diisi ketiganya.")
    elif lipat is not None:
        if lipat not in semua:
            raise ValueError(f"Peserta lipat {lipat!r} tidak ada di dataset {semua}.")
        i = semua.index(lipat)
        uji = [lipat]
        validasi = [semua[(i + 1) % len(semua)]]
        latih = [p for p in semua if p not in uji + validasi]
    else:
        beku = (meta or {}).get("split_beku")
        if beku and all(k in beku for k in ("latih", "validasi", "uji")):
            latih, validasi, uji = list(beku["latih"]), list(beku["validasi"]), list(beku["uji"])
        else:
            latih, validasi, uji = semua[:-2], [semua[-2]], [semua[-1]]

    semua_sisi = latih + validasi + uji
    if len(set(semua_sisi)) != len(semua_sisi):
        raise ValueError(f"Peserta muncul di lebih dari satu sisi split: {latih}, {validasi}, {uji}")
    for nama, sisi in (("latih", latih), ("validasi", validasi), ("uji", uji)):
        if not sisi:
            raise ValueError(f"Sisi {nama} kosong.")

    p = data["peserta"]
    return {
        "latih": np.flatnonzero(np.isin(p, latih)),
        "validasi": np.flatnonzero(np.isin(p, validasi)),
        "uji": np.flatnonzero(np.isin(p, uji)),
        "peserta_latih": latih,
        "peserta_validasi": validasi,
        "peserta_uji": uji,
    }


def _fit_normalisasi(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X2 = np.asarray(X, np.float32).reshape(len(X), -1)
    mu = X2.mean(axis=0)
    sd = X2.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu.astype(np.float32), sd.astype(np.float32)


def _terapkan(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    bentuk = X.shape
    X2 = np.asarray(X, np.float32).reshape(len(X), -1)
    return ((X2 - mu) / sd).reshape(bentuk).astype(np.float32)


def normalkan_pose(X: np.ndarray, tangan: np.ndarray | None = None) -> np.ndarray:
    """Normalisasi pose per sampel: cermin tangan kiri, titik asal pergelangan, skala tangan.

    Menghilangkan perbedaan posisi/ukuran tangan antar peserta tanpa memakai
    statistik data uji. Bentuk keluaran tetap [N, 21, 3].
    """
    X = np.asarray(X, np.float32).copy()
    if tangan is not None and len(tangan) == len(X):
        kiri = np.array([str(t).strip().lower() in ("l", "left") for t in tangan], bool)
        X[kiri, :, 0] *= -1.0
    X = X - X[:, 0:1, :]
    skala = np.linalg.norm(X[:, 9, :2], axis=1)
    skala = np.where(skala < 1e-3, np.maximum(np.linalg.norm(X[:, :, :2], axis=2).max(axis=1), 1e-3), skala)
    return (X / skala[:, None, None]).astype(np.float32)


def muat_detail(
    folder: str | Path | None = None,
    lipat: str | None = None,
    latih: list[str] | None = None,
    validasi: list[str] | None = None,
    uji: list[str] | None = None,
    pose: bool = True,
) -> dict:
    data, meta = muat_mentah(folder)
    s = bagi(data, meta, lipat=lipat, latih=latih, validasi=validasi, uji=uji)
    tangan = data.get("tangan")

    Xtr, ytr = data["X"][s["latih"]], data["y"][s["latih"]]
    Xva, yva = data["X"][s["validasi"]], data["y"][s["validasi"]]
    Xte, yte = data["X"][s["uji"]], data["y"][s["uji"]]
    if pose:
        Xtr = normalkan_pose(Xtr, tangan[s["latih"]] if tangan is not None else None)
        Xva = normalkan_pose(Xva, tangan[s["validasi"]] if tangan is not None else None)
        Xte = normalkan_pose(Xte, tangan[s["uji"]] if tangan is not None else None)
    mu, sd = _fit_normalisasi(Xtr)
    hasil = {
        "X_latih": _terapkan(Xtr, mu, sd),
        "y_latih": ytr.astype(np.int64),
        "X_validasi": _terapkan(Xva, mu, sd),
        "y_validasi": yva.astype(np.int64),
        "X_uji": _terapkan(Xte, mu, sd),
        "X_pose_latih": Xtr,
        "X_pose_validasi": Xva,
        "X_pose_uji": Xte,
        "y_uji": yte.astype(np.int64),
        "peserta_latih": data["peserta"][s["latih"]],
        "peserta_validasi": data["peserta"][s["validasi"]],
        "peserta_uji": data["peserta"][s["uji"]],
        "kondisi_latih": data["kondisi"][s["latih"]],
        "kondisi_validasi": data["kondisi"][s["validasi"]],
        "kondisi_uji": data["kondisi"][s["uji"]],
        "frame_latih": data["frame"][s["latih"]],
        "frame_validasi": data["frame"][s["validasi"]],
        "frame_uji": data["frame"][s["uji"]],
        "normalisasi": {"mean": mu, "std": sd},
        "split": {"latih": s["peserta_latih"], "validasi": s["peserta_validasi"], "uji": s["peserta_uji"]},
        "meta": meta,
        "data": data,
    }
    return hasil


def muat(folder: str | Path | None = None, lipat: str | None = None):
    """Kontrak ``data.muat()``: (X_train, y_train, X_val, y_val, X_test, y_test)."""
    d = muat_detail(folder=folder, lipat=lipat)
    return (
        d["X_latih"], d["y_latih"],
        d["X_validasi"], d["y_validasi"],
        d["X_uji"], d["y_uji"],
    )


def info_split(folder: str | Path | None = None, lipat: str | None = None) -> dict:
    d = muat_detail(folder=folder, lipat=lipat)
    return {
        "split": d["split"],
        "n_latih": int(len(d["y_latih"])),
        "n_validasi": int(len(d["y_validasi"])),
        "n_uji": int(len(d["y_uji"])),
        "peserta_latih": sorted(set(d["peserta_latih"].tolist())),
        "peserta_validasi": sorted(set(d["peserta_validasi"].tolist())),
        "peserta_uji": sorted(set(d["peserta_uji"].tolist())),
        "n_kondisi_latih": int(len(set(d["kondisi_latih"].tolist()))),
        "n_kondisi_validasi": int(len(set(d["kondisi_validasi"].tolist()))),
        "n_kondisi_uji": int(len(set(d["kondisi_uji"].tolist()))),
    }


def hash_dataset(folder: str | Path | None = None) -> str:
    p = folder_sesi(folder) / DATASET
    h = hashlib.sha256()
    with p.open("rb") as f:
        for potong in iter(lambda: f.read(1 << 20), b""):
            h.update(potong)
    return h.hexdigest()[:32]
