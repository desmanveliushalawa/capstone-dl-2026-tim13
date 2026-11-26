"""Pemeriksaan kebocoran data: peserta, kondisi, frame, dan normalisasi."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.protokol import sesi


def _cek(nama: str, lulus: bool, detail) -> dict:
    return {"nama": nama, "lulus": bool(lulus), "detail": detail}


def periksa_sesi(folder: str | Path | None = None) -> dict:
    data, meta = sesi.muat_mentah(folder)
    peserta = sesi.daftar_peserta(data)
    p = data["peserta"]
    frame = data["frame"]
    kondisi = data["kondisi"]
    y = data["y"].astype(int)

    laporan: list[dict] = []
    laporan.append(_cek("jumlah_peserta_minimal_3", len(peserta) >= 3, peserta))
    laporan.append(_cek("label_0_sampai_7", bool(y.min() >= 0 and y.max() <= 7),
                        {"min": int(y.min()), "max": int(y.max())}))
    laporan.append(_cek("tidak_ada_frame_ganda", len(set(frame.tolist())) == len(frame),
                        {"n_sampel": int(len(frame)), "n_frame_unik": int(len(set(frame.tolist())))}))
    laporan.append(_cek("setiap_kondisi_satu_peserta",
                        all(len(set(p[kondisi == k].tolist())) == 1 for k in set(kondisi.tolist())),
                        "kondisi = peserta + lampu + gestur + ulangan"))

    semua_mode: list[str | None] = [None] + peserta
    ringkas_split = []
    for mode in semua_mode:
        d = sesi.muat_detail(folder=folder, lipat=mode)
        label = "beku" if mode is None else f"lipat-{mode}"
        set_l = set(d["split"]["latih"])
        set_v = set(d["split"]["validasi"])
        set_u = set(d["split"]["uji"])
        kondisi_l = set(d["kondisi_latih"].tolist())
        kondisi_v = set(d["kondisi_validasi"].tolist())
        kondisi_u = set(d["kondisi_uji"].tolist())
        frame_l = set(d["frame_latih"].tolist())
        frame_v = set(d["frame_validasi"].tolist())
        frame_u = set(d["frame_uji"].tolist())

        laporan.append(_cek(f"{label}:peserta_latih_uji_terpisah", not (set_l & set_u), sorted(set_l & set_u)))
        laporan.append(_cek(f"{label}:peserta_latih_validasi_terpisah", not (set_l & set_v), sorted(set_l & set_v)))
        laporan.append(_cek(f"{label}:peserta_validasi_uji_terpisah", not (set_v & set_u), sorted(set_v & set_u)))
        laporan.append(_cek(f"{label}:kondisi_latih_uji_terpisah", not (kondisi_l & kondisi_u),
                            sorted(kondisi_l & kondisi_u)[:5]))
        laporan.append(_cek(f"{label}:kondisi_latih_validasi_terpisah", not (kondisi_l & kondisi_v),
                            sorted(kondisi_l & kondisi_v)[:5]))
        laporan.append(_cek(f"{label}:kondisi_validasi_uji_terpisah", not (kondisi_v & kondisi_u),
                            sorted(kondisi_v & kondisi_u)[:5]))
        laporan.append(_cek(f"{label}:frame_latih_uji_terpisah", not (frame_l & frame_u),
                            {"irisan": len(frame_l & frame_u)}))
        laporan.append(_cek(f"{label}:frame_latih_validasi_terpisah", not (frame_l & frame_v),
                            {"irisan": len(frame_l & frame_v)}))
        laporan.append(_cek(f"{label}:frame_validasi_uji_terpisah", not (frame_v & frame_u),
                            {"irisan": len(frame_v & frame_u)}))

        # Normalisasi harus persis mean/std data latih mentah (setelah normalisasi pose).
        mask_l = np.isin(data["peserta"], list(set_l))
        mask_v = np.isin(data["peserta"], list(set_v))
        mask_u = np.isin(data["peserta"], list(set_u))
        tangan = data.get("tangan")
        Xtr = sesi.normalkan_pose(data["X"][mask_l], tangan[mask_l] if tangan is not None else None)
        Xtr = Xtr.reshape(mask_l.sum(), -1).astype(np.float64)
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        mu_dipakai = np.asarray(d["normalisasi"]["mean"], np.float64)
        sd_dipakai = np.asarray(d["normalisasi"]["std"], np.float64)
        laporan.append(_cek(f"{label}:mean_dari_latih", bool(np.allclose(mu, mu_dipakai, atol=1e-5)),
                            {"maks_selisih": float(np.abs(mu - mu_dipakai).max())}))
        laporan.append(_cek(f"{label}:std_dari_latih", bool(np.allclose(sd, sd_dipakai, atol=1e-5)),
                            {"maks_selisih": float(np.abs(sd - sd_dipakai).max())}))
        Xv = sesi.normalkan_pose(data["X"][mask_v], tangan[mask_v] if tangan is not None else None)
        Xv = Xv.reshape(mask_v.sum(), -1).astype(np.float64)
        Xv_harap = ((Xv - mu_dipakai) / sd_dipakai).astype(np.float32)
        laporan.append(_cek(f"{label}:validasi_dinormalisasi_dengan_stat_latih",
                            bool(np.allclose(Xv_harap, d["X_validasi"].reshape(mask_v.sum(), -1), atol=1e-5)),
                            {"bentuk": list(d["X_validasi"].shape)}))
        Xu = sesi.normalkan_pose(data["X"][mask_u], tangan[mask_u] if tangan is not None else None)
        Xu = Xu.reshape(mask_u.sum(), -1).astype(np.float64)
        Xu_harap = ((Xu - mu_dipakai) / sd_dipakai).astype(np.float32)
        laporan.append(_cek(f"{label}:uji_dinormalisasi_dengan_stat_latih",
                            bool(np.allclose(Xu_harap, d["X_uji"].reshape(mask_u.sum(), -1), atol=1e-5)),
                            {"bentuk": list(d["X_uji"].shape)}))

        ringkas_split.append({
            "mode": label,
            "latih": {"peserta": sorted(set_l), "n": int(len(d["y_latih"])), "kondisi": len(kondisi_l)},
            "validasi": {"peserta": sorted(set_v), "n": int(len(d["y_validasi"])), "kondisi": len(kondisi_v)},
            "uji": {"peserta": sorted(set_u), "n": int(len(d["y_uji"])), "kondisi": len(kondisi_u)},
        })

    lulus = all(x["lulus"] for x in laporan)
    return {
        "folder": str(sesi.folder_sesi(folder)),
        "hash_dataset": sesi.hash_dataset(folder),
        "n_sampel": int(len(y)),
        "peserta": peserta,
        "split": ringkas_split,
        "pemeriksaan": laporan,
        "lulus": lulus,
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", nargs="?", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    hasil = periksa_sesi(a.folder)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(hasil, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    gagal = [x for x in hasil["pemeriksaan"] if not x["lulus"]]
    print(f"Pemeriksaan: {len(hasil['pemeriksaan'])} butir, {len(gagal)} gagal")
    for x in gagal:
        print("  GAGAL:", x["nama"], x["detail"])
    print("LULUS" if hasil["lulus"] else "TIDAK LULUS")
    raise SystemExit(0 if hasil["lulus"] else 1)


if __name__ == "__main__":
    main()
