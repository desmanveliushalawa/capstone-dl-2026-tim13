"""Pemotong sesi: rencana -> slot kondisi -> sampel landmark + ROI."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import deteksi, rekaman, rencana


@dataclass
class Opsi:
    durasi: float = 3.0
    buang: float = 1.0
    jeda: float = 0.5
    urutan: str = "lampu-gestur-ulangan"
    conf: float = 0.1
    num_hands: int = 1
    ukuran_roi: int = 224
    maks_frame: int | None = None
    batas: list[int] | None = None
    pakai_cache: bool = True
    model: str | None = None
    slot: str = "rencana"  # "rencana" = cadence durasi+jeda, jangkar frame terakhir; "auto" = bagi rata


def _deteksi_semua(
    tar_path: Path,
    frames: list[rekaman.Frame],
    terlihat: np.ndarray,
    opsi: Opsi,
    cache: Path,
    n: int,
) -> dict[str, np.ndarray]:
    if opsi.pakai_cache and cache.exists():
        z = np.load(cache, allow_pickle=False)
        if int(z["landmark"].shape[0]) == n:
            print(f"  memakai cache landmark: {cache.name}", flush=True)
            return {k: z[k] for k in z.files}

    landmark = np.full((n, 21, 3), np.nan, np.float32)
    bbox = np.full((n, 4), np.nan, np.float32)
    skor = np.full(n, np.nan, np.float32)
    tangan = np.array([""] * n, dtype="<U8")
    valid = np.zeros(n, bool)
    t0 = frames[0].t_host

    print(f"  ekstraksi landmark {int(terlihat[:n].sum())} frame terlihat ...", flush=True)
    with tarfile.open(tar_path) as tf, deteksi.DetektorTangan(
        model_path=opsi.model, conf=opsi.conf, num_hands=opsi.num_hands, mode="VIDEO"
    ) as det:
        diproses = 0
        for i in range(n):
            if not terlihat[i]:
                continue
            img = rekaman.baca_citra(tf, frames[i].nama)
            lm, bb, sk, tg = det.deteksi(img, int(round((frames[i].t_host - t0) * 1000)))
            diproses += 1
            if lm is not None:
                landmark[i] = lm
                bbox[i] = bb
                skor[i] = sk
                tangan[i] = tg
                valid[i] = True
            if diproses % 100 == 0:
                print(f"    {diproses} frame", flush=True)

    np.savez_compressed(
        cache,
        landmark=landmark,
        bbox=bbox,
        skor=skor,
        tangan=tangan,
        valid=valid,
        t_host=np.array([f.t_host for f in frames[:n]], np.float64),
        nama=np.array([f.nama for f in frames[:n]]),
    )
    print(f"  cache landmark -> {cache.name}", flush=True)
    return {
        "landmark": landmark,
        "bbox": bbox,
        "skor": skor,
        "tangan": tangan,
        "valid": valid,
        "t_host": np.array([f.t_host for f in frames[:n]], np.float64),
        "nama": np.array([f.nama for f in frames[:n]]),
    }


def _hash_berkas(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for potong in iter(lambda: f.read(1 << 20), b""):
            h.update(potong)
    return h.hexdigest()[:32]


def jalankan(
    plan_path: str | Path,
    tar_path: str | Path,
    out_dir: str | Path,
    opsi: Opsi | None = None,
) -> dict:
    opsi = opsi or Opsi()
    plan_path, tar_path, out_dir = Path(plan_path), Path(tar_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rencana_data = rencana.baca(plan_path)
    info = rencana.info(rencana_data)
    kondisi = rencana.kondisi_urut(rencana_data, opsi.urutan)
    durasi = opsi.durasi or info["durasi"]
    buang = opsi.buang
    jeda = opsi.jeda if opsi.jeda is not None else info["jeda"]

    print("[1/5] membaca indeks rekaman", flush=True)
    frames = rekaman.baca_frames(tar_path, opsi.maks_frame)
    print(f"  {len(frames)} frame, interval terukur {rekaman.interval_terukur(frames):.3f} s", flush=True)

    print("[2/5] profil gerak & batas peserta", flush=True)
    profil = rekaman.hitung_profil(tar_path, frames)
    n_vis = int(np.flatnonzero(profil["terlihat"])[-1]) + 1
    if n_vis < len(frames):
        print(f"  frame gelap/tidak terlihat setelah indeks {n_vis - 1} diabaikan", flush=True)
    if opsi.batas:
        batas = sorted(set(int(x) for x in opsi.batas))
    else:
        batas = rekaman.deteksi_batas(profil, maks_batas=max(0, len(info["peserta"]) - 1))
        print(f"  batas terdeteksi otomatis: {batas}", flush=True)
    titik = [0] + [b for b in batas if 0 < b < n_vis] + [n_vis]
    titik = sorted(set(titik))
    if len(titik) < 2:
        raise RuntimeError("Tidak ada blok peserta yang bisa dipotong.")
    print(f"  blok peserta: {len(titik) - 1}", flush=True)

    cache = out_dir / "landmark_mentah.npz"
    print("[3/5] ekstraksi 21 landmark MediaPipe", flush=True)
    det = _deteksi_semua(tar_path, frames, profil["terlihat"], opsi, cache, n_vis)
    valid = det["valid"]
    landmark = det["landmark"]
    bbox = det["bbox"]
    t = det["t_host"]

    kode_peserta = list(info["peserta"]) or [f"P{i + 1:02d}" for i in range(len(titik) - 1)]
    if len(titik) - 1 > len(kode_peserta):
        kode_peserta += [f"P{i + 1:02d}" for i in range(len(kode_peserta), len(titik) - 1)]

    roi_dir = out_dir / "roi"
    if roi_dir.exists():
        shutil.rmtree(roi_dir)
    roi_dir.mkdir(parents=True)
    baris_label: list[dict] = []
    sampel: list[dict] = []
    blok_info: list[dict] = []

    print("[4/5] memotong kondisi (buang 1 detik pertama)", flush=True)
    with tarfile.open(tar_path) as tf:
        for b in range(len(titik) - 1):
            a, z = titik[b], titik[b + 1]
            peserta = kode_peserta[b]
            idx_valid = np.flatnonzero(valid[a:z])
            if len(idx_valid) == 0:
                print(f"  {peserta}: tidak ada tangan terdeteksi, dilewati", flush=True)
                blok_info.append({"peserta": peserta, "frame": [a, z], "aktif": None, "cadence": None})
                for k in kondisi:
                    baris_label.append({
                        "peserta": peserta, "kondisi": rencana.id_kondisi(peserta, k),
                        "gestur": k["gestur"], "nama_gestur": k["nama_gestur"], "lampu": k["lampu"],
                        "ulangan": k["ulangan"], "slot_mulai": "", "slot_selesai": "",
                        "n_frame": 0, "n_sampel": 0, "roi": "",
                    })
                continue
            f = a + int(idx_valid[0])
            l = a + int(idx_valid[-1])
            f = max(a, f - 1)
            l = min(z - 1, l + 1)
            cad_auto = (l - f + 1) / len(kondisi)
            if opsi.slot == "rencana" and jeda > 0:
                cad = durasi + jeda
                t_akhir = t[l]
                t_mulai = t_akhir - ((len(kondisi) - 1) * cad + durasi)
                if t_mulai < t[a]:
                    cad, t_mulai, mode_slot = cad_auto, t[f], "auto (span kurang)"
                else:
                    mode_slot = "rencana"
            else:
                cad, t_mulai, mode_slot = cad_auto, t[f], "auto"
            print(f"  {peserta}: frame {a}-{z - 1}, aktif {f}-{l}, "
                  f"slot {mode_slot} {cad:.2f} s mulai {t_mulai - t[0]:.1f} s", flush=True)

            n_sampel_blok = 0
            for k_idx, k in enumerate(kondisi):
                kid = rencana.id_kondisi(peserta, k)
                mulai = t_mulai + k_idx * cad
                akhir = mulai + durasi if mode_slot == "rencana" else min(mulai + durasi, mulai + cad)
                if akhir - mulai <= buang:
                    sel = np.array([], dtype=int)
                else:
                    sel = np.flatnonzero(
                        valid & (t >= mulai + buang) & (t < akhir) & (np.arange(len(t)) >= a) & (np.arange(len(t)) < z)
                    )
                n_frame = int(len(sel))

                luas = np.array([np.prod(bbox[i][2:] - bbox[i][:2]) if valid[i] else 0.0 for i in sel]) if n_frame else np.array([])
                i_roi = int(sel[int(np.argmax(luas))]) if n_frame else None
                jalur_roi = ""
                if i_roi is not None:
                    img = rekaman.baca_citra(tf, frames[i_roi].nama)
                    crop, _ = deteksi.potong_roi(img, landmark[i_roi], ukuran=opsi.ukuran_roi)
                    folder = roi_dir / peserta
                    folder.mkdir(parents=True, exist_ok=True)
                    tujuan = folder / f"{kid}.jpg"
                    cv2.imwrite(str(tujuan), crop)
                    jalur_roi = str(tujuan.relative_to(out_dir)).replace("\\", "/")

                if n_frame:
                    for i in sel:
                        sampel.append({
                            "X": landmark[i], "y": int(k["gestur"]), "peserta": peserta,
                            "kondisi": kid, "frame": int(i), "t_host": float(t[i]),
                            "gestur": int(k["gestur"]), "lampu": int(k["lampu"]), "ulangan": int(k["ulangan"]),
                            "bbox": bbox[i],
                        })
                    n_sampel_blok += n_frame

                baris_label.append({
                    "peserta": peserta, "kondisi": kid, "gestur": k["gestur"], "nama_gestur": k["nama_gestur"],
                    "lampu": k["lampu"], "ulangan": k["ulangan"],
                    "slot_mulai": round(mulai, 3), "slot_selesai": round(akhir, 3),
                    "n_frame": n_frame, "n_sampel": n_frame, "roi": jalur_roi,
                })
            blok_info.append({
                "peserta": peserta, "frame": [a, z], "aktif": [f, l + 1],
                "mode_slot": mode_slot, "cadence": round(cad, 4),
                "slot_mulai_detik": round(t_mulai - t[0], 3),
                "slot_selesai_detik": round(t_mulai - t[0] + len(kondisi) * cad, 3),
                "n_sampel": n_sampel_blok,
            })

    if not sampel:
        raise RuntimeError("Tidak ada sampel yang lolos pemotongan. Periksa batas/pilihan frame.")

    print("[5/5] menyimpan dataset sesi", flush=True)
    X = np.stack([s["X"] for s in sampel]).astype(np.float32)
    y = np.array([s["y"] for s in sampel], np.int8)
    peserta_arr = np.array([s["peserta"] for s in sampel])
    kondisi_arr = np.array([s["kondisi"] for s in sampel])
    frame_arr = np.array([s["frame"] for s in sampel], np.int32)
    t_arr = np.array([s["t_host"] for s in sampel], np.float64)
    bbox_arr = np.stack([s["bbox"] for s in sampel]).astype(np.float32)
    gestur_arr = np.array([s["gestur"] for s in sampel], np.int8)
    lampu_arr = np.array([s["lampu"] for s in sampel], np.int16)
    ulangan_arr = np.array([s["ulangan"] for s in sampel], np.int8)
    tangan_arr = np.array([det["tangan"][s["frame"]] for s in sampel])

    np.savez_compressed(
        out_dir / "dataset.npz",
        X=X, y=y, peserta=peserta_arr, kondisi=kondisi_arr, frame=frame_arr, t_host=t_arr,
        bbox=bbox_arr, gestur=gestur_arr, lampu=lampu_arr, ulangan=ulangan_arr, tangan=tangan_arr,
        versi=np.array(1, np.int16),
    )
    with (out_dir / "label.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(baris_label[0].keys()))
        w.writeheader()
        w.writerows(baris_label)

    peserta_ada = sorted(set(peserta_arr.tolist()))
    if len(peserta_ada) >= 3:
        split = {"latih": peserta_ada[:-2], "validasi": [peserta_ada[-2]], "uji": [peserta_ada[-1]]}
    else:
        raise RuntimeError(
            f"Perlu minimal 3 peserta untuk split latih/validasi/uji yang terpisah; terdeteksi {peserta_ada}."
        )

    sampel_per_peserta = {p: int((peserta_arr == p).sum()) for p in peserta_ada}
    kondisi_terisi = sum(1 for r in baris_label if int(r["n_sampel"]) > 0)
    sesi = {
        "versi": 1,
        "kasus": info["kasus"],
        "rencana": str(plan_path),
        "sumber_tar": str(tar_path),
        "resolusi": info["resolusi"],
        "interval_nominal_detik": info["interval_nominal"],
        "interval_terukur_detik": round(rekaman.interval_terukur(frames), 4),
        "jumlah_frame": len(frames),
        "frame_terlihat": n_vis,
        "peserta_rencana": info["jumlah_peserta"],
        "peserta_terdeteksi": len(peserta_ada),
        "peserta_hilang": info["peserta"][len(peserta_ada):],
        "batas": titik,
        "blok": blok_info,
        "kondisi_per_peserta": len(kondisi),
        "urutan_kondisi": opsi.urutan,
        "mode_slot": opsi.slot,
        "durasi_detik": durasi,
        "buang_awal_detik": buang,
        "jeda_rencana_detik": jeda,
        "deteksi": {
            "model": "hand_landmarker.task",
            "api": "mediapipe (tasks/solutions)",
            "mode": "VIDEO",
            "confidence": opsi.conf,
            "num_hands": opsi.num_hands,
        },
        "split_beku": split,
        "normalisasi": "mean/std dihitung dari data latih saja",
        "jumlah_sampel": int(len(y)),
        "sampel_per_peserta": sampel_per_peserta,
        "kondisi_terisi": int(kondisi_terisi),
        "kondisi_total": len(baris_label),
        "distribusi_kelas": {str(c): int((y == c).sum()) for c in range(info["jumlah_gestur"])},
        "roi": str(roi_dir.relative_to(out_dir)).replace("\\", "/"),
        "hash_dataset": _hash_berkas(out_dir / "dataset.npz"),
        "catatan": [
            "Rekaman kamera web berjarak 1 detik/frame (selangMs=1000), bukan 0,1 detik seperti rencana.",
            "Slot kondisi dibagi rata pada rentang aktif tiap peserta (deteksi tangan pertama-terakhir).",
            "Label gestur mengikuti urutan rencana; rekaman tidak menyertakan label.csv sesi.",
            "Frame tanpa deteksi tangan tidak diisi/interpolasi.",
            "Kode peserta P01-P05 adalah urutan blok pada rekaman web ini, bukan identitas partisipan terverifikasi; sesi aktual tim memakai kode P01, P02, P04, P05, P06.",
        ],
    }
    (out_dir / "sesi.json").write_text(json.dumps(sesi, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\nRingkasan sesi")
    print(f"  sampel          : {len(y)}")
    print(f"  peserta         : {peserta_ada}")
    print(f"  sampel/peserta  : {sampel_per_peserta}")
    print(f"  kondisi terisi  : {kondisi_terisi}/{len(baris_label)}")
    print(f"  split beku      : {split}")
    print(f"  dataset         : {out_dir / 'dataset.npz'}")
    return sesi
