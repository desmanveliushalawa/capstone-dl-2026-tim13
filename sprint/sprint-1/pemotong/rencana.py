"""Membaca rencana meja uji (YAML) dan menyusun urutan 48 kondisi per peserta."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import yaml

URUTAN_SAH = ("lampu-gestur-ulangan", "gestur-lampu-ulangan")


def baca(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Rencana tidak ditemukan: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "protokol_eksperimen" not in data:
        raise ValueError(f"{p} bukan rencana meja uji Tim 13 (kunci 'protokol_eksperimen' tidak ada).")
    return data


def info(rencana: dict) -> dict:
    pe = rencana["protokol_eksperimen"]
    return {
        "kasus": rencana.get("metadata", {}).get("kasus", ""),
        "peserta": list(pe["partisipan"]["daftar_kode"]),
        "jumlah_peserta": int(pe["partisipan"]["jumlah"]),
        "gestur": {int(k): str(v) for k, v in pe["gestur"]["daftar_kelas"].items()},
        "jumlah_gestur": int(pe["gestur"]["jumlah_kelas"]),
        "lampu": [int(x) for x in pe["kondisi_pengujian"]["taraf_lampu_pwm"]],
        "ulangan": int(pe["kondisi_pengujian"]["jumlah_ulangan"]),
        "kondisi_per_blok": int(pe["kondisi_pengujian"]["kondisi_per_blok"]),
        "durasi": float(pe["gestur"]["durasi_per_kondisi"]),
        "jeda": float(pe["gestur"]["jeda_antar_kondisi"]),
        "interval_nominal": 0.1,
        "resolusi": str(pe["modalitas_data"]["kamera"]["resolusi"]),
    }


def kondisi_urut(rencana: dict, urutan: str = "lampu-gestur-ulangan") -> list[dict]:
    """Kembalikan 48 kondisi dalam urutan yang dipakai simulator.

    Urutan bawaan adalah lampu -> gestur -> ulangan, supaya satu taraf lampu
    diselesaikan lebih dulu (8 gestur x 3 ulangan = 24 kondisi) sebelum lampu
    diganti. Urutan ini ditulis di sesi.json agar dapat direproduksi.
    """
    if urutan not in URUTAN_SAH:
        raise ValueError(f"urutan harus salah satu dari {URUTAN_SAH}, bukan {urutan!r}")
    pe = rencana["protokol_eksperimen"]
    gestur = {int(k): str(v) for k, v in pe["gestur"]["daftar_kelas"].items()}
    lampu = [int(x) for x in pe["kondisi_pengujian"]["taraf_lampu_pwm"]]
    ulangan = range(1, int(pe["kondisi_pengujian"]["jumlah_ulangan"]) + 1)

    hasil: list[dict] = []
    if urutan == "lampu-gestur-ulangan":
        for lamp in lampu:
            for g in sorted(gestur):
                for u in ulangan:
                    hasil.append({"gestur": g, "nama_gestur": gestur[g], "lampu": lamp, "ulangan": u})
    else:
        for g in sorted(gestur):
            for lamp in lampu:
                for u in ulangan:
                    hasil.append({"gestur": g, "nama_gestur": gestur[g], "lampu": lamp, "ulangan": u})

    harus = int(pe["kondisi_pengujian"]["kondisi_per_blok"])
    if len(hasil) != harus:
        raise ValueError(f"urutan menghasilkan {len(hasil)} kondisi, rencana menulis {harus}.")
    return hasil


def id_kondisi(peserta: str, k: dict) -> str:
    return f"{peserta}-L{k['lampu']:03d}-G{k['gestur']:02d}-U{k['ulangan']:02d}"
