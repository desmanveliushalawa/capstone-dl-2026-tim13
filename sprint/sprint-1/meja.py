"""meja.py — pemotong sesi simulasi Tim 13 (Sprint 1).

Contoh:
    python meja.py --simulasi jalankan ../rencana/t13-gestur-kamera.yaml
    python meja.py --simulasi periksa sesi_simulasi
    python meja.py --simulasi ringkas sesi_simulasi

Pemotong membaca tar rekaman langsung (tanpa ekstrak), mengekstraksi 21
landmark MediaPipe per frame, membuang 1 detik pertama tiap kondisi, lalu
menulis ``sesi_simulasi/dataset.npz`` + ``label.csv`` + ``sesi.json`` + ROI.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

AKAR = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(AKAR))
sys.path.insert(0, str(HERE))

from pemotong import potong  # noqa: E402


def cari_tar() -> Path:
    kandidat = sorted((AKAR / "data").glob("tim13_web-*.tar"))
    if not kandidat:
        raise FileNotFoundError("Tidak ada data/tim13_web-*.tar; berikan --tar secara eksplisit.")
    return kandidat[-1]


def _batas(teks: str | None) -> list[int] | None:
    if not teks:
        return None
    return [int(x) for x in teks.replace(";", ",").split(",") if x.strip()]


def jalankan(a) -> None:
    rencana = Path(a.target)
    if not rencana.exists() and not rencana.is_absolute():
        alternatif = Path(__file__).resolve().parents[1] / "rencana" / rencana.name
        if alternatif.exists():
            rencana = alternatif
    tar = Path(a.tar) if a.tar else cari_tar()
    keluaran = Path(a.keluaran) if a.keluaran else HERE / "sesi_simulasi"
    opsi = potong.Opsi(
        durasi=a.durasi,
        buang=a.buang,
        jeda=a.jeda,
        urutan=a.urutan,
        conf=a.conf,
        num_hands=a.num_hands,
        ukuran_roi=a.ukuran_roi,
        maks_frame=a.maks_frame,
        batas=_batas(a.batas),
        pakai_cache=not a.tanpa_cache,
        model=a.model,
        slot=a.slot,
    )
    potong.jalankan(rencana, tar, keluaran, opsi)


def periksa(a) -> None:
    from pemotong import periksa as modul

    folder = Path(a.target) if a.target else HERE / "sesi_simulasi"
    hasil = modul.periksa_sesi(folder)
    tujuan = Path(a.out) if a.out else HERE / "hasil" / "periksa_kebocoran.json"
    tujuan.parent.mkdir(parents=True, exist_ok=True)
    tujuan.write_text(json.dumps(hasil, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    gagal = [x for x in hasil["pemeriksaan"] if not x["lulus"]]
    print(f"Pemeriksaan: {len(hasil['pemeriksaan'])} butir, {len(gagal)} gagal")
    for x in gagal:
        print("  GAGAL:", x["nama"], x["detail"])
    print("LULUS" if hasil["lulus"] else "TIDAK LULUS")
    print("laporan ->", tujuan)
    raise SystemExit(0 if hasil["lulus"] else 1)


def ringkas(a) -> None:
    folder = Path(a.target) if a.target else HERE / "sesi_simulasi"
    meta = json.loads((folder / "sesi.json").read_text(encoding="utf-8"))
    print(json.dumps({
        k: meta.get(k) for k in (
            "peserta_rencana", "peserta_terdeteksi", "peserta_hilang", "interval_nominal_detik",
            "interval_terukur_detik", "jumlah_frame", "frame_terlihat", "jumlah_sampel",
            "sampel_per_peserta", "kondisi_terisi", "kondisi_total", "split_beku", "batas",
        )
    }, indent=2, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--simulasi", required=True, choices=["jalankan", "periksa", "ringkas"])
    ap.add_argument("target", nargs="?", help="YAML rencana (jalankan) atau folder sesi (periksa/ringkas)")
    ap.add_argument("--tar", default=None)
    ap.add_argument("--keluaran", default=None)
    ap.add_argument("--batas", default=None, help="indeks frame batas peserta, mis. 0,241,439,634,848,1049")
    ap.add_argument("--durasi", type=float, default=3.0)
    ap.add_argument("--buang", type=float, default=1.0, help="detik pertama tiap kondisi yang dibuang")
    ap.add_argument("--jeda", type=float, default=0.5)
    ap.add_argument("--slot", default="rencana", choices=["rencana", "auto"],
                    help="rencana = cadence durasi+jeda dijangkarkan ke frame terakhir; auto = bagi rata")
    ap.add_argument("--urutan", default="lampu-gestur-ulangan",
                    choices=["lampu-gestur-ulangan", "gestur-lampu-ulangan"])
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--num-hands", type=int, default=1)
    ap.add_argument("--ukuran-roi", type=int, default=224)
    ap.add_argument("--maks-frame", type=int, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--tanpa-cache", action="store_true")
    ap.add_argument("--out", default=None, help="berkas laporan untuk periksa")
    a = ap.parse_args()

    if a.simulasi == "jalankan":
        jalankan(a)
    elif a.simulasi == "periksa":
        periksa(a)
    else:
        ringkas(a)


if __name__ == "__main__":
    main()
