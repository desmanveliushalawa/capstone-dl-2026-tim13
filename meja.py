#!/usr/bin/env python3
"""Simulator sesi meja uji Tim 13 (mode tanpa perangkat keras).

Contoh pemakaian (dari akar repo):

    python meja.py --simulasi jalankan ../rencana/t13-gestur-kamera.yaml

Skrip ini:
1. membaca rencana YAML,
2. memvalidasi 8 partisipan x 8 gestur x 2 taraf lampu x 3 ulangan = 384 kondisi,
3. mencetak "384 kondisi",
4. membuat folder sesi simulasi berisi label.csv 384 baris, bahan slot, dan
   panduan latihan kering,
5. menjalankan latihan kering tanpa meja (tanpa kamera/ESP32/lampu),
6. menyalin folder sesi + zip ke folder bagikan untuk Penjaga Data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import yaml

AKAR = Path(__file__).resolve().parent
BASE_WAKTU = datetime(2026, 11, 12, 11, 0, 0)
DURASI_GESTUR = 3.0
JEDA = 0.5

DESKRIPSI_GESTUR = {
    "Tangan terbuka": "Telapak datar menghadap kamera, jari-jari merapat.",
    "Kepalan": "Jari-jari menggenggam rapat, punggung tangan menghadap kamera.",
    "Jempol ke atas": "Kepalan dengan ibu jari menunjuk ke atas.",
    "Jempol ke bawah": "Kepalan dengan ibu jari menunjuk ke bawah.",
    "Telunjuk ke atas": "Telunjuk menunjuk vertikal, jari lain menggenggam.",
    "Dua jari (V)": "Telunjuk dan jari tengah membentuk huruf V.",
    "Tanda OK": "Ujung telunjuk menyentuh ujung ibu jari, jari lain terbuka.",
    "Telapak menghadap kamera lalu turun": "Telapak menghadap kamera, lalu diturunkan perlahan.",
}


def baca_rencana(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"[meja] rencana tidak ditemukan: {path}")
    with path.open(encoding="utf-8") as berkas:
        data = yaml.safe_load(berkas)
    if not isinstance(data, dict):
        raise SystemExit("[meja] isi YAML bukan objek")
    return data


def ambil_protokol(data: dict) -> dict:
    try:
        protokol = data["protokol_eksperimen"]
        peserta = [str(x) for x in protokol["partisipan"]["daftar_kode"]]
        kelas = protokol["gestur"]["daftar_kelas"]
        gestur = [str(kelas[k]) for k in sorted(kelas, key=lambda x: int(x))]
        lampu = [int(x) for x in protokol["kondisi_pengujian"]["taraf_lampu_pwm"]]
        ulangan = int(protokol["kondisi_pengujian"]["jumlah_ulangan"])
        per_blok = int(protokol["kondisi_pengujian"]["kondisi_per_blok"])
        total = int(protokol["kondisi_pengujian"]["total_kondisi_meja"])
    except KeyError as galat:
        raise SystemExit(f"[meja] kunci YAML hilang: {galat}") from galat

    hitung_per_blok = len(gestur) * len(lampu) * ulangan
    hitung_total = len(peserta) * hitung_per_blok
    if hitung_per_blok != per_blok:
        raise SystemExit(
            f"[meja] kondisi per blok tidak konsisten: {hitung_per_blok} != {per_blok}"
        )
    if hitung_total != total:
        raise SystemExit(
            f"[meja] total kondisi tidak konsisten: {hitung_total} != {total}"
        )
    if total != 384:
        raise SystemExit(f"[meja] total kondisi harus 384, bukan {total}")

    return {
        "peserta": peserta,
        "gestur": gestur,
        "lampu": lampu,
        "ulangan": ulangan,
        "per_blok": per_blok,
        "total": total,
        "slot": data.get("metadata", {}).get("slot_jadwal", ""),
        "lokasi": data.get("metadata", {}).get("lokasi", ""),
        "pengelola": data.get("metadata", {}).get("pengelola_meja", ""),
    }


def buat_baris(protokol: dict) -> list[dict]:
    baris = []
    nomor = 0
    for blok, peserta in enumerate(protokol["peserta"], start=1):
        for gestur in protokol["gestur"]:
            for lampu in protokol["lampu"]:
                for ulangan in range(1, protokol["ulangan"] + 1):
                    nomor += 1
                    mulai = BASE_WAKTU + timedelta(
                        seconds=(nomor - 1) * (DURASI_GESTUR + JEDA)
                    )
                    selesai = mulai + timedelta(seconds=DURASI_GESTUR)
                    baris.append(
                        {
                            "no": nomor,
                            "ulangan_blok": blok,
                            "blok": blok,
                            "ulangan": ulangan,
                            "partisipan": peserta,
                            "gestur": gestur,
                            "lampu": lampu,
                            "t_mulai_host": f"{mulai.timestamp():.4f}",
                            "t_selesai_host": f"{selesai.timestamp():.4f}",
                            "mulai_iso": mulai.isoformat(timespec="milliseconds"),
                            "pintu_awal": "",
                            "pintu_akhir": "",
                            "suhu_C": f"{22.5 + 0.1 * ((nomor % 5) - 2):.1f}",
                            "lux": "245.0" if lampu == 128 else "66.7",
                        }
                    )
    return baris


def tulis_label(path: Path, baris: list[dict]) -> None:
    kolom = [
        "no",
        "ulangan_blok",
        "blok",
        "ulangan",
        "partisipan",
        "gestur",
        "lampu",
        "t_mulai_host",
        "t_selesai_host",
        "mulai_iso",
        "pintu_awal",
        "pintu_akhir",
        "suhu_C",
        "lux",
    ]
    with path.open("w", encoding="utf-8", newline="") as berkas:
        penulis = csv.DictWriter(berkas, fieldnames=kolom)
        penulis.writeheader()
        penulis.writerows(baris)


def latihan_kering(baris: list[dict], protokol: dict) -> tuple[bool, list[str]]:
    cek: list[tuple[str, bool, str]] = []

    def tambah(nama: str, lulus: bool, detail: str) -> None:
        cek.append((nama, bool(lulus), detail))

    tambah("jumlah total", len(baris) == 384, f"{len(baris)} kondisi")

    per_peserta = Counter(r["partisipan"] for r in baris)
    tambah(
        "setiap partisipan 48 kondisi",
        all(per_peserta[p] == 48 for p in protokol["peserta"]),
        ", ".join(f"{p}={per_peserta[p]}" for p in protokol["peserta"]),
    )

    per_gestur = Counter(r["gestur"] for r in baris)
    tambah(
        "setiap gestur 48 kondisi",
        all(per_gestur[g] == 48 for g in protokol["gestur"]),
        ", ".join(f"{g}={per_gestur[g]}" for g in protokol["gestur"]),
    )

    per_lampu = Counter(r["lampu"] for r in baris)
    tambah(
        "taraf lampu seimbang",
        all(per_lampu[l] == 192 for l in protokol["lampu"]),
        ", ".join(f"PWM {l}={per_lampu[l]}" for l in protokol["lampu"]),
    )

    per_ulangan = Counter(r["ulangan"] for r in baris)
    tambah(
        "ulangan seimbang",
        all(per_ulangan[u] == 128 for u in range(1, protokol["ulangan"] + 1)),
        ", ".join(f"ulangan {u}={per_ulangan[u]}" for u in range(1, protokol["ulangan"] + 1)),
    )

    unik = {
        (r["partisipan"], r["gestur"], r["lampu"], r["ulangan"]) for r in baris
    }
    tambah("tidak ada kondisi duplikat", len(unik) == len(baris), f"{len(unik)} unik")

    jeda_ok = True
    for sebelum, sesudah in zip(baris, baris[1:]):
        selisih = float(sesudah["t_mulai_host"]) - float(sebelum["t_selesai_host"])
        if selisih < JEDA - 0.01:
            jeda_ok = False
            break
    tambah("jeda antar kondisi >= 0.5 detik", jeda_ok, f"target {JEDA} detik")

    durasi_ok = all(
        abs(float(r["t_selesai_host"]) - float(r["t_mulai_host"]) - DURASI_GESTUR) < 0.01
        for r in baris
    )
    tambah("durasi gestur 3 detik", durasi_ok, f"target {DURASI_GESTUR} detik")

    tanpa_keras = True
    tambah(
        "tanpa kamera/ESP32/lampu",
        tanpa_keras,
        "latihan kering hanya memakai urutan CSV dan timer",
    )

    lulus = all(item[1] for item in cek)
    log = [
        "LATIHAN KERING TANPA MEJA - TIM 13",
        f"waktu: {datetime.now().isoformat(timespec='seconds')}",
        f"slot: {protokol['slot']}",
        f"lokasi: {protokol['lokasi']}",
        "",
    ]
    for nama, status, detail in cek:
        log.append(f"[{'LULUS' if status else 'GAGAL'}] {nama}: {detail}")
    log.extend(["", f"LATIHAN KERING: {'LULUS' if lulus else 'GAGAL'}", ""])
    return lulus, log


def tulis_readme(path: Path, sesi: Path, protokol: dict, jumlah: int) -> None:
    teks = f"""# Sesi Simulasi Meja Uji - Tim 13

Folder ini dihasilkan oleh:

```bash
python meja.py --simulasi jalankan ../rencana/t13-gestur-kamera.yaml
```

- Slot: {protokol['slot']}
- Lokasi: {protokol['lokasi']}
- Pengelola meja: {protokol['pengelola']}
- Total kondisi: **{jumlah} kondisi**
- Status latihan kering: lihat `latihan_kering.log`

## Isi folder

| Berkas | Keterangan |
|---|---|
| `label.csv` | 384 kondisi simulasi (partisipan, gestur, lampu, ulangan, waktu) |
| `ringkasan_sesi.json` | Rekap peserta, gestur, lampu, dan hash rencana |
| `latihan_kering.log` | Bukti latihan kering tanpa meja |
| `bahan_slot/` | Bahan yang dibawa ke meja uji |

## Cara memakai

1. Buka `bahan_slot/CHECKLIST_BAHAN_SLOT.md` dan centang bahan yang sudah siap.
2. Cetak `bahan_slot/KARTU_GESTUR.md` dan `bahan_slot/LEMBAR_SESI.md`.
3. Lakukan `bahan_slot/PANDUAN_LATIHAN_KERING.md` sebelum masuk slot.
4. Bagikan folder ini ke Penjaga Data (Bastian Heskia Silaban, 122450130).

> Data di folder ini adalah simulasi. Jangan dipakai sebagai data hasil
> eksperimen; gunakan hanya untuk gladi dan kesiapan slot.
"""
    path.write_text(teks, encoding="utf-8")


def tulis_ringkasan(
    path: Path, protokol: dict, baris: list[dict], rencana: Path
) -> None:
    sha = hashlib.sha256(rencana.read_bytes()).hexdigest()
    mulai = datetime.fromisoformat(baris[0]["mulai_iso"])
    selesai = datetime.fromisoformat(baris[-1]["mulai_iso"]) + timedelta(
        seconds=DURASI_GESTUR
    )
    isi = {
        "mode": "simulasi",
        "rencana": str(rencana),
        "sha256_rencana": sha,
        "slot": protokol["slot"],
        "lokasi": protokol["lokasi"],
        "pengelola_meja": protokol["pengelola"],
        "peserta": protokol["peserta"],
        "gestur": protokol["gestur"],
        "taraf_lampu_pwm": protokol["lampu"],
        "ulangan": protokol["ulangan"],
        "kondisi_per_blok": protokol["per_blok"],
        "total_kondisi": protokol["total"],
        "mulai_simulasi": mulai.isoformat(timespec="seconds"),
        "selesai_simulasi": selesai.isoformat(timespec="seconds"),
        "durasi_estimasi_menit": round(
            (protokol["total"] * (DURASI_GESTUR + JEDA)) / 60, 1
        ),
        "catatan": (
            "Nilai lux, suhu, dan pintu adalah simulasi. Latihan kering tidak "
            "memakai kamera, ESP32, atau lampu."
        ),
    }
    path.write_text(json.dumps(isi, indent=2, ensure_ascii=False), encoding="utf-8")


def tulis_kartu(path: Path, protokol: dict) -> None:
    bagian = ["# Kartu Gestur - Tim 13", "", "Cetak dan tunjukkan sebelum blok dimulai.", ""]
    for indeks, gestur in enumerate(protokol["gestur"]):
        deskripsi = DESKRIPSI_GESTUR.get(gestur, "Tahan gestur sesuai kartu.")
        bagian.extend(
            [
                f"## Kartu {indeks}: {gestur}",
                "",
                f"**Gerakan:** {deskripsi}",
                "",
                "**Durasi tahan:** 3 detik. **Turunkan tangan:** 0,5 detik.",
                "",
                "---",
                "",
            ]
        )
    path.write_text("\n".join(bagian), encoding="utf-8")


def tulis_lembar(path: Path, protokol: dict) -> None:
    bagian = [
        "# Lembar Sesi Simulasi - Tim 13",
        "",
        f"**Slot:** {protokol['slot']}  ",
        f"**Lokasi:** {protokol['lokasi']}  ",
        f"**Pengelola meja:** {protokol['pengelola']}",
        "",
        "| Kode | Nama | NIM / Prodi | Tangan (Kanan/Kiri) | 48 kondisi? | Catatan | Tanda tangan |",
        "|---|---|---|---|---|---|---|",
    ]
    for peserta in protokol["peserta"]:
        bagian.append(
            f"| **{peserta}** | | | [ ] Kanan / [ ] Kiri | [ ] Lengkap | | |"
        )
    bagian.extend(
        [
            "",
            "Total kondisi: **384** (8 partisipan x 8 gestur x 2 lampu x 3 ulangan).",
            "",
        ]
    )
    path.write_text("\n".join(bagian), encoding="utf-8")


def tulis_checklist(path: Path) -> None:
    teks = """# Checklist Bahan Slot - Tim 13

Centang sebelum berangkat ke meja uji.

## Perangkat
- [ ] Laptop perekam + charger
- [ ] Kamera USB 640x480 + kabel
- [ ] ESP32 + sensor meja + kabel USB
- [ ] Lampu meja 5V + driver PWM
- [ ] Timer/stopwatch (atau ponsel)
- [ ] Kain background/netral

## Bahan cetak
- [ ] 8 kartu gestur (`KARTU_GESTUR.md`)
- [ ] Lembar sesi (`LEMBAR_SESI.md`)
- [ ] Checklist ini
- [ ] Panduan latihan kering (`PANDUAN_LATIHAN_KERING.md`)
- [ ] Alat tulis + papan jalan

## Berkas
- [ ] `label.csv` (384 kondisi)
- [ ] `ringkasan_sesi.json`
- [ ] `latihan_kering.log` sudah LULUS
- [ ] Rencana YAML `t13-gestur-kamera.yaml`
- [ ] Folder sesi sudah dibagikan ke Penjaga Data

## Partisipan
- [ ] P01-P08 sudah dikonfirmasi hadir
- [ ] Penjelasan gestur dan tangan dominan sudah disampaikan
- [ ] Izin/daftar hadir partisipan siap

## Ruang
- [ ] Slot Ruang Prodi 1 dikonfirmasi ke Pak Ardika Satria
- [ ] Meja, kursi, dan posisi kamera sudah diuji
- [ ] Lampu PWM 128 dan 255 sudah dicek
"""
    path.write_text(teks, encoding="utf-8")


def tulis_panduan(path: Path, protokol: dict, baris: list[dict]) -> None:
    contoh = [r for r in baris if r["partisipan"] == protokol["peserta"][0]]
    bagian = [
        "# Panduan Latihan Kering Tanpa Meja - Tim 13",
        "",
        "Latihan ini dilakukan **tanpa meja, kamera, ESP32, atau lampu**. "
        "Tujuannya melatih urutan, timing, dan pembacaan kartu.",
        "",
        "## Langkah",
        "",
        "1. Siapkan 8 kartu gestur dan timer.",
        "2. Operator membacakan urutan 48 kondisi untuk satu partisipan.",
        "3. Partisipan menahan gestur 3 detik, lalu menurunkan tangan 0,5 detik.",
        "4. Operator mencatat tangan dominan dan kendala di lembar sesi.",
        "5. Ulangi untuk seluruh partisipan; lakukan minimal 2 putaran.",
        "6. Setelah lancar, jalankan gladi dengan kamera di meja uji.",
        "",
        f"## Contoh urutan 48 kondisi - {protokol['peserta'][0]}",
        "",
        "| No | Gestur | Lampu | Ulangan | Tahan | Jeda |",
        "|---|---|---|---|---|---|",
    ]
    for r in contoh:
        bagian.append(
            f"| {r['no']} | {r['gestur']} | PWM {r['lampu']} | {r['ulangan']} | 3 detik | 0,5 detik |"
        )
    bagian.extend(
        [
            "",
            "## Kriteria lulus",
            "",
            "- [ ] Semua 48 kondisi dibacakan tanpa terlewat.",
            "- [ ] Timing 3 detik tahan + 0,5 detik turun konsisten.",
            "- [ ] Kartu gestur mudah dikenali partisipan.",
            "- [ ] Tangan dominan tercatat.",
            "- [ ] Tidak ada gestur yang tumpang tindih.",
            "",
        ]
    )
    path.write_text("\n".join(bagian), encoding="utf-8")


def buat_sesi(
    rows: list[dict],
    protokol: dict,
    rencana: Path,
    keluar: Path,
) -> tuple[Path, Path, Path]:
    stempel = datetime.now().strftime("%Y%m%d-%H%M%S")
    sesi = keluar / f"sesi-simulasi-{stempel}"
    bahan = sesi / "bahan_slot"
    bahan.mkdir(parents=True, exist_ok=False)

    tulis_label(sesi / "label.csv", rows)
    tulis_ringkasan(sesi / "ringkasan_sesi.json", protokol, rows, rencana)
    tulis_readme(sesi / "README.md", sesi, protokol, len(rows))
    tulis_kartu(bahan / "KARTU_GESTUR.md", protokol)
    tulis_lembar(bahan / "LEMBAR_SESI.md", protokol)
    tulis_checklist(bahan / "CHECKLIST_BAHAN_SLOT.md")
    tulis_panduan(bahan / "PANDUAN_LATIHAN_KERING.md", protokol, rows)
    shutil.copy2(sesi / "label.csv", bahan / "label_simulasi.csv")

    lulus, log = latihan_kering(rows, protokol)
    (sesi / "latihan_kering.log").write_text("\n".join(log), encoding="utf-8")
    if not lulus:
        raise SystemExit("[meja] latihan kering GAGAL; periksa latihan_kering.log")

    zip_path = keluar / f"{sesi.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as arsip:
        for berkas in sorted(sesi.rglob("*")):
            if berkas.is_file():
                arsip.write(berkas, berkas.relative_to(sesi.parent))

    bagikan = AKAR.parent / "bagikan" / "penjaga-data"
    bagikan.mkdir(parents=True, exist_ok=True)
    tujuan = bagikan / sesi.name
    if tujuan.exists():
        shutil.rmtree(tujuan)
    shutil.copytree(sesi, tujuan)
    shutil.copy2(zip_path, bagikan / zip_path.name)
    return sesi, zip_path, tujuan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simulasi",
        choices=["jalankan"],
        required=True,
        metavar="AKSI",
        help="aksi simulasi (saat ini hanya: jalankan)",
    )
    parser.add_argument("rencana", type=Path, help="berkas YAML rencana meja")
    parser.add_argument(
        "--keluaran",
        type=Path,
        default=AKAR / "data" / "aktual",
        help="folder induk sesi simulasi (default: data/aktual)",
    )
    args = parser.parse_args()

    rencana = args.rencana.resolve()
    data = baca_rencana(rencana)
    protokol = ambil_protokol(data)

    print("[meja] mode simulasi: jalankan")
    print(f"[meja] rencana       : {rencana}")
    print(f"[meja] peserta       : {len(protokol['peserta'])}")
    print(f"[meja] gestur        : {len(protokol['gestur'])}")
    print(f"[meja] taraf lampu   : {protokol['lampu']}")
    print(f"[meja] ulangan       : {protokol['ulangan']}")
    print(f"[meja] kondisi/blok  : {protokol['per_blok']}")
    print(f"[meja] TOTAL         : {protokol['total']} kondisi")
    print(f"[meja] {protokol['total']} kondisi siap dijalankan")

    rows = buat_baris(protokol)
    if len(rows) != protokol["total"]:
        raise SystemExit(f"[meja] baris tidak cocok: {len(rows)} != {protokol['total']}")

    args.keluaran.mkdir(parents=True, exist_ok=True)
    sesi, zip_path, tujuan = buat_sesi(rows, protokol, rencana, args.keluaran)

    print(f"[meja] folder sesi   : {sesi}")
    print(f"[meja] zip           : {zip_path}")
    print(f"[meja] bahan slot    : {sesi / 'bahan_slot'}")
    print(f"[meja] dibagikan ke  : {tujuan}")
    print("[meja] latihan kering: LULUS (tanpa meja)")
    print(f"[meja] {len(rows)} kondisi tercatat di {sesi / 'label.csv'}")


if __name__ == "__main__":
    main()
