# Pull Request — Sprint 1 Tim 13

**Judul yang disarankan**

```
feat(tim13): pemotong sesi simulasi + protokol evaluasi Sprint 1
```

**Branch:** `tim13/sprint-1-pemotong-protokol` → `main`

---

## Ringkasan

PR ini menambahkan pemotong sesi simulasi dan protokol evaluasi Sprint 1 untuk
kasus gestur tangan Tim 13. Semua aturan protokol dijalankan oleh kode, bukan
hanya didokumentasikan.

Yang berubah:

- `src/protokol/PROTOKOL.md` — revisi Sprint 1: buang 1 detik pertama, split per
  partisipan, normalisasi dari data latih, F1 per partisipan.
- `src/protokol/data.py` — `muat()` mengembalikan X/y latih, validasi, uji dari
  folder sesi simulasi; jalur data aktual tetap tersedia lewat
  `PROTOKOL_SUMBER=aktual` / `muat_aktual()`.
- `src/protokol/sesi.py` — pemuatan sesi, split per partisipan, normalisasi pose +
  standardisasi statistik latih.
- `src/protokol/metrik.py` — F1-makro, F1 per partisipan, matriks konfusi.
- `sprint/sprint-1/meja.py` + `sprint/sprint-1/pemotong/` — pipeline pemotong:
  baca tar, deteksi batas partisipan, MediaPipe 21 landmark, buang 1 detik pertama,
  tulis dataset + label + ROI.
- `sprint/sprint-1/sesi_simulasi/` — dataset sesi simulasi (333 sampel, 5 blok).
- `sprint/sprint-1/evaluasi_baseline.py` + `sprint/sprint-1/hasil/` — evaluasi
  RF/MLP dengan GroupKFold per partisipan, 3 seed.
- `sprint/sprint-1/LAPORAN.md` — laporan Sprint 1.
- `data/MANIFEST.json` — menambah bagian `sesi_simulasi` (bagian `berkas` data
  aktual tidak diubah, sehingga `data_hash` tim tetap).
- `requirements.txt`, `.gitignore` — dependensi dan pengecualian model/cache.

---

## Pemenuhan kriteria selesai

| Kriteria | Bukti |
|---|---|
| `muat()` mengembalikan X/y latih, validasi, uji dari folder sesi simulasi | `src/protokol/data.py`, `src/protokol/sesi.py`; diuji dengan `data.muat()` → (166, 83, 84) sampel |
| Tidak ada satuan pemisahan yang muncul di latih dan uji sekaligus | `hasil/periksa_kebocoran.json`: **82 butir, 0 gagal** (partisipan, kondisi, frame) |
| Normalisasi dihitung dari data latih saja | Pemeriksaan `mean_dari_latih` / `std_dari_latih` / `*_dinormalisasi_dengan_stat_latih` lulus untuk split beku dan setiap lipat |
| Buang 1 detik pertama setiap kondisi | `pemotong/potong.py`: sampel hanya pada `[mulai + 1,0 s, selesai)` |
| Metrik utama F1 per partisipan | `src/protokol/metrik.py`; laporan `hasil/per_peserta.csv`, `hasil/metrics.json` |
| PROTOKOL.md diajukan ke pembimbing lewat pull request | Dokumen ini + `src/protokol/PROTOKOL.md` |

---

## Cara memverifikasi

```bash
# 1. Pasang dependensi
python -m pip install -r requirements.txt

# 2. Potong sesi simulasi (model MediaPipe diunduh otomatis)
python sprint/sprint-1/meja.py --simulasi jalankan sprint/rencana/t13-gestur-kamera.yaml

# 3. Periksa kebocoran & normalisasi
python sprint/sprint-1/meja.py --simulasi periksa sprint/sprint-1/sesi_simulasi
#    -> "Pemeriksaan: 82 butir, 0 gagal" / "LULUS"

# 4. Evaluasi baseline
python sprint/sprint-1/evaluasi_baseline.py
#    -> hasil/metrics.json, hasil/per_peserta.csv, hasil/laporan.md
```

Kontrak data:

```python
from src.protokol import data
X_train, y_train, X_val, y_val, X_test, y_test = data.muat()
```

---

## Hasil singkat

GroupKFold per partisipan pada P01–P04; P05 disisihkan sebagai uji akhir.

| Model | F1-makro antar-partisipan | Simpangan baku |
|---|---|---|
| RF | 0,082 | 0,057 |
| MLP | 0,084 | 0,018 |

F1 masih rendah karena rekaman web berjarak **1 detik/frame** (bukan 0,1 detik),
hanya 5 blok partisipan yang ada, dan label mengikuti urutan rencana. Angka ini
adalah verifikasi jalur evaluasi, bukan hasil final. Detail:
`sprint/sprint-1/LAPORAN.md`.

---

## Catatan untuk reviewer

- Rekaman `tim13_web-20261118-123656-9367.tar` tidak di-commit (di `data/`,
  di-*ignore*). Yang di-commit adalah turunan sesi di `sprint/sprint-1/sesi_simulasi/`.
- Model MediaPipe `*.task` diunduh otomatis dan tidak di-commit.
- `P05` sengaja tidak dievaluasi pada Sprint 1; jangan dipakai untuk memilih
  hyperparameter.
- Kode `P01–P05` adalah urutan blok rekaman web, bukan identitas partisipan
  terverifikasi. Sesi aktual memakai P01, P02, P04, P05, P06.
- **Kompatibilitas:** `data.muat()` kini memakai sesi simulasi secara bawaan.
  Jalur data aktual tidak dihapus: `PROTOKOL_SUMBER=aktual` atau
  `data.muat_aktual()` mengembalikan perilaku Sprint 0 (split acak 10 %, seed 2026).
  `data.muat_partisipan()` tetap ada.
- **Perubahan aturan split** (acak 10 % → per partisipan) adalah inti revisi
  protokol yang perlu disetujui pembimbing.
- `data_hash` sesi simulasi: `2122804488d5d5e95141e832ebed91ed`;
  `data_hash` data aktual tidak berubah (`1150045b5a6138cdfab7328c10b42243`).

## Checklist

- [x] Aturan protokol dijalankan kode
- [x] Pemeriksaan kebocoran lulus (82/82)
- [x] Normalisasi hanya dari data latih
- [x] `muat()` mengembalikan 6 array sesuai kontrak
- [x] Evaluasi F1 per partisipan terdokumentasi
- [x] `data/MANIFEST.json` diperbarui tanpa mengubah `data_hash` aktual
- [ ] Persetujuan pembimbing
