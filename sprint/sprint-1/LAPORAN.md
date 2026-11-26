# Laporan Sprint 1 — Pemotong & Protokol Evaluasi Tim 13

**Kasus:** Gestur tangan pengendali meja (kamera) — Ruang Cerdas 5 Volt
**Sumber:** `data/tim13_web-20261118-123656-9367.tar`
**Kode & hasil:** `sprint/sprint-1/`
**Protokol:** `src/protokol/PROTOKOL.md` (dipakai `src/protokol/data.py`)

---

## 1. Ringkasan yang dikerjakan

Sprint 1 menyiapkan **pemotong sesi simulasi** dan **protokol evaluasi** sebelum data
meja asli tersedia. Alur lengkapnya:

```
tar rekaman
  -> baca indeks waktu & profil gerak
  -> deteksi batas blok peserta
  -> ekstraksi 21 landmark MediaPipe per frame + ROI tangan
  -> potong 48 kondisi/peserta, buang 1 detik pertama
  -> dataset.npz + label.csv + sesi.json + roi/
  -> split per peserta + normalisasi dari data latih
  -> evaluasi F1-makro per peserta
```

Perintah utama:

```bash
python sprint/sprint-1/meja.py --simulasi jalankan sprint/rencana/t13-gestur-kamera.yaml
python sprint/sprint-1/meja.py --simulasi periksa sprint/sprint-1/sesi_simulasi
python sprint/sprint-1/evaluasi_baseline.py
```

Kontrak `muat()` tetap:

```python
from src.protokol import data
X_train, y_train, X_val, y_val, X_test, y_test = data.muat()
```

Secara bawaan `muat()` membaca folder sesi simulasi. Jalur data aktual Sprint 0
(`data/aktual/gestur_primer.npz`, split acak 10 %) tetap tersedia lewat
`PROTOKOL_SUMBER=aktual` atau `data.muat_aktual()`, sehingga skrip anggota tim
lain tidak putus.

---

## 2. Temuan pada data simulasi

| Hal | Nilai |
|---|---|
| Jumlah frame di tar | 1.670 |
| Frame terlihat (bukan hitam) | 1.049 (indeks 0–1048) |
| Interval rekaman | **0,998 detik/frame** (metadata `selangMs=1000`) |
| Resolusi | 640×480 |
| Blok peserta terdeteksi | **5** (P01–P05) |
| Peserta direncanakan | 8 (P06–P08 tidak ada di rekaman) |
| Batas blok (indeks frame) | `[0, 240, 440, 633, 847, 1049]` |
| Kondisi terisi | 193 / 240 |
| Sampel total | 333 |

**Catatan kode partisipan:** `P01–P05` adalah **urutan blok** pada rekaman web ini,
bukan identitas partisipan terverifikasi. Rekaman tidak menyertakan `label.csv` sesi.
Sesi aktual tim memakai kode `P01, P02, P04, P05, P06` (P03/P07 frame hitam, P08 tidak
terekam). Pemisahan per blok tetap sah karena tiap blok adalah orang berbeda.

**Catatan penting:** rencana meja menulis 0,1 detik/frame, tetapi rekaman ini
sesungguhnya **1 fps**. Protokol memakai timestamp asli. Setelah membuang 1 detik
pertama dari kondisi 3 detik, tiap kondisi hanya menyisakan sekitar 1–2 frame.
Karena itu jumlah sampel kecil dan F1 lintas-peserta belum stabil.

---

## 3. Aturan protokol yang dijalankan kode

1. **Buang 1 detik pertama setiap kondisi.**
   Slot 3 detik; sampel diambil pada `[slot_mulai + 1,0 s, slot_selesai)`.
2. **Data uji dipisah per peserta.**
   Split beku: latih `P01–P03`, validasi `P04`, uji `P05`.
   Untuk komparasi model: GroupKFold pada `P01–P04`, `P05` disisihkan.
3. **Normalisasi dari data latih saja.**
   Pose tiap sampel dinormalisasi (cermin tangan kiri, titik asal pergelangan,
   skala pergelangan→pangkal jari tengah), lalu mean/std di-*fit* pada data latih
   lipat dan diterapkan ke validasi/uji.
4. **Metrik utama: F1-makro per peserta.**
   Dilaporkan rerata ± simpangan baku antar-peserta, 3 seed (1, 2, 3).

Revisi protokol ini mengganti split acak 10 % Sprint 0 dengan split per partisipan,
karena split acak dapat menempatkan orang yang sama di latih dan uji.

### Pemeriksaan kebocoran

`python sprint/sprint-1/meja.py --simulasi periksa sprint/sprint-1/sesi_simulasi`

- **82 butir pemeriksaan, 0 gagal.**
- Mencakup irisan peserta, kondisi, frame, serta kecocokan mean/std normalisasi
  (toleransi 1e-5) untuk split beku dan setiap lipat GroupKFold.
- Laporan: `sprint/sprint-1/hasil/periksa_kebocoran.json`.

---

## 4. Hasil evaluasi baseline

Fitur:
- **RF:** 210 jarak berpasangan dari pose ternormalisasi.
- **MLP:** 63 koordinat pose ternormalisasi + standardisasi statistik latih.

### Ringkasan antar-peserta (P01–P04, 3 seed)

| Model | F1-makro | Simpangan baku | Akurasi |
|---|---|---|---|
| RF | 0,082 | 0,057 | 0,098 |
| MLP | 0,084 | 0,018 | 0,127 |

### F1-makro per peserta

| Peserta | n uji | RF F1 | RF akurasi | MLP F1 | MLP akurasi |
|---|---|---|---|---|---|
| P01 | 45 | 0,126 ± 0,013 | 0,170 | 0,082 ± 0,054 | 0,148 |
| P02 | 39 | 0,000 ± 0,000 | 0,000 | 0,079 ± 0,052 | 0,137 |
| P03 | 82 | 0,115 ± 0,029 | 0,122 | 0,109 ± 0,077 | 0,138 |
| P04 | 83 | 0,088 ± 0,018 | 0,100 | 0,065 ± 0,035 | 0,084 |

### Validasi split beku (P04)

| Model | F1-makro validasi | Simpangan baku |
|---|---|---|
| RF | 0,082 | 0,007 |
| MLP | 0,121 | 0,025 |

Data uji akhir **P05 tidak dievaluasi** pada Sprint 1.

### Tafsir

Angka lintas-peserta masih rendah. Diagnostik tambahan menunjukkan model dapat
mencapai F1 0,17–0,56 bila dilatih dan diuji pada peserta yang sama, sehingga
masalah utama bukan pipa data yang rusak, melainkan:
- generalisasi antar-peserta dengan sangat sedikit sampel,
- penyelarasan label yang hanya mengandalkan urutan rencana,
- laju rekaman 1 fps yang menghilangkan dinamika gestur.

Hasil ini adalah **verifikasi jalur evaluasi**, bukan hasil final.

---

## 5. Artefak

| Berkas | Isi |
|---|---|
| `sprint/sprint-1/meja.py` | CLI pemotong: `jalankan`, `periksa`, `ringkas` |
| `sprint/sprint-1/pemotong/rencana.py` | Baca YAML & susun 48 kondisi |
| `sprint/sprint-1/pemotong/rekaman.py` | Baca tar, indeks waktu, profil gerak, deteksi batas |
| `sprint/sprint-1/pemotong/deteksi.py` | MediaPipe HandLandmarker + ROI |
| `sprint/sprint-1/pemotong/potong.py` | Pemotongan kondisi & penulisan dataset |
| `sprint/sprint-1/pemotong/periksa.py` | 82 pemeriksaan kebocoran & normalisasi |
| `sprint/sprint-1/sesi_simulasi/dataset.npz` | X `[333,21,3]`, y, peserta, kondisi, frame, tangan, dll. |
| `sprint/sprint-1/sesi_simulasi/label.csv` | Label per kondisi (termasuk kondisi kosong) |
| `sprint/sprint-1/sesi_simulasi/sesi.json` | Metadata sesi, batas, split, hash |
| `sprint/sprint-1/sesi_simulasi/roi/` | 193 potongan tangan 224×224 |
| `sprint/sprint-1/evaluasi_baseline.py` | Evaluasi RF/MLP, GroupKFold, 3 seed |
| `sprint/sprint-1/hasil/metrics.json` | Metrik lengkap |
| `sprint/sprint-1/hasil/per_peserta.csv` | Tabel F1 per peserta |
| `sprint/sprint-1/hasil/laporan.md` | Laporan evaluasi ringkas |
| `sprint/sprint-1/hasil/periksa_kebocoran.json` | Bukti pemeriksaan kebocoran |
| `src/protokol/sesi.py` | Pemuatan sesi, split, normalisasi pose + statistik latih |
| `src/protokol/metrik.py` | F1-makro, per peserta, konfusi |
| `src/protokol/data.py` | Kontrak `muat()` |
| `src/protokol/PROTOKOL.md` | Aturan protokol |

`data_hash` dataset: `2122804488d5d5e95141e832ebed91ed`
(tercatat juga di `sesi.json`, `metrics.json`, dan `data/MANIFEST.json`).

---

## 6. Keterbatasan & langkah berikut

1. Rekaman hanya 5 peserta dan 1 fps. Saat data meja asli (10 Hz) tersedia,
   jalankan ulang pemotong yang sama; aturan protokol tidak berubah.
2. Label gestur mengikuti urutan rencana karena tidak ada `label.csv` sesi.
   Jika data asli menyertakan label, pemotong dapat memakainya langsung.
3. F1 lintas-peserta belum layak dibandingkan antar-model. Fokus sprint
   berikutnya: penambahan fitur temporal, augmentasi ringan, dan pemilihan model
   yang tahan terhadap variasi peserta.
4. `P05` tetap disisihkan sampai sprint terakhir; jangan dipakai untuk memilih
   hyperparameter.
