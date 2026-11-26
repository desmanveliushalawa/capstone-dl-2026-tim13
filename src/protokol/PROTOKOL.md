# Protokol evaluasi tim — Ruang Cerdas 5V (Tim 13)

Dokumen ini adalah **sumber aturan** untuk `src/protokol/data.py`. Setiap aturan di sini
dijalankan oleh kode, bukan hanya ditulis.

Riwayat: disusun Sprint 0, direvisi Sprint 1 (26 November 2026) untuk dataset sesi
18 November dan pemisahan per partisipan. Revisi ini **menunggu persetujuan pembimbing
lewat pull request**; setelah disetujui, protokol dibekukan kembali.

| Butir | Isi |
|---|---|
| Kasus | Gestur tangan pengendali meja (kamera) — Ruang Cerdas 5 Volt |
| Jenis tugas (`TUGAS`) | klasifikasi 8 kelas gestur statis |
| Target / label | 8 gestur: terbuka, kepalan, jempol atas, jempol bawah, telunjuk, V, OK, telapak turun |
| Sumber data utama | Sesi meja 18 Nov 2026: `data/aktual/gestur_primer.npz` (332 sampel; partisipan P01, P02, P04, P05, P06) |
| Sumber data Sprint 1 (simulasi) | Rekaman web `data/tim13_web-20261118-123656-9367.tar` → `sprint/sprint-1/sesi_simulasi/` (333 sampel) |
| Satuan pemisahan | **partisipan** — satu partisipan hanya boleh ada di satu sisi split |
| Metrik utama | **F1-makro per partisipan** (rerata ± simpangan baku antar-partisipan) |
| Metrik pendukung | akurasi per partisipan, matriks konfusi agregat, ukuran ONNX KB, latensi CPU ms |
| Praproses sama untuk semua model | Buang 1 detik pertama tiap kondisi; 21 landmark MediaPipe (x, y, z); cerminkan x bila tangan kiri; normalisasi pose; standardisasi dengan statistik data latih |
| Anggaran komputasi per run | ≤ 20 menit CPU laptop meja |
| Evaluasi data uji | sekali, oleh Penjaga Data, pada sprint terakhir |
| `data_hash` data aktual | `1150045b5a6138cdfab7328c10b42243` |
| `data_hash` sesi simulasi | `2122804488d5d5e95141e832ebed91ed` |

## 1. Sumber dan bentuk data

1. Pemotong `sprint/sprint-1/meja.py --simulasi jalankan ../rencana/t13-gestur-kamera.yaml`
   membaca tar rekaman **langsung tanpa mengekstrak seluruh citra**.
2. Setiap frame terlihat dianalisis dengan MediaPipe HandLandmarker (mode VIDEO,
   `num_hands=1`, confidence 0,1), menghasilkan 21 titik `(x, y, z)` per frame.
3. Potongan citra tangan (ROI 224×224) disimpan di `sesi_simulasi/roi/` untuk jalur TL.
4. Keluaran sesi: `dataset.npz`, `label.csv`, `sesi.json`, `landmark_mentah.npz`.

### Catatan laju rekaman

Rencana meja menulis interval **0,1 detik/frame**, tetapi metadata rekaman web
(`rekam-web.json`, `kamera.selangMs=1000`) dan indeks `kamera_index/*.csv` menunjukkan
**1 detik/frame** (interval terukur 0,998 detik). Protokol memakai timestamp asli
rekaman, bukan asumsi 0,1 detik. Akibatnya, setelah membuang 1 detik pertama, satu
kondisi 3 detik hanya menyisakan sekitar 1–2 frame pada rekaman web ini. Data aktual
dari komputer meja tetap memakai laju penuh sesuai label sesi.

## 2. Aturan pemotongan (dipakai `data.py`)

1. **Buang 1 detik pertama setiap kondisi.** Slot kondisi berdurasi 3 detik; sampel
   hanya diambil pada `[slot_mulai + 1,0 s, slot_selesai)`.
2. Slot disusun dengan kadens rencana `durasi 3,0 s + jeda 0,5 s = 3,5 s`,
   dijangkarkan mundur dari frame tangan valid terakhir tiap partisipan.
3. Frame tanpa deteksi tangan **tidak diisi dan tidak diinterpolasi**.
4. Jumlah sampel per partisipan boleh berbeda.

## 3. Aturan split (dipakai `data.py`)

1. **Data uji dipisah per partisipan.** Tidak ada partisipan yang muncul di dua sisi
   split sekaligus.
2. Split beku sesi simulasi Sprint 1:
   - latih: `P01, P02, P03`
   - validasi: `P04`
   - uji: `P05` (disisihkan, **belum dievaluasi** pada Sprint 1)
3. Untuk komparasi model dipakai **GroupKFold per partisipan** pada `P01–P04`:
   satu partisipan menjadi lipat uji, partisipan berikutnya menjadi validasi,
   sisanya latih. `P05` tetap menjadi uji akhir yang tidak disentuh.
4. Pemeriksaan kebocoran dijalankan oleh `meja.py --simulasi periksa` dan menulis
   `sprint/sprint-1/hasil/periksa_kebocoran.json`. Pemeriksaan mencakup irisan
   partisipan, irisan kondisi, irisan frame, dan kecocokan statistik normalisasi.
   Status Sprint 1: **82 butir, 0 gagal**.
5. Split acak 10 % pada protokol Sprint 0 **tidak lagi dipakai** karena melanggar
   aturan pemisahan per partisipan.

## 4. Aturan normalisasi (dipakai `data.py`)

1. **Normalisasi dihitung dari data latih saja.** Mean/std di-*fit* pada `X_latih`
   lipat yang sedang dievaluasi, lalu diterapkan ke validasi dan uji.
2. Sebelum standardisasi, setiap sampel dinormalisasi pose:
   - tangan kiri dicerminkan (`x → -x`) agar seragam dengan tangan kanan;
   - titik asal dipindah ke pergelangan tangan (landmark 0);
   - skala dibagi jarak pergelangan → pangkal jari tengah (landmark 9).
3. Normalisasi pose tidak memakai statistik lintas sampel, sehingga aman diterapkan
   sebelum split. Standardisasi mean/std tetap hanya dari data latih.
4. `periksa_kebocoran.json` memverifikasi bahwa mean/std yang dipakai benar-benar sama
   dengan mean/std data latih (toleransi 1e-5).

## 5. Metrik dan pelaporan

1. Metrik utama: **F1-makro per partisipan**, dihitung oleh
   `src/protokol/metrik.py::per_peserta`.
2. Ringkasan: rerata dan simpangan baku F1-makro antar-partisipan (3 seed: 1, 2, 3).
3. Evaluasi baseline Sprint 1 memakai:
   - Random Forest 300 pohon, `class_weight="balanced"`, fitur 210 jarak berpasangan
     dari pose ternormalisasi;
   - MLP PyTorch 2 lapisan tersembunyi (64, 32), dropout, early stopping pada validasi,
     fitur 63 koordinat pose ternormalisasi + standardisasi latih.
4. Data uji akhir hanya boleh dievaluasi **sekali** pada sprint terakhir setelah model
   final dipilih. Sprint 1 hanya melaporkan hasil lintas-partisipan pada `P01–P04` dan
   validasi `P04`.

## 6. Cara menjalankan

```bash
# 1. Potong sesi simulasi (menulis dataset + label + ROI)
python sprint/sprint-1/meja.py --simulasi jalankan sprint/rencana/t13-gestur-kamera.yaml

# 2. Periksa kebocoran partisipan/kondisi/frame + normalisasi
python sprint/sprint-1/meja.py --simulasi periksa sprint/sprint-1/sesi_simulasi

# 3. Evaluasi baseline (F1 per partisipan, GroupKFold)
python sprint/sprint-1/evaluasi_baseline.py
```

Kontrak pemuatan data tetap:

```python
from src.protokol import data
X_train, y_train, X_val, y_val, X_test, y_test = data.muat()
```

## 7. Keterbatasan yang diketahui

1. Rekaman web hanya berisi **5 blok partisipan** dari 8 yang direncanakan.
2. Kode `P01–P05` pada sesi simulasi adalah **urutan blok** rekaman web, bukan
   identitas partisipan terverifikasi. Sesi aktual tim memakai kode P01, P02, P04, P05,
   P06 (P03/P07 frame hitam, P08 tidak terekam). Pemisahan per blok tetap sah karena
   tiap blok adalah orang berbeda.
3. Rekaman web berjarak **1 detik/frame**, sehingga sampel per kondisi sangat sedikit
   dan gerakan cepat tidak terekam.
4. Rekaman web tidak menyertakan `label.csv` sesi; label gestur mengikuti urutan
   rencana. Kesalahan penyelarasan slot mungkin terjadi.
5. F1 lintas-partisipan pada Sprint 1 masih rendah (RF 0,082 ± 0,057; MLP 0,084 ± 0,018).
   Angka ini adalah **verifikasi jalur evaluasi**, bukan hasil final; lihat
   `sprint/sprint-1/LAPORAN.md`.
