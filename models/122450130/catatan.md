# Catatan model 122450130 — 1D-CNN (Sprint 1)

## 1. Ringkasan pelatihan
- Arsitektur: `Conv1d(3→16)` → `BatchNorm1d(16)` → ReLU → `MaxPool1d(2)` →
  `Conv1d(16→32)` → `BatchNorm1d(32)` → ReLU → `AdaptiveAvgPool1d(1)` →
  Flatten → `Dropout(0.2)` → `Linear(32→8)`.
- Input `[N,21,3]` di-*permute* menjadi `[N,3,21]` (channel-first) sebelum masuk `Conv1d`.
- Data: `sprint/sprint-1/sesi_simulasi/dataset.npz` dari tar
  `tim13_web-20261118-123656-9367.tar` (333 sampel, split per partisipan
  P01–P03 / P04 / P05). `n_val=83`, `n_test=0` (belum pakai `--final`).
- Seed 1: `results/122450130/metrics.json` → `val_acc=0.0000`,
  `val_macro_f1=0.00000`, durasi 64,3 detik di CPU laptop (masih di bawah
  anggaran ≤ 20 menit/run).

## 2. Alasan optimizer
Adam dipilih karena each parameter punya learning rate adaptif, cocok untuk
koordinat landmark dengan skala yang berbeda-beda; `lr=3e-3`,
`weight_decay=1e-4` memberi regularisasi L2 ringan terhadap overfitting
pada dataset kecil (333 sampel). Momentum Adam juga membantu konvergensi
dalam 50 epoch tanpa scheduler.

## 3. Regularisasi
- `BatchNorm1d` setelah tiap konvolusi: menstabilisasi distribusi aktivasi.
- `Dropout(0.2)` sebelum classifier: mencegah ko-adaptasi fitur.
- `weight_decay=1e-4` pada Adam.
- `MaxPool1d(2)` + `AdaptiveAvgPool1d(1)` mereduksi sensitivitas posisi.

## 4. Pencarian hyperparameter
Belum ada grid search penuh. Selain konfigurasi akhir (3 seed), dua
percobaan ablation dilakukan dengan seed yang sama (lihat §5). Opsi yang
belum dicoba dan direncanakan: `lr ∈ {3e-4, 1e-3}`, `wd ∈ {0, 1e-3}`,
`epoch ∈ {100}`, `dropout ∈ {0.1, 0.4, 0.5}`.

| #

 | BN | Dropout | Seed | val_acc | val_macro_f1 |
|---|---|---|---|---|---|
| 1 (final) | ✓ | 0.2 | 1 | 0.0000 | 0.00000 |
| 2 (final) | ✓ | 0.2 | 2 | 0.0723 | 0.0511 |
| 3 (final) | ✓ | 0.2 | 3 | 0.1084 | 0.0594 |
| 4 (ablasi) | ✗ | 0.2 | 1 | 0.0964 | 0.0886 |
| 5 (ablasi) | ✓ | 0.0 | 1 | 0.0843 | 0.0725 |

## 5. Ablasi (≥2)
1. Tanpa `BatchNorm1d` → `val_acc=0.0964`, `val_macro_f1=0.0886`.
2. Tanpa `Dropout` → `val_acc=0.0843`, `val_macro_f1=0.0725`.

Keduanya dijalankan dengan arsitektur & seed identik dengan referensi
(seed 1). Angka seed 1 rendah (0.0) sehingga perbandingan ini lebih
cocok dibaca sebagai indikasi varians seed, bukan bukti bahwa komponen
tertentu merugikan. Perlu 3 seed per varian sebelum kesimpulan kuat.

## 6. Hasil 3 seed konfigurasi akhir
| Seed | val_acc | val_macro_f1 |
|---|---|---|
| 1 | 0.0000 | 0.00000 |
| 2 | 0.0723 | 0.0511 |
| 3 | 0.1084 | 0.0594 |

Rerata `val_acc ≈ 0.060`, `val_macro_f1 ≈ 0.037`. Angka masih rendah,
sejalan dengan catatan umum Sprint 1 (`LAPORAN.md`): rekaman web 1 fps
dan hanya 83 sampel validasi.

## 7. Catatan teknis & keterbatasan
- Saya memperbaiki `AKAR` di `run.py`: semula dihitung dari path tar
  (menjadi `C:\Users\hp`), kini `Path(__file__).resolve().parents[2]`
  sesuai pola anggota lain, agar `src.common.evaluate` dan
  `results/122450130/` ditulis ke root repo yang benar.
- evaluator starter `src.common.evaluate` menghitung `data_hash` dari
  `data/aktual/gestur_primer.npz` yang tidak ikut di-commit; untuk
  keperluan run lokal, berkas itu saya buat dari dataset sesi simulasi
  sehingga `metrics.json` valid. `data_hash` di `metrics.json` =
  `01ac48653649b59c2590962f3492e9a1` (hash lokal, bukan hash resmi
  data aktual `1150045b5a6138cdfab7328c10b42243`).
- Prediksi tambahan (`pred_val_seed2.npy`, `pred_val_seed3.npy`,
  `pred_val_ab_no_bn.npy`, `pred_val_ab_no_drop.npy`) tersimpan di
  `results/122450130/`; skrip bantu `_tmp_abl.py` dipakai untuk
  menghasilkannya dan bukan bagian utama model.
- Data uji (`P05`, n=84) sengaja belum dievaluasi; akan dipakai sekali
  pada sprint terakhir lewat `--final`.
- F1 per partisipan & diagnosis lebih lanjut ada di `LAPORAN.md` Sprint 1.
