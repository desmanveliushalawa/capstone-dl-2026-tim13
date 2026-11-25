# Catatan model 123450032 (Transformer encoder, jalur `trf`)

Status: Sprint 1, seed 1 dilatih pada **dataset primer meja** `data/aktual/gestur_primer.npz`
(332 sampel, 21 titik MediaPipe, 8 kelas) yang dirujuk `data/MANIFEST.json`. `data_hash` run
ini `1150045b…`, sama dengan hash MANIFEST. Tiga seed, penyetelan, dan ablation dikerjakan di
Sprint 2.

## Arsitektur singkat

- Masukan `[N, 21, 3]`: 21 titik MediaPipe menjadi 21 token (x, y, z), ditambah token CLS dan embedding posisi yang dipelajari.
- Encoder 2 lapis, 4 head, `d_model=32`, `dim_feedforward=64`, pre-norm. Keluaran: logit `[N, 8]`.
- Standarisasi per koordinat memakai statistik data latih saja.
- Potongan tangan (citra) belum dipakai: `data.muat()` hanya membawa landmark.

## Alasan optimizer

**AdamW**, `lr=2e-3`, `weight_decay=1e-2`, jadwal OneCycle (pemanasan 10 %).

- Transformer peka terhadap skala gradien; Adam menyesuaikan laju belajar per parameter sehingga lebih stabil daripada SGD pada data kecil (299 sampel latih).
- AdamW memisahkan weight decay dari pembaruan gradien, jadi penalti bobot bekerja sebagaimana mestinya (pada Adam biasa, L2 ikut diskalakan oleh momen adaptif).
- OneCycle (naik lalu turun) membantu model tanpa pemanasan yang lama menghindari lonjakan awal.

## Regularisasi

- Dropout 0,1 pada lapisan encoder.
- Weight decay 0,01 lewat AdamW.
- Pre-norm (`norm_first=True`) dan LayerNorm akhir untuk menjaga kestabilan latihan.
- Ukuran model kecil (`d_model=32`, 2 lapis) sebagai pembatas kapasitas.

## Pencarian hyperparameter (tabel)

Belum ada pencarian; baru satu konfigurasi awal. Pencarian dilakukan di Sprint 2.

| Eksperimen | d_model | Lapis | Head | Dropout | LR | Batch | Epoch | Val acc | Val macro-F1 | Keterangan |
|---|---|---|---|---|---|---|---|---|---|---|
| Konfigurasi awal (data primer) | 32 | 2 | 4 | 0,1 | 2e-3 | 32 | 50 | 0,3333 | 0,2596 | Seed 1, `n_val=33`; kalah dari baseline RF (0,4545 / 0,4006) |

Rencana pencarian Sprint 2: `d_model` ∈ {16, 32, 64}, lapis ∈ {1, 2, 3}, dropout ∈ {0,1; 0,2; 0,3}, LR ∈ {1e-3; 2e-3; 5e-3}.

## Ablation (>= 2)

Belum dijalankan; direncanakan di Sprint 2 pada data primer:

1. **Token CLS vs mean pooling.** Varian A: keluaran token CLS (konfigurasi sekarang). Varian B: rata-rata semua token titik.
2. **Embedding posisi.** Varian A: dengan embedding posisi (identitas sendi). Varian B: tanpa embedding posisi (21 token diperlakukan sebagai himpunan).
3. (Opsional) **Jumlah lapis encoder:** 1 vs 2 lapis.

## Hasil validasi seed 1

Perintah:

```bash
python models/123450032/run.py --seed 1
python models/123450032/evaluasi_partisipan.py --seed 1
```

- `val_acc = 0.333333`, `val_macro_f1 = 0.259615`, `n_val = 33`, durasi 12,6 detik (batas 20 menit).
- `data_hash = 1150045b5a6138cdfab7328c10b42243`.
- Data latih/validasi/uji: 299 / 33 / —.

F1 per partisipan pada data validasi (split beku seed 2026):

| Partisipan | n | acc | macro-F1 |
|---|---|---|---|
| P01 | 7 | 0,5714 | 0,3200 |
| P02 | 11 | 0,0000 | 0,0000 |
| P04 | 3 | 0,3333 | 0,2500 |
| P05 | 7 | 0,2857 | 0,1944 |
| P06 | 5 | 0,8000 | 0,7333 |

Pembanding non-DL tim (Random Forest + sudut sendi, 93 fitur, seed 2026) pada split yang sama:
`val_acc = 0.4545`, `val_macro_f1 = 0.4006` — jadi Transformer **belum mengungguli** pembanding
dan perlu penyetelan di Sprint 2.

## Keterbatasan data aktual

- Frame kandidat 658 (setelah buang 1 detik pertama); hanya 332 frame menghasilkan 21 landmark
  (50,5 %). Sisa frame tanpa tangan/terlalu blur.
- P03 dan P07: seluruh frame blok tersebut hitam (0 piksel), 0 landmark. P08: tidak ada frame
  sama sekali (kamera berhenti 13:05:09, blok P08 13:05–13:13). Partisipan terpakai: P01, P02,
  P04, P05, P06.
- Validasi hanya 33 sampel, sehingga F1 per partisipan sangat berderau (P04 hanya 3 sampel).
  Perbandingan yang adil menunggu 3 seed dan split per partisipan di Sprint 2.

Catatan teknis: `run.py` meneruskan `PYTHONIOENCODING=utf-8` ke penilai agar karakter `→`
tidak gagal dicetak pada konsol Windows (cp1252).
