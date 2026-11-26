# Catatan model 122450125

## Alasan optimizer

Optimizer yang digunakan dalam model adalah **AdamW** dengan learning rate sebesar `0.002` dan weight decay sebesar `0.0001` (`1e-4`). AdamW dipilih karena menggunakan mekanisme *adaptive learning rate* untuk menyesuaikan pembaruan parameter berdasarkan estimasi gradien. Selain itu, AdamW menerapkan *weight decay* secara terpisah dari pembaruan gradien sehingga membantu mengendalikan kompleksitas model dan mengurangi risiko *overfitting*.

## Regularisasi

1. **Standarisasi fitur (dari data latih saja):** Setiap koordinat landmark $(21 \times 3)$ distandarisasi menggunakan rata-rata dan simpangan baku data latih untuk menjaga kestabilan gradien tanpa kebocoran data validasi.
2. **Batch Normalization (`BatchNorm1d`):** Diterapkan setelah lapisan linear (`128` dan `64`) untuk menstabilkan distribusi aktivasi selama pelatihan.
3. **Dropout (`0.3` dan `0.2`):** Mematikan sebagian neuron secara acak pada lapisan *backbone* (`128` $\rightarrow$ `64`) agar representasi fitur tidak bergantung pada satu jalur tertentu.
4. **Weight Decay (`1e-4`):** Regularisasi $L_2$ terpisah pada AdamW untuk membatasi besar bobot.

## Pencarian hyperparameter (tabel)

| Eksperimen | Arsitektur | LR | Weight Decay | Dropout | Val Acc | Val Macro-F1 | Durasi (s) | Keterangan |
|---|---|---|---|---|---|---|---|---|
| **TL-MLP (Sprint 1)** | **63 → 128 → 64 → 8** | **0.002** | **1e-4** | **0.3, 0.2** | **0.3333** | **0.2738** | **16.7** | **Dataset primer meja (`n_val=33`, seed 1)** |

*(Pencarian hyperparameter lanjutan dan adaptasi MobileNet pada potongan citra tangan akan dilanjutkan pada Sprint 2).*

## Ablation (>= 2)

*(Akan dieksekusi pada Sprint 2: perbandingan tanpa vs dengan Dropout/BatchNorm, serta perbandingan variasi dimensi feature embedding).*

## Hasil 3 seed konfigurasi akhir

* Seed 1 (Sprint 1, dataset primer): `val_acc = 0.333333`, `val_macro_f1 = 0.27381` (`data_hash: 1150045b5a6138cdfab7328c10b42243`).
* *(Seed 2 dan Seed 3 akan dijalankan pada Sprint 2).*
