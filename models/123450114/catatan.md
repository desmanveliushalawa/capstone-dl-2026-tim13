# Catatan Model 123450114 (Autoencoder - Ruang Cerdas 5 Volt)

## Alasan Optimizer
* **Pilihan:** `AdamW` (learning rate pre-training $0.005$, fine-tuning $0.002$, weight decay $1\times 10^{-4}$).
* **Alasan:**
  1. *Decoupled Weight Decay:* AdamW memisahkan pembaruan gradien berbasis momen adaptif dari penalti bobot L2. Ini sangat krusial pada Autoencoder agar bobot rekonstruksi decoder tidak meledak atau mengalami bobot jenuh (*saturation*).
  2. *Stabilitas Kompresi Laten:* Dibandingkan SGD biasa yang lambat konvergen pada ruang koordinat 3D yang memiliki korelasi spasial antar sendi jari, AdamW menyesuaikan learning rate secara per-parameter sehingga ruang laten lebih cepat membentuk manifold gestur yang teratur.

## Regularisasi
* **Bottleneck Dimensionality (16D):** Mereduksi 63 dimensi koordinat ($21 \times 3$) menjadi 16 dimensi. Kompresi ini bertindak sebagai regularisasi representasi yang meredam derau (*sensor jitter*) dari kamera meja uji.
* **Weight Decay ($1\times 10^{-4}$):** Mencegah overfitting pada lapisan Linear encoder dan decoder.
* **Joint Reconstruction Loss (Koefisien $0.1$):** Selama pelatihan pengklasifikasi, loss rekonstruksi tetap dipertahankan sebesar $10\%$ agar fitur laten tidak melupakan geometri fisik tangan.

## Pencarian Hyperparameter (Tabel Awal Sprint 1)

| Eksperimen | Dim Laten | LR Pretrain | LR Finetune | Batch | Val Acc | Val Macro-F1 | Durasi (s) | Keterangan |
|---|---|---|---|---|---|---|---|---|
| Baseline Starter | — | — | 0.1 (SGD) | 128 | 0.0875 | 0.0760 | ~0s | Regresi Logistik Dosen |
| Dummy Linear | — | — | 0.01 (SGD) | 128 | 0.1500 | 0.1407 | 3.7s | Model satu lapis |
| **AE Dasar (Sprint 1)** | **16** | **0.005** | **0.002** | **32** | **0.1375** | **0.1223** | **6.7s** | **Arsitektur AE resmi Sprint 1** |

*(Pencarian komprehensif dimensi laten [8, 16, 32] dan learning rate akan difinalkan di Sprint 2 setelah data primer meja uji 10 Nov masuk).*

## Rencana Ablation (Sprint 2)
1. **Ablation 1 (Pre-training Rekonstruksi):**
   * *Varian A:* Autoencoder dengan pre-training rekonstruksi MSE lalu dilanjutkan klasifikasi (konfigurasi saat ini).
   * *Varian B:* Autoencoder dilatih *end-to-end* murni Cross-Entropy tanpa tahap rekonstruksi.
2. **Ablation 2 (Kapasitas Bottleneck Laten):**
   * *Varian A:* Bottleneck laten 16 dimensi (representasi terkompresi).
   * *Varian B:* Tanpa bottleneck (langsung 63 dimensi ke classifier).

## Hasil 3 Seed Konfigurasi Akhir (Target Sprint 2)
*(Akan dieksekusi 3 seed pada Sprint 2 setelah penyempurnaan data primer).*
* Seed 1: `val_acc = 0.1375`, `val_macro_f1 = 0.122253` (Sprint 1 milestone)
