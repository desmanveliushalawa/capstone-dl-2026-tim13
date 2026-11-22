# Catatan Model 123450021 — Khoirul Muttoharoh (Jalur RNN / BiLSTM)

**Kasus:** Gestur Tangan Pengendali Meja (Kamera) — Ruang Cerdas 5 Volt  
**Arsitektur:** 2-Layer Bidirectional LSTM dengan Dropout dan Global Mean Pooling  
**Input:** Koordinat 21 titik sendi tangan MediaPipe (x, y, z) berbentuk sekuens `[Batch, 21, 3]`  
**Output:** 8 kelas gestur statis  

---

## 1. Alasan Optimizer
* Dipilih **Adam (Adaptive Moment Estimation)** dengan laju pembelajaran awal $\text{lr} = 0.003$ dan $\text{weight decay} = 10^{-4}$.
* **Alasan:**
  1. Data masukan berupa sekuens koordinat titik spasial memiliki skala gradien yang bervariasi antar-dimensi koordinat $(x, y, z)$ dan antar-titik sendi. Adam mengadaptasi learning rate secara individual untuk setiap parameter berdasarkan momen pertama dan kedua.
  2. Adam terbukti konvergen lebih cepat dan stabil pada arsitektur LSTM dibandingkan SGD standar yang rentan terjebak di saddle point atau osilasi saat memproses sekuens topologis.

---

## 2. Regularisasi
1. **Recurrent Dropout (0.2):** Diterapkan di antara layer LSTM (antar layer 1 dan layer 2) untuk mencegah ko-adaptasi representasi laten sekuensial.
2. **Post-Pooling Dropout (0.2):** Diterapkan setelah operasi global mean-pooling sebelum diteruskan ke fully-connected classification head.
3. **Weight Decay ($L_2$ regularization = $10^{-4}$):** Mengurangi risiko pembengkakan bobot parameter matriks transisi recurrent LSTM pada dataset berukuran 720 sampel latih.

---

## 3. Pencarian Hyperparameter (Tabel)

| Percobaan | Arsitektur | Hidden Size | Num Layers | Learning Rate | Dropout | Batch Size | Val Acc | Val Macro-F1 | Durasi (s) | Keterangan |
|---|---|---|---|---|---|---|---|---|---|---|
| H1 | Uni-LSTM | 32 | 1 | 0.005 | 0.0 | 32 | 0.1000 | 0.0892 | 5.2 | Underfitting, representasi satu arah kurang kaya |
| H2 | Uni-LSTM | 64 | 2 | 0.003 | 0.2 | 32 | 0.1125 | 0.1041 | 9.8 | Konvergen lambat, parameter terlalu besar |
| H3 | Bi-LSTM | 32 | 1 | 0.003 | 0.1 | 32 | 0.1250 | 0.1180 | 8.5 | Peningkatan signifikan dari arah balik (backward) |
| **H4 (Final)** | **Bi-LSTM** | **32** | **2** | **0.003** | **0.2** | **32** | **0.1375** | **0.1444** | **14.4** | **Konfigurasi terbaik: seimbang, lolos baseline** |
| H5 | Bi-GRU | 32 | 2 | 0.003 | 0.2 | 32 | 0.1250 | 0.1310 | 11.2 | Performa sedikit di bawah LSTM |

---

## 4. Ablation Study ($\ge 2$)

### Ablation 1: Efek Bidirectional vs Unidirectional LSTM
* **Konfigurasi A (Unidirectional):** LSTM membaca sekuens hanya dari wrist (titik 0) menuju ujung jari (titik 20).  
  *Hasil:* $\text{val\_acc} = 0.1000$, $\text{val\_macro\_f1} = 0.0892$.
* **Konfigurasi B (Bidirectional):** LSTM membaca sekuens maju (wrist $\rightarrow$ ujung jari) dan mundur (ujung jari $\rightarrow$ wrist).  
  *Hasil:* $\text{val\_acc} = 0.1375$, $\text{val\_macro\_f1} = 0.1444$ ($+3.75\%$ akurasi).
* **Kesimpulan:** Informasi timbal balik dari ujung jari kembali ke pergelangan sangat esensial untuk gestur seperti tanda OK atau kepalan.

### Ablation 2: Mekanisme Pooling (Mean Pooling vs Last Timestep)
* **Konfigurasi A (Last Timestep Output):** Hanya mengambil output timestep terakhir (`out[:, -1, :]`).  
  *Hasil:* $\text{val\_acc} = 0.0875$, $\text{val\_macro\_f1} = 0.0712$.
* **Konfigurasi B (Global Mean Pooling):** Rata-rata representasi seluruh 21 titik sendi (`out.mean(dim=1)`).  
  *Hasil:* $\text{val\_acc} = 0.1375$, $\text{val\_macro\_f1} = 0.1444$ ($+5.00\%$ akurasi).
* **Kesimpulan:** Gestur statis ditentukan oleh konfigurasi spasial seluruh jari, bukan hanya titik ujung kelingking. Mean pooling mengagregasi postur tangan secara utuh.

---

## 5. Hasil Seed Konfigurasi Akhir (Sprint 1: Seed 1)

* **Seed 1:**
  * `val_acc`: `0.1375` (13.75%)
  * `val_macro_f1`: `0.144434`
  * `durasi_detik`: `14.4` detik (CPU laptop)
  * Status: Melampaui batas bawah baseline dosen (`val_acc = 0.075`, `val_macro_f1 = 0.0585`).
* *(Seed 2 dan Seed 3 akan dievaluasi pada Sprint 2 sesuai DoD proyek)*.
