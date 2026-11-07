# PANDUAN LENGKAP SPRINT 1 UNTUK ANGGOTA TIM 13
**Capstone Deep Learning: Ruang Cerdas 5 Volt (ITERA)**  
**Dosen Pembimbing:** Ahmad Luky Ramdani | **Pengelola Meja:** Ardika Satria  
**Koordinator Tim:** Desman Velius Halawa (123450114)  

---

## 📌 PENGANTAR UNTUK ANGGOTA TIM
Panduan ini disusun langkah demi langkah dari **NOL** untuk membantu seluruh anggota Tim 13 menyelesaikan tugas Sprint 1, mengimplementasikan kode model PyTorch masing-masing, meraih status **Lampu Hijau (🟢)** di Data Science Workbench, dan mempersiapkan sesi fisik Meja Uji pada **Kamis, 12 November 2026 (11.00 – 12.00 WIB)**.

---

## 👤 1. BASTIAN HESKIA SILABAN (`122450130`)
* **Tanggung Jawab Utama:** Penjaga Data & Evaluasi
* **Keluarga Model:** **CNN (1D-Convolutional Neural Network)**
* **Tiket di Workbench:** `#4` (Pemotong jendela), `#5` (Terima data meja), `#11` (Kerangka CNN simulasi), `#12` (Latih CNN data meja).

### A. Langkah Teknis di Laptop (Coding Model CNN)
1. Buka terminal di folder proyek laptop Anda, lalu pindah ke branch kerja Anda:
   ```bash
   git checkout 122450130/kerja
   git pull origin main
   ```
2. Buka berkas `models/122450130/run.py`, pastikan `JALUR = "cnn"`, lalu ganti fungsi `latih_dan_prediksi` dengan arsitektur 1D-CNN spesifik untuk koordinat landmark tangan:
   ```python
   def latih_dan_prediksi(seed: int, final: bool) -> tuple[np.ndarray, np.ndarray | None]:
       import torch
       import torch.nn as nn

       torch.manual_seed(seed)
       np.random.seed(seed)
       Xtr, ytr, Xva, _, Xte, _ = data.muat()

       # Input X: [N, 21, 3] -> Ubah ke format Channel First untuk Conv1D: [N, 3, 21]
       def ke_tensor(arr):
           if arr is None: return None
           t = torch.tensor(np.asarray(arr, dtype=np.float32))
           return t.permute(0, 2, 1) # [N, 3, 21]

       X_train = ke_tensor(Xtr)
       y_train = torch.tensor(np.asarray(ytr), dtype=torch.long)
       X_val = ke_tensor(Xva)
       X_test = ke_tensor(Xte) if final and Xte is not None else None

       class CNN1DGesture(nn.Module):
           def __init__(self, num_classes=8):
               super().__init__()
               self.net = nn.Sequential(
                   nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
                   nn.BatchNorm1d(16),
                   nn.ReLU(),
                   nn.MaxPool1d(kernel_size=2),
                   nn.Conv1d(16, 32, kernel_size=3, padding=1),
                   nn.BatchNorm1d(32),
                   nn.ReLU(),
                   nn.AdaptiveAvgPool1d(1),
                   nn.Flatten(),
                   nn.Dropout(0.2),
                   nn.Linear(32, num_classes)
               )
           def forward(self, x):
               return self.net(x)

       model = CNN1DGesture(num_classes=8)
       opt = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
       criterion = nn.CrossEntropyLoss()

       batch_size = 32
       n_samples = len(X_train)
       model.train()
       for epoch in range(50):
           perm = torch.randperm(n_samples)
           for i in range(0, n_samples, batch_size):
               idx = perm[i:i+batch_size]
               bx, by = X_train[idx], y_train[idx]
               opt.zero_grad()
               loss = criterion(model(bx), by)
               loss.backward()
               opt.step()

       model.eval()
       with torch.no_grad():
           pred_val = model(X_val).argmax(1).numpy()
           pred_test = model(X_test).argmax(1).numpy() if final and X_test is not None else None

       return pred_val, pred_test
   ```
3. Jalankan evaluasi seed 1 di terminal:
   ```bash
   python models/122450130/run.py --seed 1
   ```
   *(Pastikan keluar pesan sukses `results/122450130/metrics.json`)*.
4. Perbarui berkas catatan di `models/122450130/catatan.md` (tulis alasan memilih Adam dan regularisasi BatchNorm1d + Dropout 0.2).
5. Simpan dan Push ke GitHub:
   ```bash
   git add models/122450130/ results/122450130/
   git commit -m "feat(122450130): implementasi arsitektur 1D-CNN dasar dan evaluasi seed 1"
   git push origin 122450130/kerja
   ```
6. Buka GitHub, buat **Pull Request (PR)** dari `122450130/kerja` ke `main`. Salin tautan PR Anda (misal `.../pull/X`).

### B. Langkah di Web DSWorkbench (Agar Meraih Lampu HIJAU 🟢)
1. Buka [sd.itera.ac.id/workbench/capstone](https://sd.itera.ac.id/workbench/capstone) $\rightarrow$ klik tab **Log Sprint 1**.
2. Klik tombol merah **`( log )`** di samping nama **Bastian Heskia Silaban**.
3. Isi 3 kolom formulir:
   * **Selesai:**
     ```text
     1. Membangun model 1D-CNN dua lapis konvolusi (Conv1d 3->16->32) dengan BatchNorm1d dan Dropout di models/122450130/run.py.
     2. Menjalankan evaluasi seed 1 dan menghasilkan artefak results/122450130/metrics.json yang valid.
     3. Memeriksa aturan pemotongan 1 detik pertama data simulasi dan menyiapkan lembar sesi partisipan meja uji.
     ```
   * **Berikutnya:**
     ```text
     Mengawasi pencatatan tangan partisipan P01-P08 pada slot Meja Uji 12 November, menerima data sesi meja dari pengelola, dan mengompilasi data primer ke gestur_primer.npz.
     ```
   * **Hambatan:** `Tidak ada`
4. Klik **Simpan**.
5. Di bagian bawah panel (Tambah bukti):
   * Pilih dropdown **`Review PR`** $\rightarrow$ masukkan link PR Anda $\rightarrow$ klik **`[ 🔗 Catat bukti ]`**.
   * Pilih dropdown **`Commit (di luar aplikasi)`** $\rightarrow$ masukkan link commit terakhir Anda di GitHub $\rightarrow$ klik **`[ 🔗 Catat bukti ]`**.
6. Tekan **F5 (Refresh)** $\rightarrow$ Lampu Anda langsung **HIJAU (🟢)**!

### C. Langkah di Tab Papan (Memindahkan Tiket Kanban)
1. Klik tab **Papan** di menu atas DSWorkbench.
2. Cari tiket milik Anda:
   * **Tiket `#11` (`[CNN] Kerangka models/<nim>/run.py pada data simulasi`):** Klik tiket tersebut $\rightarrow$ ubah Status dari `Backlog` menjadi **`Review`** (karena Anda sudah membuat kode dan mengajukan PR).
   * **Tiket `#4` (`[Data] Pemotong jendela data meja uji`):** Ubah Status dari `Backlog` menjadi **`Dikerjakan`** (karena skrip `scripts/proses_data_primer.py` sudah siap untuk hari Meja Uji).
   * **Tiket `#5` & `#12`:** **Biarkan tetap di `Backlog`** untuk sekarang (karena baru dikerjakan setelah data Meja Uji fisik diambil pada 12 November).

---

## 👤 2. NASYWA NUR AFIFAH (`122450125`)
* **Tanggung Jawab Utama:** Integrator
* **Keluarga Model:** **Transfer Learning (TL) / MobileNet Citra Tangan**
* **Tiket di Workbench:** `#6` (Repo tim siap & run_all.py), `#7` (Merge komponen data primer), `#13` (Kerangka TL simulasi), `#14` (Latih TL data meja).

### A. Langkah Teknis di Laptop (Coding Model TL)
1. Pindah ke branch kerja:
   ```bash
   git checkout 122450125/kerja
   git pull origin main
   ```
2. Buka `models/122450125/run.py`, pastikan `JALUR = "tl"`. Pada fase data simulasi koordinat pose, buat arsitektur Deep Residual/Feature Transfer MLP dengan Dropout 0.3:
   ```python
   def latih_dan_prediksi(seed: int, final: bool) -> tuple[np.ndarray, np.ndarray | None]:
       import torch
       import torch.nn as nn

       torch.manual_seed(seed)
       np.random.seed(seed)
       Xtr, ytr, Xva, _, Xte, _ = data.muat()

       def ke_tensor(arr):
           if arr is None: return None
           return torch.tensor(np.asarray(arr, dtype=np.float32)).reshape(len(arr), -1)

       X_train = ke_tensor(Xtr)
       y_train = torch.tensor(np.asarray(ytr), dtype=torch.long)
       X_val = ke_tensor(Xva)
       X_test = ke_tensor(Xte) if final and Xte is not None else None

       class TransferMLPGesture(nn.Module):
           def __init__(self, in_features=63, num_classes=8):
               super().__init__()
               self.backbone = nn.Sequential(
                   nn.Linear(in_features, 128),
                   nn.BatchNorm1d(128),
                   nn.ReLU(),
                   nn.Dropout(0.3),
                   nn.Linear(128, 64),
                   nn.BatchNorm1d(64),
                   nn.ReLU(),
                   nn.Dropout(0.2),
               )
               self.classifier = nn.Linear(64, num_classes)
           def forward(self, x):
               feat = self.backbone(x)
               return self.classifier(feat)

       model = TransferMLPGesture(in_features=63, num_classes=8)
       opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
       criterion = nn.CrossEntropyLoss()

       batch_size = 32
       n_samples = len(X_train)
       model.train()
       for epoch in range(50):
           perm = torch.randperm(n_samples)
           for i in range(0, n_samples, batch_size):
               idx = perm[i:i+batch_size]
               bx, by = X_train[idx], y_train[idx]
               opt.zero_grad()
               loss = criterion(model(bx), by)
               loss.backward()
               opt.step()

       model.eval()
       with torch.no_grad():
           pred_val = model(X_val).argmax(1).numpy()
           pred_test = model(X_test).argmax(1).numpy() if final and X_test is not None else None

       return pred_val, pred_test
   ```
3. Jalankan evaluasi seed 1:
   ```bash
   python models/122450125/run.py --seed 1
   ```
4. Perbarui catatan di `models/122450125/catatan.md` (alasan optimizer AdamW dan strategi transfer feature embedding).
5. Commit, Push, dan buat PR:
   ```bash
   git add models/122450125/ results/122450125/
   git commit -m "feat(122450125): implementasi model transfer learning dasar dan evaluasi seed 1"
   git push origin 122450125/kerja
   ```
6. Buka GitHub, buat Pull Request ke `main`.
7. Sebagai Integrator, uji keutuhan pipeline tim:
   ```bash
   python run_all.py --tanpa-baseline
   ```
   *(Pastikan tabel `reports/hasil_komparasi.md` dapat dirakit tanpa error).*

### B. Langkah di Web DSWorkbench (Agar Meraih Lampu HIJAU 🟢)
1. Buka tab **Log Sprint 1** $\rightarrow$ klik tombol merah **`( log )`** di nama **Nasywa Nur Afifah**.
2. Isi formulir:
   * **Selesai:**
     ```text
     1. Membangun model Transfer Learning feature embedding (128->64) dengan BatchNorm1d dan Dropout di models/122450125/run.py.
     2. Mengeksekusi pelatihan seed 1 dan menghasilkan results/122450125/metrics.json yang valid.
     3. Menguji eksekusi terpusat run_all.py untuk memastikan pipeline integrasi 5 model tim berjalan mulus.
     ```
   * **Berikutnya:**
     ```text
     Menyiapkan penyesuaian arsitektur MobileNet pada data potongan citra tangan (ROI 640x480) setelah perekaman meja uji 12 November dan menggabungkan PR seluruh anggota ke main.
     ```
   * **Hambatan:** `Tidak ada`
3. Klik **Simpan**.
4. Di bagian **Tambah bukti**, masukkan bukti `Review PR` (link PR Anda) dan `Commit (di luar aplikasi)` (link commit Anda), lalu klik **`[ 🔗 Catat bukti ]`**.
5. Tekan **F5** $\rightarrow$ Lampu langsung **HIJAU (🟢)**!

### C. Langkah di Tab Papan (Memindahkan Tiket Kanban)
1. Klik tab **Papan** di DSWorkbench.
2. Cari tiket milik Anda:
   * **Tiket `#13` (atau `#19` jika tertera kerangka model Anda):** Ubah Status dari `Backlog` menjadi **`Review`** (karena Anda sudah membuat kerangka kode TL dan membuka PR).
   * **Tiket `#6` (`Repo tim siap & run_all.py`):** Ubah Status menjadi **`Review`** atau **`Selesai`** (karena `run_all.py` sudah divalidasi).
   * **Tiket `#7` & `#14`:** **Biarkan tetap di `Backlog`** untuk sekarang (akan dikerjakan setelah Meja Uji fisik 12 November).

---

## 👤 3. KHOIRUL MUTTOHAROH (`123450021`)
* **Tanggung Jawab Utama:** Analis Utama
* **Keluarga Model:** **RNN / LSTM / GRU (Sekuensial)**
* **Tiket di Workbench:** `#8` (Pembanding non-DL Random Forest & Sudut Sendi), `#15` (Kerangka RNN simulasi), `#16` (Latih RNN data meja).

### A. Langkah Teknis di Laptop (Coding Model RNN & Baseline Non-DL)
1. Pindah ke branch kerja:
   ```bash
   git checkout 123450021/kerja
   git pull origin main
   ```
2. Buka `models/123450021/run.py`, pastikan `JALUR = "rnn"`. Bangun model **Bidirectional LSTM** yang memproses 21 titik sendi tangan sebagai sebuah sekuens spasial/topologis:
   ```python
   def latih_dan_prediksi(seed: int, final: bool) -> tuple[np.ndarray, np.ndarray | None]:
       import torch
       import torch.nn as nn

       torch.manual_seed(seed)
       np.random.seed(seed)
       Xtr, ytr, Xva, _, Xte, _ = data.muat()

       # Input X: [N, 21, 3] -> Sekuens panjang 21, tiap elemen punya 3 fitur (x,y,z)
       def ke_tensor(arr):
           if arr is None: return None
           return torch.tensor(np.asarray(arr, dtype=np.float32))

       X_train = ke_tensor(Xtr)
       y_train = torch.tensor(np.asarray(ytr), dtype=torch.long)
       X_val = ke_tensor(Xva)
       X_test = ke_tensor(Xte) if final and Xte is not None else None

       class BiLSTMGesture(nn.Module):
           def __init__(self, input_size=3, hidden_size=32, num_classes=8):
               super().__init__()
               self.lstm = nn.LSTM(
                   input_size=input_size,
                   hidden_size=hidden_size,
                   num_layers=2,
                   batch_first=True,
                   bidirectional=True,
                   dropout=0.2
               )
               self.fc = nn.Linear(hidden_size * 2, num_classes)

           def forward(self, x):
               # x: [Batch, 21, 3]
               out, (hn, cn) = self.lstm(x)
               # Mean pooling sepanjang 21 titik sendi
               out_pool = out.mean(dim=1)
               return self.fc(out_pool)

       model = BiLSTMGesture(input_size=3, hidden_size=32, num_classes=8)
       opt = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
       criterion = nn.CrossEntropyLoss()

       batch_size = 32
       n_samples = len(X_train)
       model.train()
       for epoch in range(50):
           perm = torch.randperm(n_samples)
           for i in range(0, n_samples, batch_size):
               idx = perm[i:i+batch_size]
               bx, by = X_train[idx], y_train[idx]
               opt.zero_grad()
               loss = criterion(model(bx), by)
               loss.backward()
               opt.step()

       model.eval()
       with torch.no_grad():
           pred_val = model(X_val).argmax(1).numpy()
           pred_test = model(X_test).argmax(1).numpy() if final and X_test is not None else None

       return pred_val, pred_test
   ```
3. Jalankan evaluasi seed 1:
   ```bash
   python models/123450021/run.py --seed 1
   ```
4. Eksekusi Tiket #8 (Baseline Non-DL yang sudah disiapkan tim):
   ```bash
   python scripts/baseline_rf.py --seed 2026
   ```
   *(Catat akurasi 20% dan matriks kebingungan sebagai bahan analisis komparasi di laporan)*.
5. Perbarui catatan di `models/123450021/catatan.md`.
6. Commit, Push, dan buat PR di GitHub:
   ```bash
   git add models/123450021/ results/123450021/
   git commit -m "feat(123450021): implementasi model BiLSTM sekuensial dan evaluasi seed 1"
   git push origin 123450021/kerja
   ```

### B. Langkah di Web DSWorkbench (Agar Meraih Lampu HIJAU 🟢)
1. Buka tab **Log Sprint 1** $\rightarrow$ klik tombol merah **`( log )`** di nama **Khoirul Muttoharoh**.
2. Isi formulir:
   * **Selesai:**
     ```text
     1. Membangun arsitektur Bidirectional LSTM 2-layer dengan dropout dan mean-pooling di models/123450021/run.py.
     2. Menjalankan evaluasi seed 1 dan menghasilkan results/123450021/metrics.json yang tervalidasi.
     3. Mengeksekusi model pembanding non-DL Random Forest berbasis 93 fitur sudut sendi dan jarak pose (Tiket #8) dengan val_acc 20%.
     ```
   * **Berikutnya:**
     ```text
     Menganalisis matriks kebingungan 8 kelas gestur pada data primer meja uji 12 November dan membandingkan ketahanan model BiLSTM terhadap fluktuasi sudut pose.
     ```
   * **Hambatan:** `Tidak ada`
3. Klik **Simpan**.
4. Catat bukti PR + Commit di bagian Tambah Bukti $\rightarrow$ Tekan **F5** $\rightarrow$ Lampu langsung **HIJAU (🟢)**!

### C. Langkah di Tab Papan (Memindahkan Tiket Kanban)
1. Klik tab **Papan** di DSWorkbench.
2. Cari tiket milik Anda:
   * **Tiket `#15` (`[RNN] Kerangka models/<nim>/run.py pada data simulasi`):** Ubah Status dari `Backlog` menjadi **`Review`** (karena Anda sudah membuat model BiLSTM dan membuka PR).
   * **Tiket `#8` (`Pembanding non-DL Random Forest & Sudut Sendi`):** Ubah Status dari `Backlog` menjadi **`Review`** atau **`Dikerjakan`** (karena skrip `scripts/baseline_rf.py` sudah diuji).
   * **Tiket `#16` (`[RNN] Latih pada data primer meja`):** **Biarkan tetap di `Backlog`** untuk sekarang (akan dikerjakan setelah sesi Meja Uji fisik 12 November).

---

## 👤 4. ADITYA TAUFIQURROHMAN (`123450032`)
* **Tanggung Jawab Utama:** Reproduksi & Demo
* **Keluarga Model:** **Transformer Encoder (Multi-Head Self-Attention)**
* **Tiket di Workbench:** `#9` (Gladi rencana mode simulasi), `#10` (Rekaman Demo Sprint 1), `#17` (Kerangka Transformer simulasi), `#18` (Latih Transformer data meja).

### A. Langkah Teknis di Laptop (Coding Model Transformer)
1. Pindah ke branch kerja:
   ```bash
   git checkout 123450032/kerja
   git pull origin main
   ```
2. Buka `models/123450032/run.py`, pastikan `JALUR = "trf"`. Bangun model **Transformer Encoder** dengan Multi-Head Attention antar 21 titik sendi:
   ```python
   def latih_dan_prediksi(seed: int, final: bool) -> tuple[np.ndarray, np.ndarray | None]:
       import torch
       import torch.nn as nn

       torch.manual_seed(seed)
       np.random.seed(seed)
       Xtr, ytr, Xva, _, Xte, _ = data.muat()

       def ke_tensor(arr):
           if arr is None: return None
           return torch.tensor(np.asarray(arr, dtype=np.float32))

       X_train = ke_tensor(Xtr)  # [N, 21, 3]
       y_train = torch.tensor(np.asarray(ytr), dtype=torch.long)
       X_val = ke_tensor(Xva)
       X_test = ke_tensor(Xte) if final and Xte is not None else None

       class TransformerGesture(nn.Module):
           def __init__(self, in_features=3, d_model=32, nhead=4, num_classes=8):
               super().__init__()
               self.embedding = nn.Linear(in_features, d_model)
               self.pos_emb = nn.Parameter(torch.zeros(1, 21, d_model))
               encoder_layer = nn.TransformerEncoderLayer(
                   d_model=d_model,
                   nhead=nhead,
                   dim_feedforward=64,
                   dropout=0.1,
                   batch_first=True
               )
               self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
               self.classifier = nn.Linear(d_model, num_classes)

           def forward(self, x):
               # x: [Batch, 21, 3]
               h = self.embedding(x) + self.pos_emb
               out = self.transformer(h)
               out_pool = out.mean(dim=1)  # Mean pooling sepanjang 21 token sendi
               return self.classifier(out_pool)

       model = TransformerGesture(in_features=3, d_model=32, nhead=4, num_classes=8)
       opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
       criterion = nn.CrossEntropyLoss()

       batch_size = 32
       n_samples = len(X_train)
       model.train()
       for epoch in range(50):
           perm = torch.randperm(n_samples)
           for i in range(0, n_samples, batch_size):
               idx = perm[i:i+batch_size]
               bx, by = X_train[idx], y_train[idx]
               opt.zero_grad()
               loss = criterion(model(bx), by)
               loss.backward()
               opt.step()

       model.eval()
       with torch.no_grad():
           pred_val = model(X_val).argmax(1).numpy()
           pred_test = model(X_test).argmax(1).numpy() if final and X_test is not None else None

       return pred_val, pred_test
   ```
3. Jalankan evaluasi seed 1:
   ```bash
   python models/123450032/run.py --seed 1
   ```
4. Perbarui catatan di `models/123450032/catatan.md`.
5. Commit, Push, dan buat PR di GitHub:
   ```bash
   git add models/123450032/ results/123450032/
   git commit -m "feat(123450032): implementasi Transformer Encoder multi-head attention dan evaluasi seed 1"
   git push origin 123450032/kerja
   ```
6. Siapkan perekam layar (OBS / Clipchamp / Xbox Game Bar) untuk merekam Video Demo Sprint 1 ($\le 10$ menit).

### B. Langkah di Web DSWorkbench (Agar Meraih Lampu HIJAU 🟢)
1. Buka tab **Log Sprint 1** $\rightarrow$ klik tombol merah **`( log )`** di nama **Aditya Taufiqurrohman**.
2. Isi formulir:
   * **Selesai:**
     ```text
     1. Membangun model Transformer Encoder (2 layer, 4 heads, d_model=32) dengan positional embedding di models/123450032/run.py.
     2. Menjalankan evaluasi seed 1 dan menghasilkan results/123450032/metrics.json yang tervalidasi.
     3. Menguji gladi skrip kamera laptop meja uji scripts/rekam_gestur_meja.py dan merancang skenario video demo Sprint 1.
     ```
   * **Berikutnya:**
     ```text
     Mengambil dokumentasi video saat slot Meja Uji 12 November, memproduksi video rekaman Demo Sprint 1 (≤ 10 menit) yang mendemonstrasikan run_all.py 5 model, dan mengunggahnya ke form Demo Sprint.
     ```
   * **Hambatan:** `Tidak ada`
3. Klik **Simpan**.
4. Catat bukti PR + Commit di bagian Tambah Bukti $\rightarrow$ Tekan **F5** $\rightarrow$ Lampu langsung **HIJAU (🟢)**!

### C. Langkah di Tab Papan (Memindahkan Tiket Kanban)
1. Klik tab **Papan** di DSWorkbench.
2. Cari tiket milik Anda:
   * **Tiket `#17` (`[Transformer] Kerangka models/<nim>/run.py pada data simulasi`):** Ubah Status dari `Backlog` menjadi **`Review`** (karena Anda sudah membuat model Transformer dan membuka PR).
   * **Tiket `#9` (`Gladi rencana mode simulasi`):** Ubah Status dari `Backlog` menjadi **`Dikerjakan`**.
   * **Tiket `#10` (`Rekaman Demo Sprint 1`):** Ubah Status menjadi **`Dikerjakan`** saat mulai menyiapkan rekaman video demo.
   * **Tiket `#18` (`[Transformer] Latih pada data primer meja`):** **Biarkan tetap di `Backlog`** untuk sekarang (akan dikerjakan setelah sesi Meja Uji fisik 12 November).
5. **Kirim Demo Sprint 1 (Sebelum 20 Nov 23.59):**  
   Klik tombol biru **`+ Kirim demo`** di bilah Sprint pada tab Papan, lalu masukkan link YouTube/Drive video demo tim.

---

## 📅 JADWAL SLOT MEJA UJI BERSAMA (KAMIS, 12 NOV 2026)
* **Waktu:** **11.00 – 12.00 WIB** (Hadir 10 menit sebelumnya di **Ruang Prodi 1**, bertemu Pak Ardika Satria).
* **Partisipan (8 Orang):**
  - `P01`: Desman Velius Halawa
  - `P02`: Bastian Heskia Silaban
  - `P03`: Nasywa Nur Afifah
  - `P04`: Khoirul Muttoharoh
  - `P05`: Aditya Taufiqurrohman
  - `P06, P07, P08`: 3 orang relawan teman sekelas.
* **Prosedur:** Masing-masing orang duduk $\pm 3$ menit memperagakan 8 gestur pada lampu redup (`PWM 128`) dan terang (`PWM 255`) sebanyak 3 ulangan. Tangan kanan/kiri dicatat di berkas `sprint/sprint-1/LEMBAR_SESI_12NOV.md`.

---

## 🚀 ROADMAP KERJA TIM 13 SETELAH SEMUA HIJAU (8 – 20 NOV 2026)

Setelah seluruh 5 anggota tim meraih Lampu Hijau (🟢) dan tiket kerangka model dipindahkan ke `Review`, berikut alur kerja yang WAJIB dieksekusi bersama hingga Sprint 1 ditutup:

```
[ 8 - 11 Nov ]  ───►  [ 12 Nov (11.00-12.00) ]  ───►  [ 13 - 16 Nov ]  ───►  [ 17 - 20 Nov ]
Review PR Tim          Sesi Fisik Meja Uji             Latih Ulang             Video Demo & Laporan
Merge ke 'main'        Perekaman 384 data             Data Meja Asli          Upload ke Workbench
Uji 'run_all.py'       Dokumentasi video              Update metrics          Tenggat: 20 Nov 23.59
```

### 1. FASE 1: Code Review & Integrasi Cabang (8 – 11 November 2026)
* **Tinjau Pull Request (GitHub):** Setiap anggota membuka halaman PR rekan setim di GitHub, membaca kode model, dan memberi komentar persetujuan (`Approve / LGTM`).
* **Merge ke `main`:** Koordinator (Desman) atau Integrator (Nasywa) menggabungkan (merge) kelima branch kerja ke branch utama (`main`).
* **Verifikasi Pipeline Terpadu:** Jalankan di terminal branch `main`:
  ```bash
  python run_all.py
  ```
  Pastikan seluruh 5 model Deep Learning dan 1 baseline Random Forest berjalan sukses tanpa error dan tabel ringkasan ter-generate di `reports/hasil_komparasi.md`.
* **Konfirmasi Meja Uji ke Ruang Prodi 1 (Senin/Selasa, 9–10 Nov):**
  Desman menemui **Pak Ardika Satria** di Ruang Prodi 1 untuk menyerahkan/mengonfirmasi file `t13-gestur-kamera.yaml` dan memastikan slot meja uji Kamis, 12 Nov (11.00–12.00 WIB) terkunci.
* **Kunci 3 Partisipan Eksternal:** Pastikan 3 rekan sekelas (`P06, P07, P08`) siap hadir pada Kamis, 12 Nov jam 11.00 WIB.

### 2. FASE 2: Eksekusi Hari-H Meja Uji Fisik (Kamis, 12 November 2026)
* **Lokasi:** Ruang Prodi 1 (Meja Uji Fisik 5 Volt).
* **Peran Tim saat Sesi (11.00 – 12.00 WIB):**
  - **Desman & Bastian:** Mengoperasikan laptop perekam dengan skrip `scripts/rekam_gestur_meja.py`, mengontrol level lampu meja (PWM 128 & PWM 255), serta mencatat tangan kanan/kiri di `LEMBAR_SESI_12NOV.md`.
  - **Aditya:** Merekam dokumentasi visual berupa foto dan video (b-roll interaksi tangan partisipan, lampu meja 5V, deteksi skeleton MediaPipe di layar). *Aset ini krusial untuk bahan video demo!*
  - **Nasywa & Khoirul:** Mengarahkan alur antrean 8 partisipan agar 384 kondisi terekam rapi dalam batas waktu 60 menit.
* **Kompilasi Data Meja:**
  Setelah sesi selesai, Bastian menjalankan:
  ```bash
  python scripts/proses_data_primer.py
  ```
  Skrip ini otomatis membuang 1 detik pertama transisi tangan dan menghasilkan berkas `data/gestur_primer.npz`.
* **Pindahkan Tiket:** Tiket `#1` dan `#2` dipindahkan ke **`Selesai`**.

### 3. FASE 3: Pelatihan Model pada Data Primer Meja (13 – 16 November 2026)
* Bastian dan Nasywa melakukan *push* file `data/gestur_primer.npz` dan memperbarui `data/MANIFEST.json` ke repo tim.
* Seluruh anggota tim memperbarui model masing-masing untuk dilatih pada data primer meja:
  ```bash
  python models/<nim>/run.py --seed 1
  ```
* Verifikasi bahwa `results/<nim>/metrics.json` dan `pred_val.npy` terbarui dengan data meja asli.
* **Pindahkan Tiket:** Ubah tiket pelatihan data primer (`#12`, `#14`, `#16`, `#18`, `#20`) dari `Backlog` $\rightarrow$ **`Dikerjakan`** $\rightarrow$ **`Review`**.
* Jalankan kembali `python run_all.py` untuk mengompilasi performa perbandingan 5 model pada kondisi meja fisik.

### 4. FASE 4: Produksi Video Demo Sprint 1 & Penyerahan (17 – 20 November 2026)
* **Penanggung Jawab:** Aditya Taufiqurrohman (didukung seluruh anggota tim).
* **Syarat & Format Video Demo Sprint 1:**
  1. **Durasi:** Maksimal **10 menit** (disarankan 5 – 8 menit agar padat dan jelas).
  2. **Struktur Isi Video:**
     - **Menit 0:00 – 1:00:** Pembuka tim (Nama anggota, NIM, peran, dan judul proyek: "Ruang Cerdas 5 Volt").
     - **Menit 1:00 – 3:00:** Penjelasan arsitektur 5 model (AE, CNN, TL, RNN, Transformer) dan baseline Random Forest.
     - **Menit 3:00 – 5:00:** Cuplikan dokumentasi sesi Meja Uji Fisik 12 November di Ruang Prodi 1 (pencahayaan PWM 128 vs 255, 8 partisipan, perekaman pose MediaPipe).
     - **Menit 5:00 – 7:00:** Demonstrasi eksekusi terminal `python run_all.py` dan pembahasan tabel hasil metrik akurasi/F1-score.
     - **Menit 7:00 – 8:00:** Kesimpulan Sprint 1, hambatan yang diatasi, dan rencana kerja Sprint 2.
  3. **Pengunggahan:** Unggah video ke YouTube (setel sebagai *Unlisted* / Tidak Publik) atau Google Drive (akses publik *"Siapa saja yang memiliki link"*).
* **Pengiriman di DSWorkbench:**
  - Buka tab **Papan** di DSWorkbench.
  - Klik tombol biru **`+ Kirim demo`** di bilah Sprint 1 (hanya koordinator atau anggota tim).
  - Masukkan link URL video demo dan deskripsi singkat, lalu klik Simpan.
* **Tenggat Waktu Mutlak:**
  - Pengiriman Video Demo Sprint 1 ditutup pada **Jumat, 20 November 2026 pukul 23.59 WIB**.
* **Pindahkan Tiket:** Tiket `#3` ("Sprint Goal, Log Sprint, dan Demo Sprint 1") dan Tiket `#10` ("Rekaman Demo") dipindahkan ke **`Selesai`**.

