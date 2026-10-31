# Protokol evaluasi tim — Ruang Cerdas 5V (Tim 13)

Diisi Penjaga Data & Evaluasi pada Sprint 0, digabung ke `main` lewat *pull request* yang
disetujui pembimbing, lalu **dibekukan**. Perubahan setelah itu hanya lewat *pull request*
baru yang disetujui pembimbing.

| Butir | Isi |
|---|---|
| Kasus (dari kartu Kasus tim) | Gestur tangan pengendali meja (kamera) — Ruang Cerdas 5 Volt |
| Sumber dataset | Rekaman data primer meja uji (10 Nov 2026); sampel awal gestur tangan MediaPipe |
| Jenis tugas (`TUGAS`) | klasifikasi (8 kelas gestur statis) |
| Target / label | 8 gestur: tangan terbuka, kepalan, jempol atas, jempol bawah, telunjuk atas, V (2 jari), OK, telapak hadap kamera lalu turun |
| Ukuran data latih / validasi / uji | 720 / 80 / — (total 800 sampel) |
| Cara split | `split.indeks` seed 2026, porsi validasi 10 % |
| Metrik utama untuk komparasi | macro-F1 dan akurasi (F1 per partisipan, ukuran ONNX KB, latensi CPU ms) |
| Praproses yang sama untuk semua model | Standarisasi fitur 21 titik koordinat MediaPipe (x, y, z) |
| Anggaran komputasi per run | ≤ 20 menit CPU laptop meja |
| Evaluasi data uji | sekali, oleh Penjaga Data, pada sprint terakhir (`--final`) |
| `data_hash` saat dibekukan | `148ea43c92e5045c3713e47e851bbe79` (dari `results/baseline/metrics.json`) |

Disetujui pembimbing pada: Sprint 0 (31 Oktober 2026)