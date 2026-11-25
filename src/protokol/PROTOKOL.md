# Protokol evaluasi tim — Ruang Cerdas 5V (Tim 13)

Diisi Penjaga Data & Evaluasi pada Sprint 0, digabung ke `main` lewat *pull request* yang
disetujui pembimbing, lalu **dibekukan**. Perubahan setelah itu hanya lewat *pull request*
baru yang disetujui pembimbing.

| Butir | Isi |
|---|---|
| Kasus (dari kartu Kasus tim) | Gestur tangan pengendali meja (kamera) — Ruang Cerdas 5 Volt |
| Sumber dataset | Rekaman web meja uji 18 Nov 2026 (`data/aktual/sesi-20261118-123649` + `data/aktual/kamera`); landmark MediaPipe 21 titik dari 658 frame kandidat |
| Jenis tugas (`TUGAS`) | klasifikasi (8 kelas gestur statis) |
| Target / label | 8 gestur: tangan terbuka, kepalan, jempol atas, jempol bawah, telunjuk atas, V (2 jari), OK, telapak hadap kamera lalu turun |
| Ukuran data latih / validasi / uji | 299 / 33 / — (total 332 sampel; partisipan P01, P02, P04, P05, P06 — P03/P07 frame hitam, P08 tidak terekam) |
| Cara split | `split.indeks` seed 2026, porsi validasi 10 % |
| Metrik utama untuk komparasi | macro-F1 dan akurasi (F1 per partisipan, ukuran ONNX KB, latensi CPU ms) |
| Praproses yang sama untuk semua model | Buang 1 detik pertama tiap kondisi; ekstraksi landmark MediaPipe 21 titik (x, y, z); cerminkan x bila tangan kiri; standarisasi per koordinat dengan statistik data latih |
| Anggaran komputasi per run | ≤ 20 menit CPU laptop meja |
| Evaluasi data uji | sekali, oleh Penjaga Data, pada sprint terakhir (`--final`) |
| `data_hash` saat dibekukan | `1150045b5a6138cdfab7328c10b42243` (dari `results/123450032/metrics.json`) |

Disetujui pembimbing pada: Sprint 0 (31 Oktober 2026)
Diperbarui Sprint 1 (25 November 2026) ke dataset primer `data/aktual/gestur_primer.npz`; menunggu persetujuan pembimbing lewat *pull request*.