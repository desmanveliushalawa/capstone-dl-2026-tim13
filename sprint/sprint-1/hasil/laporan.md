# Hasil evaluasi baseline — sesi simulasi Sprint 1

Metrik utama: **F1-makro per peserta**. Split GroupKFold per peserta pada P01, P02, P03, P04; data uji akhir `P05` disisihkan dan belum dievaluasi.

## Ringkasan (rerata antar-peserta, 3 seed)

| Model | F1-makro | Simpangan baku | Akurasi |
|---|---|---|---|
| RF | 0.082 | 0.057 | 0.098 |
| MLP | 0.084 | 0.018 | 0.127 |

## F1-makro per peserta

| Peserta | n uji | RF F1 | RF akurasi | MLP F1 | MLP akurasi |
|---|---|---|---|---|---|
| P01 | 45 | 0.126 ± 0.013 | 0.170 | 0.082 ± 0.054 | 0.148 |
| P02 | 39 | 0.000 ± 0.000 | 0.000 | 0.079 ± 0.052 | 0.137 |
| P03 | 82 | 0.115 ± 0.029 | 0.122 | 0.109 ± 0.077 | 0.138 |
| P04 | 83 | 0.088 ± 0.018 | 0.100 | 0.065 ± 0.035 | 0.084 |

## Validasi split beku (P04), uji akhir P05 disisihkan

| Model | F1-makro validasi | Simpangan baku |
|---|---|---|
| RF | 0.082 | 0.007 |
| MLP | 0.121 | 0.025 |

## Catatan keterbatasan

- Rekaman yang tersedia berjarak **1 detik/frame** (selangMs=1000), bukan 0,1 detik seperti rencana. Setelah membuang 1 detik pertama, tiap kondisi hanya menyisakan ~1–2 frame.
- Hanya 5 blok peserta terdeteksi dari 8 yang direncanakan; 3 blok terakhir tidak ada di rekaman.
- Label gestur mengikuti urutan rencana karena rekaman tidak menyertakan label.csv sesi; kesalahan penyelarasan slot mungkin terjadi.
- Angka di atas adalah verifikasi jalur evaluasi, bukan hasil final. Data uji akhir P05 baru boleh dievaluasi sekali pada sprint terakhir.
