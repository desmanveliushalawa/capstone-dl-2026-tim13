# LEMBAR PENCATATAN SESI MEJA UJI — TIM 13
**Kasus:** Gestur Tangan Pengendali Meja (Kamera) — Ruang Cerdas 5 Volt  
**Jadwal Slot:** Kamis, 12 November 2026 | Pukul: 11.00 – 12.00 WIB  
**Lokasi:** Ruang Prodi 1 (Pengelola Meja: Pak Ardika Satria)  
**Operator Utama:** Desman Velius Halawa & Bastian Heskia Silaban  

---

## 📌 ATURAN PENTING PENGAMBILAN DATA (Dari Kartu Kasus)
1. **8 Partisipan Berbeda:** 1 Partisipan = 1 Blok (P01 s.d. P08).
2. **Tunjukkan Kartu Gambar:** Tunjukkan kartu gambar 8 gestur kepada partisipan sebelum blok dimulai.
3. **Pencatatan Tangan:** Wajib mencatat tangan dominan yang digunakan (**Kanan / Kiri**). *Koordinat tangan kiri akan dicerminkan (flip horizontal) pada praproses agar seragam.*
4. **Siklus Gerak:** Partisipan menurunkan tangan pada jeda (0.5 detik), lalu mengangkat dan menahannya saat gestur baru tampil (3 detik).
5. **Pemotongan Jendela:** 1 detik pertama saat tangan naik akan dibuang secara otomatis oleh skrip praproses.

---

## 📋 TABEL PENCATATAN PARTISIPAN (P01 – P08)

| Kode Blok | Nama Partisipan | NIM / Prodi | Tangan (Kanan / Kiri) | 48 Kondisi Lengkap? (Ya/Tidak) | Catatan / Kendala Sensor | Tanda Tangan |
|---|---|---|---|---|---|---|
| **P01** | | | [ ] Kanan / [ ] Kiri | [ ] Lengkap | | |
| **P02** | | | [ ] Kanan / [ ] Kiri | [ ] Lengkap | | |
| **P03** | | | [ ] Kanan / [ ] Kiri | [ ] Lengkap | | |
| **P04** | | | [ ] Kanan / [ ] Kiri | [ ] Lengkap | | |
| **P05** | | | [ ] Kanan / [ ] Kiri | [ ] Lengkap | | |
| **P06** | | | [ ] Kanan / [ ] Kiri | [ ] Lengkap | | |
| **P07** | | | [ ] Kanan / [ ] Kiri | [ ] Lengkap | | |
| **P08** | | | [ ] Kanan / [ ] Kiri | [ ] Lengkap | | |

---

## 🖐️ DAFTAR 8 GESTUR STATIS (3 DETIK PER KONDISI)
* **0:** Tangan terbuka (telapak datar menghadap kamera)
* **1:** Kepalan (jari-jari menggenggam rapat)
* **2:** Jempol ke atas (*Thumbs Up*)
* **3:** Jempol ke bawah (*Thumbs Down*)
* **4:** Telunjuk ke atas (menunjuk vertikal)
* **5:** Dua jari membentuk huruf V (*Peace sign*)
* **6:** Tanda OK (ujung telunjuk menyentuh ujung jempol)
* **7:** Telapak menghadap kamera lalu bergerak turun

## 💡 KONDISI TARAF LAMPU MEJA (OTOMATIS ESP32)
* Taraf 1: `PWM 128` (Pencahayaan redup/sedang) — 3 kali ulangan per gestur
* Taraf 2: `PWM 255` (Pencahayaan terang maksimal) — 3 kali ulangan per gestur
* Total kondisi per orang: $8 \text{ gestur} \times 2 \text{ lampu} \times 3 \text{ ulangan} = 48 \text{ kondisi}$.
* Total seluruh sesi tim: $8 \text{ orang} \times 48 = 384 \text{ kondisi}$ (Estimasi waktu efektif $\approx 30$ menit).

---

## 🔍 PEMERIKSAAN SETELAH SESI SELESAI (Hari yang Sama)
- [ ] Seluruh data mentah sesi diterima dari pengelola meja / tersimpan di laptop.
- [ ] Jumlah rekaman pas 384 berkas / kondisi.
- [ ] Persentase frame tanpa deteksi tangan diperiksa antar taraf lampu 128 vs 255.
- [ ] Penjaga Data (Bastian) mengompilasi `gestur_primer.npz` dan memperbarui `data/MANIFEST.json`.
