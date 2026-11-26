"""Pemotong sesi simulasi Tim 13 (Sprint 1).

Paket ini berisi pipeline:
  rencana  -> membaca YAML rencana meja uji
  rekaman  -> membaca tar rekaman, indeks waktu, profil gerak, batas peserta
  deteksi  -> ekstraksi 21 landmark MediaPipe Hands + potongan ROI
  potong   -> memotong tiap kondisi, membuang 1 detik pertama, menyusun dataset
"""
__all__ = ["rencana", "rekaman", "deteksi", "potong"]
