"""Model Autoencoder (AE) untuk Klasifikasi 8 Gestur Tangan Meja 5V.
NIM: 123450114 (Desman Velius Halawa - Tim 13)

Penggunaan:
    python models/123450114/run.py --seed 1
    python models/123450114/run.py --seed 1 --final
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))
from src.protokol import data  # noqa: E402  (protokol tim yang dibekukan)

NIM = Path(__file__).resolve().parent.name     # nama folder = NIM
JALUR = "ae"                                   # cnn | tl | rnn | trf | ae


def latih_dan_prediksi(seed: int, final: bool) -> tuple[np.ndarray, np.ndarray | None]:
    """Melatih Autoencoder (rekonstruksi laten) + Classifier Head untuk prediksi gestur."""
    import torch
    import torch.nn as nn

    # 1. Reproduksibilitas seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 2. Muat data melalui protokol resmi tim yang dibekukan
    Xtr, ytr, Xva, _, Xte, _ = data.muat()

    def ke_tensor(arr: np.ndarray | None) -> torch.Tensor | None:
        if arr is None:
            return None
        return torch.tensor(np.asarray(arr, dtype=np.float32)).reshape(len(arr), -1)

    X_train = ke_tensor(Xtr)
    y_train = torch.tensor(np.asarray(ytr), dtype=torch.long)
    X_val = ke_tensor(Xva)
    X_test = ke_tensor(Xte) if final and Xte is not None else None

    in_dim = X_train.shape[1]  # 21 titik x 3 koordinat = 63 fitur
    num_classes = 8            # 8 gestur tangan statis

    # 3. Definisi Arsitektur Autoencoder
    #    Encoder: 63 -> 32 -> 16 (representasi laten tahan noise getaran sensor)
    #    Decoder: 16 -> 32 -> 63 (rekonstruksi titik koordinat)
    #    Classifier: 16 -> 8 (klasifikasi gestur berbasis representasi laten)
    class AutoencoderGesture(nn.Module):
        def __init__(self, in_features: int = 63, latent_dim: int = 16, n_classes: int = 8):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(in_features, 32),
                nn.ReLU(),
                nn.Linear(32, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 32),
                nn.ReLU(),
                nn.Linear(32, in_features),
            )
            self.classifier = nn.Linear(latent_dim, n_classes)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            z = self.encoder(x)
            x_rec = self.decoder(z)
            logits = self.classifier(z)
            return z, x_rec, logits

    model = AutoencoderGesture(in_features=in_dim, latent_dim=16, n_classes=num_classes)

    # 4. Pelatihan Tahap 1: Pre-training Rekonstruksi Autoencoder (Unsupervised)
    #    Tujuan: Memaksa bottleneck laten (16D) mempelajari struktur koordinat tangan yang valid
    opt_recon = torch.optim.AdamW(
        list(model.encoder.parameters()) + list(model.decoder.parameters()),
        lr=0.005,
        weight_decay=1e-4,
    )
    criterion_recon = nn.MSELoss()

    batch_size = 32
    n_samples = len(X_train)
    pretrain_epochs = 40

    model.train()
    for _ in range(pretrain_epochs):
        perm = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_size):
            b_idx = perm[i : i + batch_size]
            batch_x = X_train[b_idx]

            opt_recon.zero_grad()
            _, x_rec, _ = model(batch_x)
            loss_rec = criterion_recon(x_rec, batch_x)
            loss_rec.backward()
            opt_recon.step()

    # 5. Pelatihan Tahap 2: Supervised Fine-Tuning + Training Classifier Head
    #    Tujuan: Menyesuaikan representasi laten untuk memisahkan 8 kelas gestur
    opt_cls = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    criterion_cls = nn.CrossEntropyLoss()
    finetune_epochs = 40

    for _ in range(finetune_epochs):
        perm = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_size):
            b_idx = perm[i : i + batch_size]
            batch_x = X_train[b_idx]
            batch_y = y_train[b_idx]

            opt_cls.zero_grad()
            _, x_rec, logits = model(batch_x)
            loss_rec = criterion_recon(x_rec, batch_x)
            loss_cls = criterion_cls(logits, batch_y)
            # Joint loss: utamakan klasifikasi sambil tetap menjaga regularisasi rekonstruksi
            total_loss = loss_cls + 0.1 * loss_rec
            total_loss.backward()
            opt_cls.step()

    # 6. Evaluasi / Inferensi
    model.eval()
    with torch.no_grad():
        _, _, logits_val = model(X_val)
        pred_val = logits_val.argmax(dim=1).numpy()

        pred_test = None
        if final and X_test is not None:
            _, _, logits_te = model(X_test)
            pred_test = logits_te.argmax(dim=1).numpy()

    return pred_val, pred_test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--final", action="store_true", help="sprint terakhir: evaluasi data uji sekali")
    ap.add_argument("--perangkat", default="cpu laptop")
    a = ap.parse_args()

    t0 = time.time()
    pred_val, pred_test = latih_dan_prediksi(a.seed, a.final)
    durasi = time.time() - t0

    out = AKAR / "results" / NIM
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_val.npy", np.asarray(pred_val))

    perintah = [
        sys.executable,
        "-m",
        "src.common.evaluate",
        "--nim",
        NIM,
        "--model",
        JALUR,
        "--seed",
        str(a.seed),
        "--pred-val",
        str(out / "pred_val.npy"),
        "--durasi",
        f"{durasi:.1f}",
        "--perangkat",
        a.perangkat,
    ]
    if pred_test is not None:
        np.save(out / "pred_test.npy", np.asarray(pred_test))
        perintah += ["--pred-test", str(out / "pred_test.npy")]

    subprocess.run(perintah, check=True, cwd=AKAR, env=dict(os.environ, PYTHONIOENCODING="utf-8"))


if __name__ == "__main__":
    main()
