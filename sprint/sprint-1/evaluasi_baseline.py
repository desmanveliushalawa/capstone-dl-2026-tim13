"""Evaluasi baseline pada sesi simulasi: F1-makro per peserta.

Model:
  * Random Forest (pembanding non-DL, 300 pohon, class_weight balanced)
  * MLP PyTorch kecil (2 lapisan tersembunyi, dropout, early stopping di validasi)

Protokol:
  * GroupKFold per peserta pada P01-P04; P05 disisihkan sebagai data uji akhir
    (belum dievaluasi pada Sprint 1).
  * Normalisasi sudah dihitung dari data latih tiap lipat oleh src.protokol.sesi.
  * Metrik utama: F1-makro per peserta, dilaporkan rerata +/- simpangan baku.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(AKAR))

from src.protokol import metrik, sesi  # noqa: E402

HASIL = HERE / "hasil"
SEEDS = [1, 2, 3]
LABEL = list(range(8))
PESERTA_UJI_AKHIR = None  # diisi dari sesi.json


def _ratakan(X: np.ndarray) -> np.ndarray:
    return np.asarray(X, np.float32).reshape(len(X), -1)


def fitur_dist(X: np.ndarray) -> np.ndarray:
    """Jarak berpasangan 21 titik (210 fitur) dari pose yang sudah dinormalisasi."""
    X = np.asarray(X, np.float32)
    D = np.linalg.norm(X[:, :, None, :2] - X[:, None, :, :2], axis=-1)
    iu = np.triu_indices(21, 1)
    return D[:, iu[0], iu[1]].astype(np.float32)


def latih_rf(Xtr, ytr, seed: int):
    from sklearn.ensemble import RandomForestClassifier

    m = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    m.fit(Xtr, ytr)
    return m


def latih_mlp(Xtr, ytr, Xva, yva, seed: int):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    model = nn.Sequential(
        nn.Linear(Xtr.shape[1], 64), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(32, 8),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    Xv = torch.tensor(Xva, dtype=torch.float32)
    yv = np.asarray(yva, np.int64)

    terbaik, sabar, state = -1.0, 0, None
    for _ in range(300):
        model.train()
        urut = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 32):
            idx = urut[i:i + 32]
            opt.zero_grad()
            loss = lossf(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xv).argmax(1).numpy()
        f1 = metrik.f1_makro(yv, pv, LABEL)
        if f1 > terbaik + 1e-6:
            terbaik, sabar = f1, 0
            state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            sabar += 1
            if sabar >= 40:
                break
    if state is not None:
        model.load_state_dict(state)
    model.eval()
    return model


def prediksi(m, X):
    import torch

    if hasattr(m, "predict"):
        return m.predict(X).astype(np.int64)
    with torch.no_grad():
        return m(torch.tensor(X, dtype=torch.float32)).argmax(1).numpy().astype(np.int64)


def jalankan():
    HASIL.mkdir(parents=True, exist_ok=True)
    data, meta = sesi.muat_mentah()
    semua = sesi.daftar_peserta(data)
    uji_akhir = (meta.get("split_beku") or {}).get("uji", [semua[-1]])[0]
    cv = [p for p in semua if p != uji_akhir]
    if len(cv) < 3:
        raise SystemExit(f"Perlu >=3 peserta untuk GroupKFold; tersedia {cv}")
    print(f"Peserta CV: {cv}; data uji akhir disisihkan: {uji_akhir}")

    baris: list[dict] = []
    lipat_info: list[dict] = []
    prediksi_uji: dict[str, dict[str, np.ndarray]] = {"rf": {}, "mlp": {}}

    for i, p_uji in enumerate(cv):
        p_val = cv[(i + 1) % len(cv)]
        p_latih = [p for p in cv if p not in (p_uji, p_val)]
        d = sesi.muat_detail(latih=p_latih, validasi=[p_val], uji=[p_uji])
        # RF: jarak berpasangan dari pose ternormalisasi (invarian skala/posisi).
        Rtr, Rte = fitur_dist(d["X_pose_latih"]), fitur_dist(d["X_pose_uji"])
        # MLP: 63 koordinat pose ternormalisasi lalu distandardisasi statistik latih.
        Mtr, Mva, Mte = _ratakan(d["X_latih"]), _ratakan(d["X_validasi"]), _ratakan(d["X_uji"])
        ytr, yva, yte = d["y_latih"], d["y_validasi"], d["y_uji"]
        info = {
            "peserta_uji": p_uji, "peserta_validasi": p_val, "peserta_latih": p_latih,
            "n_latih": int(len(ytr)), "n_validasi": int(len(yva)), "n_uji": int(len(yte)),
        }
        lipat_info.append(info)
        print(f"\nLipat uji {p_uji}: latih {p_latih} ({len(ytr)}), val {p_val} ({len(yva)}), uji {len(yte)}")

        for seed in SEEDS:
            rf = latih_rf(Rtr, ytr, seed)
            prf = prediksi(rf, Rte)
            mrf = metrik.per_peserta(yte, prf, d["peserta_uji"], LABEL)[0]
            baris.append({"model": "rf", "seed": seed, **info, **mrf})
            prediksi_uji["rf"].setdefault(p_uji, prf)

            mlp = latih_mlp(Mtr, ytr, Mva, yva, seed)
            pmlp = prediksi(mlp, Mte)
            mmlp = metrik.per_peserta(yte, pmlp, d["peserta_uji"], LABEL)[0]
            baris.append({"model": "mlp", "seed": seed, **info, **mmlp})
            prediksi_uji["mlp"].setdefault(p_uji, pmlp)
            print(f"  seed {seed}: RF F1={mrf['f1_makro']:.3f} acc={mrf['akurasi']:.3f} | "
                  f"MLP F1={mmlp['f1_makro']:.3f} acc={mmlp['akurasi']:.3f}")

    # Ringkasan per peserta (rerata 3 seed), lalu rerata antar-peserta.
    per_peserta: list[dict] = []
    for model in ("rf", "mlp"):
        for p in cv:
            sub = [b for b in baris if b["model"] == model and b["peserta_uji"] == p]
            per_peserta.append({
                "model": model, "peserta": p,
                "n": int(sub[0]["n_uji"]),
                "f1_rerata": float(np.mean([b["f1_makro"] for b in sub])),
                "f1_sb": float(np.std([b["f1_makro"] for b in sub], ddof=1)) if len(sub) > 1 else 0.0,
                "akurasi_rerata": float(np.mean([b["akurasi"] for b in sub])),
            })

    ringkasan = {}
    for model in ("rf", "mlp"):
        sub = [x for x in per_peserta if x["model"] == model]
        f1 = np.array([x["f1_rerata"] for x in sub])
        acc = np.array([x["akurasi_rerata"] for x in sub])
        ringkasan[model] = {
            "f1_makro_rerata": float(f1.mean()),
            "f1_makro_simpangan_baku": float(f1.std(ddof=1)),
            "akurasi_rerata": float(acc.mean()),
            "akurasi_simpangan_baku": float(acc.std(ddof=1)),
            "n_peserta": len(sub),
        }

    # Evaluasi split beku pada validasi saja; uji akhir TIDAK disentuh.
    d_fix = sesi.muat_detail()
    Rtr, Rva = fitur_dist(d_fix["X_pose_latih"]), fitur_dist(d_fix["X_pose_validasi"])
    Mtr, Mva = _ratakan(d_fix["X_latih"]), _ratakan(d_fix["X_validasi"])
    ytr, yva = d_fix["y_latih"], d_fix["y_validasi"]
    val_beku = {}
    for model in ("rf", "mlp"):
        f1s = []
        for seed in SEEDS:
            m = latih_rf(Rtr, ytr, seed) if model == "rf" else latih_mlp(Mtr, ytr, Mva, yva, seed)
            p = prediksi(m, Rva if model == "rf" else Mva)
            f1s.append(metrik.f1_makro(yva, p, LABEL))
        val_beku[model] = {"f1_rerata": float(np.mean(f1s)), "f1_sb": float(np.std(f1s, ddof=1))}

    # Konfusi agregat (prediksi seed pertama tiap lipat).
    konfusi = {}
    for model in ("rf", "mlp"):
        total = np.zeros((8, 8), np.int64)
        for p in cv:
            y_p = data["y"][data["peserta"] == p]
            total += metrik.matriks_konfusi(y_p, prediksi_uji[model][p], LABEL)
        konfusi[model] = total.tolist()

    hasil = {
        "metrik_utama": "F1-makro per peserta",
        "protokol": {
            "group_kfold": "satu peserta = satu lipat uji; validasi = peserta berikutnya",
            "peserta_cv": cv,
            "peserta_uji_akhir": uji_akhir,
            "catatan_uji_akhir": "data uji akhir disisihkan dan belum dievaluasi pada Sprint 1",
            "seed": SEEDS,
            "normalisasi": "mean/std data latih tiap lipat",
            "data_hash": sesi.hash_dataset(),
            "n_sampel": int(len(data["y"])),
        },
        "lipat": lipat_info,
        "per_peserta": per_peserta,
        "ringkasan": ringkasan,
        "validasi_split_beku": val_beku,
        "split_beku": meta.get("split_beku"),
        "konfusi": konfusi,
        "label": LABEL,
    }
    (HASIL / "metrics.json").write_text(json.dumps(hasil, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    with (HASIL / "per_peserta.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "peserta", "n", "f1_rerata", "f1_sb", "akurasi_rerata"])
        w.writeheader()
        w.writerows(per_peserta)

    tulis_laporan(hasil)
    print("\nRingkasan F1-makro per peserta:")
    for model, r in ringkasan.items():
        print(f"  {model.upper()}: {r['f1_makro_rerata']:.3f} +/- {r['f1_makro_simpangan_baku']:.3f} "
              f"(acc {r['akurasi_rerata']:.3f})")
    print("hasil ->", HASIL)


def tulis_laporan(hasil: dict) -> None:
    baris_per = hasil["per_peserta"]
    r = hasil["ringkasan"]
    v = hasil["validasi_split_beku"]
    teks = [
        "# Hasil evaluasi baseline — sesi simulasi Sprint 1",
        "",
        "Metrik utama: **F1-makro per peserta**. Split GroupKFold per peserta pada "
        f"{', '.join(hasil['protokol']['peserta_cv'])}; data uji akhir "
        f"`{hasil['protokol']['peserta_uji_akhir']}` disisihkan dan belum dievaluasi.",
        "",
        "## Ringkasan (rerata antar-peserta, 3 seed)",
        "",
        "| Model | F1-makro | Simpangan baku | Akurasi |",
        "|---|---|---|---|",
    ]
    for model in ("rf", "mlp"):
        teks.append(f"| {model.upper()} | {r[model]['f1_makro_rerata']:.3f} | "
                    f"{r[model]['f1_makro_simpangan_baku']:.3f} | {r[model]['akurasi_rerata']:.3f} |")
    teks += ["", "## F1-makro per peserta", "",
             "| Peserta | n uji | RF F1 | RF akurasi | MLP F1 | MLP akurasi |",
             "|---|---|---|---|---|---|"]
    for p in hasil["protokol"]["peserta_cv"]:
        rf = next(x for x in baris_per if x["model"] == "rf" and x["peserta"] == p)
        ml = next(x for x in baris_per if x["model"] == "mlp" and x["peserta"] == p)
        teks.append(f"| {p} | {rf['n']} | {rf['f1_rerata']:.3f} ± {rf['f1_sb']:.3f} | "
                    f"{rf['akurasi_rerata']:.3f} | {ml['f1_rerata']:.3f} ± {ml['f1_sb']:.3f} | "
                    f"{ml['akurasi_rerata']:.3f} |")
    teks += ["", "## Validasi split beku (P04), uji akhir P05 disisihkan", "",
             "| Model | F1-makro validasi | Simpangan baku |", "|---|---|---|"]
    for model in ("rf", "mlp"):
        teks.append(f"| {model.upper()} | {v[model]['f1_rerata']:.3f} | {v[model]['f1_sb']:.3f} |")
    teks += [
        "",
        "## Catatan keterbatasan",
        "",
        "- Rekaman yang tersedia berjarak **1 detik/frame** (selangMs=1000), bukan 0,1 detik "
        "seperti rencana. Setelah membuang 1 detik pertama, tiap kondisi hanya menyisakan ~1–2 frame.",
        "- Hanya 5 blok peserta terdeteksi dari 8 yang direncanakan; 3 blok terakhir tidak ada di rekaman.",
        "- Label gestur mengikuti urutan rencana karena rekaman tidak menyertakan label.csv sesi; "
        "kesalahan penyelarasan slot mungkin terjadi.",
        "- Angka di atas adalah verifikasi jalur evaluasi, bukan hasil final. Data uji akhir P05 "
        "baru boleh dievaluasi sekali pada sprint terakhir.",
        "",
    ]
    (HASIL / "laporan.md").write_text("\n".join(teks), encoding="utf-8")


if __name__ == "__main__":
    jalankan()
