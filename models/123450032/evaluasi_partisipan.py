"""Metrik per partisipan untuk jalur Transformer dan pembanding non-DL.

    python models/123450032/evaluasi_partisipan.py --seed 1

Membaca prediksi validasi hasil ``run.py`` lalu menghitung akurasi dan macro-F1 per partisipan
pada split beku yang sama. Pembanding non-DL memakai fitur sudut sendi + Random Forest dari
``scripts/baseline_rf.py`` dengan seed 2026.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))
from src.common import evaluate, manifest, split  # noqa: E402
from src.protokol import data  # noqa: E402

NIM = Path(__file__).resolve().parent.name


def muat_baseline_rf():
    path = AKAR / "scripts" / "baseline_rf.py"
    spec = importlib.util.spec_from_file_location("baseline_rf", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def metrik_per_partisipan(y: np.ndarray, p: np.ndarray, partisipan: np.ndarray) -> dict:
    hasil = {}
    for nama in sorted(set(partisipan.tolist())):
        mask = partisipan == nama
        m = evaluate.metrik_klasifikasi(y[mask], p[mask])
        hasil[str(nama)] = {"n": int(mask.sum()), "acc": round(m["acc"], 6),
                            "macro_f1": round(m["macro_f1"], 6)}
    return hasil


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    X_full = np.load(manifest.AKAR / "data" / "aktual" / "gestur_primer.npz")["X"]
    _, _, _, y_val, _, _ = data.muat()
    part_full = data.muat_partisipan()
    _, va = split.indeks(len(X_full), porsi_val=0.1, seed=split.SEED_SPLIT)
    part_val = part_full[va]

    pred = np.load(AKAR / "results" / NIM / "pred_val.npy")
    overall = evaluate.metrik_klasifikasi(np.asarray(y_val), pred)

    mod = muat_baseline_rf()
    Xtr, ytr, Xva, yva, _, _ = data.muat()
    from sklearn.ensemble import RandomForestClassifier

    Ftr = mod.ekstraksi_fitur_tangan(np.asarray(Xtr))
    Fva = mod.ekstraksi_fitur_tangan(np.asarray(Xva))
    rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=split.SEED_SPLIT, n_jobs=-1)
    rf.fit(Ftr, ytr)
    p_rf = rf.predict(Fva)
    baseline_overall = evaluate.metrik_klasifikasi(np.asarray(yva), p_rf)

    hasil = {
        "model": "trf",
        "seed": int(a.seed),
        "data_hash": manifest.data_hash(f"split:{split.SEED_SPLIT}".encode()),
        "n_val": int(len(y_val)),
        "overall": {"val_acc": round(overall["acc"], 6), "val_macro_f1": round(overall["macro_f1"], 6)},
        "per_partisipan": metrik_per_partisipan(np.asarray(y_val), pred, part_val),
        "partisipan_tersedia": sorted(set(part_full.tolist())),
        "partisipan_tanpa_data": sorted({f"P{i:02d}" for i in range(1, 9)} - set(part_full.tolist())),
        "pembanding_non_dl": {
            "model": "random_forest_joint_angles",
            "n_estimators": 100,
            "n_features": int(Ftr.shape[1]),
            "overall": {"val_acc": round(baseline_overall["acc"], 6),
                        "val_macro_f1": round(baseline_overall["macro_f1"], 6)},
            "per_partisipan": metrik_per_partisipan(np.asarray(yva), p_rf, part_val),
        },
    }
    out = AKAR / "results" / NIM / "metrics_per_partisipan.json"
    out.write_text(json.dumps(hasil, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"overall": hasil["overall"], "per_partisipan": hasil["per_partisipan"],
                      "pembanding_non_dl": hasil["pembanding_non_dl"]["overall"]}, indent=2))
    print("->", out.relative_to(AKAR))


if __name__ == "__main__":
    main()
