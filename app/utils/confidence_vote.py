# utils/confidence_vote.py

import pandas as pd
import numpy as np
from collections import defaultdict

# Ganti LABELS ke Bahasa Indonesia
LABELS = ["Lainnya", "Protagonis", "Antagonis"]

def confidence_weighted_vote(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Lakukan confidence-weighted voting per karakter (Tokoh) menggunakan probabilitas softmax per kalimat.

    Parameters:
        pred_df: DataFrame dengan satu baris per kalimat yang memuat:
            - story_id, person
            - conf_Lainnya, conf_Protagonis, conf_Antagonis  (nama kolom confidence baru)
            - aliases  (jika ada)
    Returns:
        DataFrame satu baris per Tokoh (story_id, person) dengan kolom:
            - predicted_type  (Lainnya / Protagonis / Antagonis)
            - confidence_Lainnya, confidence_Protagonis, confidence_Antagonis
            - aliases  (bila tersedia di pred_df)
    """

    # Mapping label ke indeks
    label_to_index = {label: i for i, label in enumerate(LABELS)}
    confidence_dict = defaultdict(lambda: np.zeros(len(LABELS)))

    # Jika kolom person berisi list, ubah dulu ke string agar bisa dipakai sebagai key
    if pred_df["person"].apply(lambda x: isinstance(x, list)).any():
        pred_df["person"] = pred_df["person"].apply(
            lambda x: ", ".join(map(str, x)) if isinstance(x, list) else str(x)
        )

    # Pastikan kolom confidence sudah ada; jika belum, default 0
    for row in pred_df.itertuples():
        key = (row.story_id, row.person)
        conf = np.array([
            getattr(row, "conf_Lainnya", 0),
            getattr(row, "conf_Protagonis", 0),
            getattr(row, "conf_Antagonis", 0),
        ])
        confidence_dict[key] += conf

    results = []
    for (story_id, person), conf_vec in confidence_dict.items():
        label_idx = np.argmax(conf_vec)
        results.append({
            "story_id": story_id,
            "person": person,
            "predicted_type": LABELS[label_idx],
            "confidence_Lainnya": conf_vec[0],
            "confidence_Protagonis": conf_vec[1],
            "confidence_Antagonis": conf_vec[2],
        })

    final_df = pd.DataFrame(results)

    # Jika kolom "aliases" ada di pred_df, merge ke final_df
    if "aliases" in pred_df.columns:
        alias_map = pred_df[["story_id", "person", "aliases"]].copy()
        alias_map["person"] = alias_map["person"].astype(str)
        alias_map["aliases"] = alias_map["aliases"].apply(
            lambda x: ", ".join(map(str, x)) if isinstance(x, list) else str(x)
        )

        final_df["person"] = final_df["person"].astype(str)
        final_df = final_df.merge(alias_map.drop_duplicates(), on=["story_id", "person"], how="left")

    # Urutkan kolom sesuai format yang diinginkan
    column_order = [
        "story_id", "person", "aliases",
        "predicted_type",
        "confidence_Lainnya", "confidence_Protagonis", "confidence_Antagonis"
    ]
    for col in column_order:
        if col not in final_df.columns:
            final_df[col] = None

    final_df = final_df[column_order]
    return final_df
