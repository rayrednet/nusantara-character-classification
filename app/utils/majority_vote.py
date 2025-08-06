# utils/majority_vote.py

import os
import pandas as pd

def run_majority_vote(
    pred_df: pd.DataFrame,
    enriched_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Jalankan majority vote per Tokoh, menggunakan label Bahasa Indonesia:
      - "Protagonis"  (sebelumnya "protagonist")
      - "Antagonis"   (sebelumnya "antagonist")
      - "Lainnya"     (sebelumnya "others")
    """

    majority_out_full = "majority/final_predicted_majority_sorted_custom.csv"
    out_csv_min      = "majority/final_predicted_majority_minimal.csv"
    out_csv_full     = "majority/final_normalized_labeled.csv"

    df = pred_df.copy()

    # -- AGGREGATION --
    # Hitung berapa kali tiap Tokoh diprediksi sebagai Protagonis / Antagonis / Lainnya
    agg_fields = {
        "pro_cnt" : ("predicted_type", lambda s: (s == "Protagonis").sum()),
        "ant_cnt" : ("predicted_type", lambda s: (s == "Antagonis").sum()),
        "oth_cnt" : ("predicted_type", lambda s: (s == "Lainnya").sum()),
    }

    # Jika kolom confidence ada, gunakan; kalau tidak, pakai 0
    if "conf_Protagonis" in df.columns:
        agg_fields["pro_conf_total"] = ("conf_Protagonis", "sum")
    else:
        agg_fields["pro_conf_total"] = ("predicted_type", lambda s: 0)

    if "conf_Antagonis" in df.columns:
        agg_fields["ant_conf_total"] = ("conf_Antagonis", "sum")
    else:
        agg_fields["ant_conf_total"] = ("predicted_type", lambda s: 0)

    if "mention_count" in df.columns:
        agg_fields["mention_total"] = ("mention_count", "sum")
    else:
        # fallback: gabung dari enriched_df
        merged = df.merge(
            enriched_df[["story_id", "person", "sentence_id", "mention_count"]],
            on=["story_id", "person"], how="left"
        )
        df["mention_count"] = merged["mention_count"].fillna(0)
        agg_fields["mention_total"] = ("mention_count", "sum")

    agg = df.groupby(["story_id", "person"]).agg(**agg_fields).reset_index()

    # Fungsi majority: jika tidak ada pro/ant, label = "Lainnya"
    def majority(row):
        if row.pro_cnt == 0 and row.ant_cnt == 0:
            return "Lainnya"
        # cari maksimum di antara pro_cnt, ant_cnt, oth_cnt
        pilihan = {
            "Protagonis": row.pro_cnt,
            "Antagonis" : row.ant_cnt,
            "Lainnya"   : row.oth_cnt
        }
        return max(pilihan, key=lambda k: pilihan[k])

    agg["label"] = agg.apply(majority, axis=1)

    # LOGIKA FINAL
    REL_THRESH = 0.2
    THR_ANT   = 0.4

    def choose(sub, sort_cols, asc):
        return sub.sort_values(sort_cols, ascending=asc).index[0]

    for sid, idx in agg.groupby("story_id").groups.items():
        sub = agg.loc[idx]

        # Pastikan selalu ada satu Protagonis
        if (sub.label == "Protagonis").sum() == 0:
            pro_cand = sub[sub.pro_cnt > 0]
            cand = (choose(pro_cand, ["pro_cnt", "pro_conf_total", "mention_total"], asc=[False]*3)
                    if not pro_cand.empty
                    else choose(sub, ["mention_total"], asc=[False]))
            agg.loc[cand, "label"] = "Protagonis"

        sub = agg.loc[idx]
        # Pastikan selalu ada satu Antagonis
        if (sub.label == "Antagonis").sum() == 0:
            ant_cand = sub[sub.ant_cnt > 0]
            if not ant_cand.empty:
                cand = choose(ant_cand, ["ant_cnt", "ant_conf_total", "mention_total"], asc=[False]*3)
                agg.loc[cand, "label"] = "Antagonis"

        sub = agg.loc[idx]
        # Jika belum ada Antagonis, coba threshold confidence
        if (sub.label == "Antagonis").sum() == 0:
            others = sub[sub.label == "Lainnya"].copy()
            others["ant_conf_avg"] = others.ant_conf_total / others.oth_cnt.clip(lower=1)
            cand_conf = others[others.ant_conf_avg >= THR_ANT]
            if not cand_conf.empty:
                cand = choose(cand_conf, ["ant_conf_avg", "ant_conf_total", "mention_total"], asc=[False]*3)
                agg.loc[cand, "label"] = "Antagonis"

        sub = agg.loc[idx]
        # Pastikan sekali lagi ada minimal satu Protagonis
        if (sub.label == "Protagonis").sum() == 0:
            cand = choose(sub, ["mention_total"], asc=[False])
            agg.loc[cand, "label"] = "Protagonis"

        sub = agg.loc[idx]
        # Jika belum ada Antagonis, cari berdasarkan mention_total relatif
        if (sub.label == "Antagonis").sum() == 0:
            max_m = sub.mention_total.max()
            cand_sub = sub[(sub.label == "Lainnya") & (sub.mention_total >= REL_THRESH * max_m)]
            if not cand_sub.empty:
                cand = choose(cand_sub, ["pro_conf_total", "mention_total"], asc=[True, False])
                agg.loc[cand, "label"] = "Antagonis"

    # Tambahkan kolom bantu untuk sorting agar urutan Tokoh-1, Tokoh-2, dst.
    agg["person_num"] = agg["person"].str.extract(r"Tokoh-(\d+)", expand=False).astype(int)

    # Merge kembali kolom "aliases" (list alias) dari df‐awal
    original = df[["story_id", "person", "aliases"]].drop_duplicates()
    agg = agg.merge(original, on=["story_id", "person"], how="left")

    # Susun kolom final
    final_df = (
        agg.sort_values(["story_id", "person_num"])
           .rename(columns={"label": "predicted_type"})
           .drop(columns="person_num")
           .reset_index(drop=True)
    )

    # Letakkan kolom “story_id, person, aliases, predicted_type, … ”
    cols = ["story_id", "person", "aliases", "predicted_type"] + \
           [c for c in final_df.columns if c not in ["story_id", "person", "aliases", "predicted_type"]]
    final_df = final_df[cols]

    os.makedirs("majority", exist_ok=True)
    final_df.to_csv(majority_out_full, index=False)

    df_min = final_df[["story_id", "person", "predicted_type"]]
    df_min.to_csv(out_csv_min, index=False)

    # Merge final_df (predicted_type) kembali ke enriched_df
    df_merge = enriched_df.merge(df_min, on=["story_id", "person"], how="left")

    final_columns = [
        "story_id", "person", "aliases",
        "predicted_type",
        "sentence_id", "text", "mention_count", "word_count",
        "text_prev", "text_next", "bert_context",
        "is_primary_in_sentence"
    ]
    df_merge = df_merge[final_columns]
    df_merge.to_csv(out_csv_full, index=False)

    return final_df
