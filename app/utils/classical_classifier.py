import os
import pandas as pd
from scipy.sparse import hstack
import joblib

# === LOAD MODEL ARTEFACTS ===
ROOT_DIR   = "models/random_forest_normalized"
BEST_MODEL = os.path.join(ROOT_DIR, "best_model.pkl")
TFIDF_VEC  = os.path.join(ROOT_DIR, "tfidf.pkl")
LBL_ENCOD  = os.path.join(ROOT_DIR, "label_encoder.pkl")

rf    = joblib.load(BEST_MODEL)
tfidf = joblib.load(TFIDF_VEC)
le    = joblib.load(LBL_ENCOD)

# Map ke label Bahasa Indonesia
label_map = {
    "others":       "Lainnya",
    "protagonist":  "Protagonis",
    "antagonist":   "Antagonis"
}

# === CLASSIFIER FUNCTION ===
def classify_characters(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Fill missing text column (for TF-IDF)
    df["text"] = df["text"].fillna("")

    # Preprocess numeric features
    num_cols = ["mention_count", "word_count", "is_primary_in_sentence"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)

    # Gabungkan fitur numerik dan teks
    X_num = df[num_cols].values
    X_text = tfidf.transform(df["text"])
    X = hstack([X_num, X_text])

    # Prediksi
    y_pred = rf.predict(X)
    probs  = rf.predict_proba(X)

    # Inverse transform ke label asli, lalu map ke Bahasa Indonesia
    df["predicted_type"] = le.inverse_transform(y_pred)
    df["predicted_type"] = df["predicted_type"].map(label_map)

    # Buat kolom confidence sesuai label baru
    for idx, cls in enumerate(le.classes_):
        indo_cls = label_map[cls]  # misalnya "others" → "Lainnya"
        df[f"conf_{indo_cls}"] = probs[:, idx]

    # Convert alias lists ke string untuk deduplikasi
    df["aliases"] = df["aliases"].apply(
        lambda x: ", ".join(x) if isinstance(x, list) else str(x)
    )

    # Kembalikan seluruh DataFrame (bukan cuma ringkasannya)
    return df
