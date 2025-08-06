import pandas as pd

def build_sentence_level_dataset(merged_df: pd.DataFrame, preprocessed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert merged alias clusters into sentence-level format with full text for classification.
    
    Parameters:
        merged_df (pd.DataFrame): DataFrame containing merged alias clusters with 'person', 'aliases', and 'sentence_ids'.
        preprocessed_df (pd.DataFrame): Preprocessed token-level DataFrame with 'sentence_id' and 'word'.

    Returns:
        pd.DataFrame: Sentence-level dataset containing story_id, person, aliases, sentence_id, and full text.
    """
    if "sentence_id" not in preprocessed_df.columns or "word" not in preprocessed_df.columns:
        raise ValueError("preprocessed_df must contain 'sentence_id' and 'word' columns")

    # Rebuild sentence lookup table
    preprocessed_df["word"] = preprocessed_df["word"].astype(str)
    sentence_texts = (
        preprocessed_df.groupby("sentence_id")["word"]
        .agg(" ".join)
        .reset_index()
        .rename(columns={"word": "text"})
    )

    sent_lookup = sentence_texts.set_index("sentence_id")["text"].to_dict()

    # Expand merged_df by sentence_id
    rows = []
    for r in merged_df.itertuples():
        sids = getattr(r, "sentence_ids", [])
        if isinstance(sids, str):
            try:
                sids = eval(sids)
            except:
                sids = []
        for sid in sids:
            rows.append({
                "story_id"   : getattr(r, "story_id", 1), 
                "person"     : r.person,
                "aliases": getattr(r, "aliases", []),
                "sentence_id": sid,
                "text"       : sent_lookup.get(sid, "")
            })

    return pd.DataFrame(rows)
