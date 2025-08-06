import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import re
from collections import defaultdict

def add_features_for_classification(cluster_df: pd.DataFrame,
                                    token_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the sentence-level feature table exactly the way it was done in training.

    Assumes `cluster_df` already has:
        • story_id
        • person
        • aliases   (list[str])
        • sentence_ids (list[int])   ← added in app.py
    """

    # ------------  clean token table  -----------------
    token_df = token_df.dropna(subset=["word"]).copy()
    token_df["word"] = token_df["word"].astype(str)

    # ------------  helper dicts (per-story) -----------
    story_tokens  = defaultdict(list)
    for r in token_df.itertuples(index=False):
        story_tokens[r.story_id].append(r.word.lower())

    story_strings = {sid: " ".join(tok) for sid, tok in story_tokens.items()}

    # ------------  mention_count (training logic) -----
    mention_dict = {}
    for (sid, person), sub in cluster_df.groupby(["story_id", "person"]):
        alias_set = {a.lower().strip()
                     for aliases in sub["aliases"] for a in aliases}

        toks = story_tokens[sid]
        txt = story_strings.get(sid, "")

        cnt = 0
        for al in alias_set:
            if len(al.split()) == 1:
                cnt += toks.count(al)
            else:
                cnt += len(re.findall(rf"\b{re.escape(al)}\b", txt))
        mention_dict[(sid, person)] = cnt

    # ------------  total word_count per Tokoh ---------
    # sentence length lookup
    sent_len = token_df.groupby("sentence_id")["word"].apply(len).to_dict()

    # sum sentence lengths for each Tokoh
    word_count_dict = {
        (row.story_id, row.person): sum(sent_len.get(s, 0) for s in row.sentence_ids)
        for row in cluster_df.itertuples()
    }

    # ------------  build sentence-level rows ----------
    all_aliases   = cluster_df["aliases"].explode().dropna().map(str.lower).map(str.strip).unique()
    alias_pat_map = {a: re.compile(rf"\b{re.escape(a)}\b") for a in all_aliases}

    rows = []
    for row in cluster_df.itertuples():
        sid, pid = row.story_id, row.person
        aliases  = [a.lower() for a in row.aliases]

        # only look inside sentences of THIS story
        story_sent_df = token_df[token_df["story_id"] == sid]

        for sent_id, sent_group in story_sent_df.groupby("sentence_id"):
            sent_txt = " ".join(sent_group.word).lower()
            if any(al in sent_txt for al in aliases):
                # -------- is_primary logic ----------
                first_alias, first_pos = None, None
                for al in aliases:
                    m = alias_pat_map[al].search(sent_txt)
                    if m and (first_pos is None or m.start() < first_pos):
                        first_alias, first_pos = al, m.start()
                is_primary = 1 if first_alias in aliases else 0

                # ---------- append row --------------
                rows.append({
                    "story_id"   : sid,
                    "person"     : pid,
                    "aliases"    : row.aliases,
                    "sentence_id": sent_id,
                    "text"       : sent_txt,
                    "mention_count": mention_dict[(sid, pid)],
                    "word_count"   : word_count_dict[(sid, pid)],
                    "is_primary_in_sentence": is_primary
                })

    df = pd.DataFrame(rows)

    # ------------  add context & scale ---------------
    if not df.empty:
        sent_text_lookup = (
            token_df.groupby("sentence_id")["word"]
                    .apply(lambda x: " ".join(x))
                    .to_dict()
        )
        df["text_prev"] = df["sentence_id"].apply(lambda x: sent_text_lookup.get(x - 1, ""))
        df["text_next"] = df["sentence_id"].apply(lambda x: sent_text_lookup.get(x + 1, ""))
        df["bert_context"] = (
            df["text_prev"] + " [SEP] " + df["text"] + " [SEP] " + df["text_next"]
        )

        scaler = MinMaxScaler()
        df[["mention_count", "word_count"]] = scaler.fit_transform(
            df[["mention_count", "word_count"]]
        )

    return df
