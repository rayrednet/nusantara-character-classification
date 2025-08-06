import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline
import streamlit as st
import os
from utils.alias_normalizer import normalize_alias_custom

@st.cache_resource
def load_ner_model():
    base = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base, "models", "ner_model")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    return tokenizer, model

def extract_characters(token_lists):
    tokenizer, model = load_ner_model()
    model.eval()

    ner_pipeline = TokenClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=0 if torch.cuda.is_available() else -1
    )

    raw_results = []

    for i, tokens in enumerate(token_lists):
        sentence = " ".join(tokens)
        try:
            preds = ner_pipeline(sentence)
        except Exception as e:
            print(f"NER failed on sentence {i}: {e}")
            continue

        for pred in preds:
            if pred["entity_group"] == "PER":
                raw_char = pred["word"].replace("##", "").strip()
                norm_char = normalize_alias_custom(raw_char)
                raw_results.append({
                    "sentence_id": i,
                    "Character": raw_char,
                    "Normalized": norm_char,
                    "Confidence": round(pred["score"], 3),
                    "Tokens": sentence
                })

    # === Optional: flatten to one row per unique character mention ===
    df = pd.DataFrame(raw_results)
    if df.empty:
        return df

    # One row per sentence-level detection (original structure)
    return df
