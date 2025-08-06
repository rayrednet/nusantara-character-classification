# utils/bert_classifier.py

import os
import torch
import pandas as pd
from collections import Counter, defaultdict
from transformers import BertTokenizer, BertModel
from safetensors.torch import load_file
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

LABELS = ["Lainnya", "Protagonis", "Antagonis"]
FOLDS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === MODEL ===
class IndoBERTWithNumeric(nn.Module):
    def __init__(self, model_name="cahya/bert-base-indonesian-1.5G", num_labels=3, num_numeric_features=3):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size + num_numeric_features, num_labels)

    def forward(self, input_ids, attention_mask, numeric_feats):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.pooler_output
        combined = torch.cat((pooled_output, numeric_feats), dim=1)
        logits = self.classifier(self.dropout(combined))
        return logits

# === DATASET ===
class SentenceDataset(Dataset):
    def __init__(self, df, tokenizer, numeric_cols):
        self.encodings = tokenizer(df["bert_context"].tolist(), truncation=True, padding=True, max_length=128)
        self.numeric_feats = torch.tensor(df[numeric_cols].values, dtype=torch.float)
        self.meta = df[["story_id", "person", "sentence_id"]].reset_index(drop=True)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["numeric_feats"] = self.numeric_feats[idx]
        item["meta"] = self.meta.loc[idx].to_dict()
        return item

def custom_collate_fn(batch):
    keys = batch[0].keys()
    collated = {k: [d[k] for d in batch] for k in keys}
    for k in ["input_ids", "attention_mask", "numeric_feats"]:
        collated[k] = torch.stack(collated[k])
    return collated

# === MAIN FUNCTION ===
def classify_characters(enriched_df: pd.DataFrame) -> pd.DataFrame:
    tokenizer = BertTokenizer.from_pretrained("cahya/bert-base-indonesian-1.5G")
    numeric_cols = ["mention_count", "word_count", "is_primary_in_sentence"]
    dataset = SentenceDataset(enriched_df, tokenizer, numeric_cols)
    dataloader = DataLoader(dataset, batch_size=16, collate_fn=custom_collate_fn)

    predictions_per_sentence = [[] for _ in range(len(enriched_df))]

    for fold_idx in range(FOLDS):
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", "V4_CahyaBERT", f"best_fold_{fold_idx + 1}", "model.safetensors")
        model = IndoBERTWithNumeric()
        model.load_state_dict(load_file(model_path), strict=False)
        model.to(DEVICE)
        model.eval()

        probs_per_sentence = [[] for _ in range(len(enriched_df))]

        with torch.no_grad():
            idx = 0
            for batch in dataloader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                numeric_feats = batch["numeric_feats"].to(DEVICE)

                logits = model(input_ids, attention_mask, numeric_feats)
                probs = F.softmax(logits, dim=1).cpu()
                preds = torch.argmax(probs, dim=1).tolist()

                for p, prob in zip(preds, probs):
                    predictions_per_sentence[idx].append(p)
                    probs_per_sentence[idx].append(prob.numpy())  # Save actual softmax probs
                    idx += 1


    final_preds = [Counter(votes).most_common(1)[0][0] for votes in predictions_per_sentence]
    enriched_df["predicted_type"] = [LABELS[i] for i in final_preds]

    # Compute confidence scores
    probs_avg = np.array([np.mean(probs, axis=0) for probs in probs_per_sentence])
    enriched_df["conf_Lainnya"] = probs_avg[:, 0]
    enriched_df["conf_Protagonis"] = probs_avg[:, 1]
    enriched_df["conf_Antagonis"] = probs_avg[:, 2]

    return enriched_df

