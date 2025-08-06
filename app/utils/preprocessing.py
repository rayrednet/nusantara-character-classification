import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd

# Ensure punkt tokenizer is downloaded
nltk.download('punkt')

def clean_text(text):
    text = text.replace('\t', ' ')
    text = text.replace('""', '"')
    text = text.replace('",', '')
    text = text.replace(',"', '')
    text = text.replace('","', '')
    text = text.replace('“', '"').replace('”', '"').replace('’', "'")
    text = re.sub(r'"\s*,\s*', ' ', text)
    text = re.sub(r'\s*,\s*"', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    return text

def preprocess_folktale(text, story_id="1", title="uploaded"):
    cleaned_text = clean_text(text)
    sentences = sent_tokenize(cleaned_text)

    rows = []
    for sent_id, sentence in enumerate(sentences):
        words = word_tokenize(sentence)
        for word in words:
            rows.append({
                'story_id': story_id,
                'judul': title,
                'sentence_id': sent_id,
                'sentence': sentence,
                'word': word
            })

    df = pd.DataFrame(rows)
    df = df[~df['word'].isin(['"', '`'])]  # Remove junk quote tokens
    return df
