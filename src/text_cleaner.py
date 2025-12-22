# src/text_cleaner.py  (spaCy version – NLTK removed)

import re
import spacy

# Load spaCy model once
nlp = spacy.load("en_core_web_sm")

def clean_text(text: str) -> str:
    text = text or ""

    # remove emails, urls, phone numbers, normalize spaces
    text = text.replace('\r', ' ').replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'http[s]?://\S+', ' ', text)
    text = re.sub(r'\+?\d[\d\s\-]{6,}', ' ', text)
    text = re.sub(r'[^0-9a-zA-Z#\+\- ]', ' ', text)
    text = text.lower()

    doc = nlp(text)

    cleaned_tokens = []
    for token in doc:
        if token.is_stop:
            continue
        if token.is_punct or token.is_space:
            continue
        if len(token.text) <= 1:
            continue
        cleaned_tokens.append(token.lemma_)   # spaCy lemmatization

    return " ".join(cleaned_tokens)

def tokenize(text: str):
    return clean_text(text).split()
