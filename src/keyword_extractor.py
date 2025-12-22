# src/keyword_extractor.py

import spacy

nlp = spacy.load("en_core_web_sm")

def normalize_keywords(words):
    """
    Normalize keywords: lowercase, lemmatize, remove duplicates
    """
    normalized = set()
    for w in words:
        doc = nlp(w.lower())
        for token in doc:
            if token.is_stop or token.is_punct or token.is_space:
                continue
            normalized.add(token.lemma_)
    return list(normalized)

def extract_keywords_from_jd(job_description: str):
    """
    Extract important keywords from Job Description using spaCy
    """
    if not job_description:
        return []

    doc = nlp(job_description.lower())

    keywords = []
    for token in doc:
        if token.is_stop or token.is_punct or token.is_space:
            continue
        if len(token.text) < 2:
            continue
        if token.pos_ in {"NOUN", "PROPN", "ADJ", "VERB"}:
            keywords.append(token.lemma_)

    return normalize_keywords(keywords)
