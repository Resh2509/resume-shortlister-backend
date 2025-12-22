# src/skill_ranker.py

from collections import Counter
import spacy

nlp = spacy.load("en_core_web_sm")

def rank_skills(resume_text: str, jd_keywords=None, top_k=20):
    """
    Rank skills based on frequency in resume and relevance to JD keywords.
    Returns top_k skills with scores.
    """
    if not resume_text:
        return []

    jd_keywords = set(jd_keywords or [])

    doc = nlp(resume_text.lower())

    skills = []
    for token in doc:
        if token.is_stop or token.is_punct or token.is_space:
            continue
        if len(token.text) < 2:
            continue
        if token.pos_ in {"NOUN", "PROPN"}:
            skills.append(token.lemma_)

    freq = Counter(skills)

    ranked = []
    for skill, count in freq.most_common():
        score = count
        if skill in jd_keywords:
            score += 2   # boost if JD-relevant
        ranked.append({
            "skill": skill,
            "score": score
        })

    return ranked[:top_k]
