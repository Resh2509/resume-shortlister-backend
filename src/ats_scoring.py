# src/ats_scoring.py

from src.text_cleaner import tokenize

def compute_ats_score(resume_text: str, jd_keywords):
    """
    Compute ATS score based on keyword overlap
    """

    if not resume_text or not jd_keywords:
        return {
            "score": 0.0,
            "matched": [],
            "missing": []
        }

    # Convert BOTH to sets
    resume_tokens = set(tokenize(resume_text))
    jd_set = set(jd_keywords)

    matched = sorted(list(jd_set & resume_tokens))
    missing = sorted(list(jd_set - resume_tokens))

    score = round((len(matched) / len(jd_set)) * 100, 2) if jd_set else 0.0

    return {
        "score": score,
        "matched": matched,
        "missing": missing
    }
