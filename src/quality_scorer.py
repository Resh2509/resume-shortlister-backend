# src/quality_scorer.py

import os
import re

def quality_score(resume_text: str, pdf_path: str = None):
    """
    Simple resume quality scoring based on:
    - length
    - section presence
    - formatting heuristics
    """
    score = 0
    details = {}

    text = resume_text or ""

    # Length score
    length = len(text.split())
    if length > 400:
        score += 30
    elif length > 250:
        score += 20
    else:
        score += 10
    details["length_words"] = length

    # Section checks
    sections = ["experience", "education", "skills", "project"]
    found_sections = 0
    for sec in sections:
        if re.search(sec, text.lower()):
            found_sections += 1

    section_score = found_sections * 10
    score += section_score
    details["sections_found"] = found_sections

    # File size heuristic (optional)
    if pdf_path and os.path.exists(pdf_path):
        size_kb = os.path.getsize(pdf_path) / 1024
        if size_kb > 50:
            score += 10
        details["file_size_kb"] = round(size_kb, 2)

    return {
        "score": min(score, 100),
        "details": details
    }
