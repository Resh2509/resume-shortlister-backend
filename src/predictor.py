# src/predictor.py
import joblib
import os
import numpy as np
import fitz

from sentence_transformers import SentenceTransformer

from src.pdf_extractor import extract_text_from_pdf
from src.text_cleaner import clean_text
from src.keyword_extractor import extract_keywords_from_jd
from src.ats_scoring import compute_ats_score

from src.experience_extractor import estimate_experience_years
from src.skill_ranker import rank_skills
from src.project_relevance import project_relevance_score
from src.quality_scorer import quality_score
from src.semantic_matcher import embed, cosine_sim


EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")


def pdf_has_photo(pdf_path: str) -> bool:
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            if page.get_images(full=True):
                return True
    except Exception:
        return False
    return False


class ResumePredictor:
    def __init__(self, model_path="models/svm_bert.pkl", scaler_path="models/scaler.pkl"):
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model not found. Train first.")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError("Scaler not found. Train first.")
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict_from_pdf(self, pdf_path: str, job_description: str = ""):
        # Photo check
        if pdf_has_photo(pdf_path):
            return {
                "pred_label": 0,
                "pred_proba": None,
                "reason": "Rejected: Resume contains a photo",
                "ats": None,
                "extracted_text_sample": None,
                # Advanced placeholders
                "experience_years": 0.0,
                "skill_ranking": [],
                "project_relevance": {"score": 0.0, "projects_found": 0, "details": []},
                "quality": {"score": 0.0}
            }

        raw = extract_text_from_pdf(pdf_path)
        cleaned = clean_text(raw)

        # ML prediction (BERT+SVM)
        emb = embed(cleaned)[0].reshape(1, -1)
        emb_scaled = self.scaler.transform(emb)
        pred_label = int(self.model.predict(emb_scaled)[0])
        try:
            pred_proba = float(self.model.predict_proba(emb_scaled)[0][1])
        except Exception:
            pred_proba = None

        # ATS
        jd_keywords = extract_keywords_from_jd(job_description)
        ats = compute_ats_score(cleaned, jd_keywords) if job_description else {'score': None, 'matched': [], 'missing': []}

        # Advanced features
        experience_years = estimate_experience_years(raw)
        skill_ranking = rank_skills(cleaned, jd_keywords, top_k=20)
        project_relevance = project_relevance_score(raw, job_description)
        quality = quality_score(raw, pdf_path=pdf_path)

        # Role match: compare job_role phrase (if provided in job_description) to resume title lines
        # Simple heuristic: take first 10 lines of resume and compute semantic similarity
        resume_header = "\n".join((raw or "").splitlines()[:10])
        role_sim = 0.0
        if job_description:
            try:
                role_sim = cosine_sim(embed(resume_header)[0], embed(job_description)[0])
            except Exception:
                role_sim = 0.0

        return {
            "pred_label": pred_label,
            "pred_proba": pred_proba,
            "ats": ats,
            "extracted_text_sample": cleaned[:1000],
            # new fields
            "experience_years": experience_years,
            "skill_ranking": skill_ranking,
            "project_relevance": project_relevance,
            "quality": quality,
            "role_match_score": round(float(role_sim), 3)
        }
