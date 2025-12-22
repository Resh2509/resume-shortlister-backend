# src/project_relevance.py

from sentence_transformers import SentenceTransformer
import re

_model = SentenceTransformer("all-MiniLM-L6-v2")

def project_relevance_score(resume_text: str, job_description: str):
    """
    Scores how relevant resume projects are to the job description
    using semantic similarity (BERT).
    """
    if not resume_text or not job_description:
        return {
            "score": 0.0,
            "projects_found": 0,
            "details": []
        }

    # Extract project-like sections
    lines = resume_text.splitlines()
    projects = []
    current = []

    for line in lines:
        if re.search(r"(project|projects)", line.lower()):
            if current:
                projects.append(" ".join(current))
                current = []
        current.append(line)

    if current:
        projects.append(" ".join(current))

    if not projects:
        return {
            "score": 0.0,
            "projects_found": 0,
            "details": []
        }

    jd_emb = _model.encode(job_description)
    proj_embs = _model.encode(projects)

    scores = []
    for proj, emb in zip(projects, proj_embs):
        sim = float((emb @ jd_emb) / ( (emb @ emb) ** 0.5 * (jd_emb @ jd_emb) ** 0.5 ))
        scores.append({"project": proj[:200], "similarity": round(sim, 3)})

    avg_score = sum(s["similarity"] for s in scores) / len(scores)

    return {
        "score": round(avg_score * 100, 2),
        "projects_found": len(projects),
        "details": scores
    }
