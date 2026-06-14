Resume Shortlister Backend

Overview

Resume Shortlister Backend is an AI-powered resume screening system that automates candidate shortlisting by analyzing resumes and matching them against job descriptions.

The system extracts text from PDF resumes, preprocesses content using Natural Language Processing (NLP), generates feature vectors, and evaluates candidate suitability through machine learning classification and ATS-style matching techniques.

This project helps recruiters reduce manual screening effort and identify qualified candidates more efficiently.


Problem Statement

Recruiters often receive hundreds of resumes for a single job posting.

Manual screening is:

* Time-consuming
* Inconsistent
* Prone to human bias
* Difficult to scale

This system automates resume evaluation by ranking candidates based on relevance to job requirements.


Features

Resume Parsing

* Extract text from PDF resumes
* Handle multiple resume formats
* Structured content preprocessing

NLP Processing

* Text cleaning and normalization
* Stopword removal
* Tokenization
* Feature extraction

ATS Matching

* Keyword-based scoring
* Skill matching
* Job description similarity analysis
* Resume ranking

Machine Learning Classification

* Resume suitability prediction
* Candidate categorization
* Automated shortlisting

Backend API

* Upload resume files
* Process candidate profiles
* Return ATS scores
* Generate shortlist recommendations



Technology Stack

Backend

* Python
* Flask / FastAPI (update based on your implementation)

Machine Learning

* Scikit-learn
* Pandas
* NumPy

NLP

* NLTK
* TF-IDF Vectorization

File Processing

* PDF Parsing Libraries

Development

* Jupyter Notebook


## Project Structure

```text
resume-shortlister-backend/
│
├── app/
├── data/
├── models/
├── src/
├── temp/
│
├── api.py
├── main.py
├── requirements.txt
├── README.md
└── Untitled.ipynb
```
Workflow

1. Upload candidate resume
2. Extract text from PDF
3. Perform NLP preprocessing
4. Generate feature vectors
5. Compare against job description
6. Calculate ATS score
7. Predict candidate suitability
8. Return ranked results

---

System Architecture

```text
Resume PDF
      │
      ▼
Text Extraction
      │
      ▼
NLP Preprocessing
      │
      ▼
Feature Engineering
      │
      ▼
TF-IDF Vectorization
      │
      ▼
ML Classification
      │
      ▼
ATS Matching Engine
      │
      ▼
Candidate Ranking
```

Installation

```bash
git clone https://github.com/Resh2509/resume-shortlister-backend.git

cd resume-shortlister-backend

pip install -r requirements.txt
```


Running the Project

```bash
python main.py
```


```bash
python api.py
```

Skills Demonstrated

* Machine Learning
* Natural Language Processing
* Resume Parsing
* ATS Scoring Systems
* Feature Engineering
* Python Backend Development
* API Development
* Data Preprocessing
* Model Deployment

Future Improvements

* BERT-based Resume Matching
* Semantic Skill Extraction
* Interview Recommendation Engine
* Candidate Ranking Dashboard
* Recruiter Analytics Portal
* Explainable AI Scoring

Author

Reshma R

Aspiring AI Engineer | Machine Learning Developer | Backend Developer
