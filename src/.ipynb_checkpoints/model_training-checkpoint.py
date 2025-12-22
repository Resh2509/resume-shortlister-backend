# src/model_training.py

import argparse
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

from sentence_transformers import SentenceTransformer
from src.text_cleaner import clean_text

EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text: str):
    """Convert cleaned resume text → BERT embedding"""
    if not text.strip():
        return np.zeros(EMBEDDER.get_sentence_embedding_dimension())
    return EMBEDDER.encode([text], show_progress_bar=False)[0]

def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    df["clean_text"] = df["resume_text"].fillna("").apply(clean_text)
    return df

def train(csv_path: str, out_dir: str):

    df = load_data(csv_path)
    X = df["clean_text"].tolist()
    y = df["label"].values

    print("Generating BERT embeddings...")
    X_emb = np.vstack([embed_text(t) for t in X])

    X_train, X_test, y_train, y_test = train_test_split(
        X_emb, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = SVC(kernel="rbf", probability=True)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    import os
    os.makedirs(out_dir, exist_ok=True)

    joblib.dump(model, f"{out_dir}/svm_bert.pkl")
    joblib.dump(scaler, f"{out_dir}/scaler.pkl")

    print("Saved BERT + SVM model to models/svm_bert.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--out_dir", default="models")
    args = parser.parse_args()

    train(args.train_csv, args.out_dir)
