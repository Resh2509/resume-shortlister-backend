# main.py (PROJECT ROOT)

import argparse
from src.predictor import ResumePredictor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--jd", default="")
    args = parser.parse_args()

    rp = ResumePredictor(
        model_path="models/svm_bert.pkl",
        scaler_path="models/scaler.pkl"
    )

    result = rp.predict_from_pdf(
        pdf_path=args.pdf,
        job_description=args.jd
    )

    print(result)

if __name__ == "__main__":
    main()
