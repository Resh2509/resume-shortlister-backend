# src/pdf_extractor.py

from typing import List
import pdfplumber
from PyPDF2 import PdfReader


def extract_text_from_pdf(path: str) -> str:
    """
    Try pdfplumber first (better results),
    fallback to PyPDF2 if it fails.
    """
    text_parts: List[str] = []

    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

    except Exception:
        # Fallback to PyPDF2
        try:
            reader = PdfReader(path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        except Exception as e:
            raise RuntimeError(f"Failed to extract PDF text: {e}")

    return "\n".join(text_parts)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        print(extract_text_from_pdf(sys.argv[1]))
    else:
        print("Usage: python pdf_extractor.py <file.pdf>")
