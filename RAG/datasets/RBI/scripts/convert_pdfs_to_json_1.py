import os
import json
import pdfplumber
import re
import nltk
from tqdm import tqdm
from nltk.tokenize import sent_tokenize

# Download NLTK tokenizer if not available
nltk.download("punkt")

# Directories
PDF_DIR = "RBI_FAQs_PDFs"
JSON_DIR = "RBI_FAQs_JSON"
os.makedirs(JSON_DIR, exist_ok=True)  # Ensure JSON output directory exists

# Improved Question Regex Pattern
QUESTION_REGEX = re.compile(r"^(Q|Q\.|Q:|\d+\.)\s*(.*)", re.IGNORECASE)

def extract_text_from_pdf(pdf_path):
    """Extracts raw text from a given PDF."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        return text
    except Exception as e:
        print(f"⚠️ Error extracting {pdf_path}: {e}")
        return ""

def extract_qna_pairs(text):
    """Extracts Q&A pairs from raw PDF text."""
    sentences = sent_tokenize(text)  # Split into sentences
    qna_pairs = []
    current_question = None
    current_answer = []

    for sentence in sentences:
        match = QUESTION_REGEX.match(sentence)  # Identify a question
        if match:
            if current_question and current_answer:
                qna_pairs.append({"question": current_question, "answer": " ".join(current_answer)})
            
            current_question = match.group(2).strip()  # Extract the question text
            current_answer = []
        elif current_question:  # If already in a Q&A block, append answers
            current_answer.append(sentence.strip())

    if current_question and current_answer:
        qna_pairs.append({"question": current_question, "answer": " ".join(current_answer)})

    return qna_pairs

def process_pdfs(pdf_dir, json_dir):
    """Processes all PDFs and saves extracted Q&A pairs as JSON."""
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_path = os.path.join(pdf_dir, pdf_file)
        json_filename = os.path.splitext(pdf_file)[0] + ".json"
        json_path = os.path.join(json_dir, json_filename)

        # Extract and process text
        raw_text = extract_text_from_pdf(pdf_path)
        if not raw_text.strip():  
            print(f"⚠️ No extractable text found in {pdf_file}, skipping...")
            continue

        qna_pairs = extract_qna_pairs(raw_text)

        if qna_pairs:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(qna_pairs, f, indent=4, ensure_ascii=False)
            print(f"✅ Saved: {json_path}")
        else:
            print(f"⚠️ No Q&A found in {pdf_file}, skipping...")

if __name__ == "__main__":
    process_pdfs(PDF_DIR, JSON_DIR)
    print("\n🎉 All PDFs have been processed into JSON!")
