import os
import json
import pdfplumber
import re
import nltk
from tqdm import tqdm

nltk.download("punkt")  # Ensure required data is available

# Directories
PDF_DIR = "RBI_FAQs_PDFs"
JSON_DIR = "RBI_FAQs_JSON_with_renewed_working"
DEBUG_DIR = "RBI_FAQs_Debug"
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

# Question Patterns (Handling Different Formats)
QUESTION_REGEX = re.compile(r"^(Q(?:\.)?|Q:|\d+\.)\s*(.+)", re.IGNORECASE)
ALT_QUESTION_REGEX = re.compile(r"^\(\w+\)\s*(.+)")  # Handles "(i)", "(ii)", "(iii)" type questions
ANSWER_REGEX = re.compile(r"^\s*(Ans:|Answer:)\s*", re.IGNORECASE)  # Detects answer

def extract_text_from_pdf(pdf_path):
    """Extracts raw text from a given PDF."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        return text.strip()
    except Exception as e:
        print(f"⚠️ Error extracting {pdf_path}: {e}")
        return ""

def extract_qna_pairs(text):
    """Extracts Q&A pairs, handling both numbered and unstructured formats."""
    lines = text.split("\n")  # Split text into lines
    qna_pairs = []
    current_question = None
    current_answer = []

    for line in lines:
        line = line.strip()

        # Match explicit "Q:" or "1." style questions
        match = QUESTION_REGEX.match(line) or ALT_QUESTION_REGEX.match(line)
        if match:
            # Save previous Q&A before starting a new one
            if current_question and current_answer:
                qna_pairs.append({"question": current_question, "answer": " ".join(current_answer)})
            
            current_question = match.group(1).strip()
            current_answer = []
        elif ANSWER_REGEX.match(line):  # Detect answers explicitly
            continue  # Skip "Ans:" and just collect the content
        elif current_question:  # Append to the answer
            current_answer.append(line)

    # Save last Q&A pair
    if current_question and current_answer:
        qna_pairs.append({"question": current_question, "answer": " ".join(current_answer)})

    return qna_pairs

def process_pdfs(pdf_dir, json_dir, debug_dir):
    """Processes all PDFs and saves extracted Q&A pairs as JSON."""
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    skipped_pdfs = []

    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_path = os.path.join(pdf_dir, pdf_file)
        json_filename = os.path.splitext(pdf_file)[0] + ".json"
        json_path = os.path.join(json_dir, json_filename)
        debug_txt_path = os.path.join(debug_dir, os.path.splitext(pdf_file)[0] + ".txt")

        raw_text = extract_text_from_pdf(pdf_path)
        if not raw_text.strip():
            print(f"⚠️ No extractable text found in {pdf_file}, skipping...")
            skipped_pdfs.append(pdf_file)
            continue

        qna_pairs = extract_qna_pairs(raw_text)

        if qna_pairs:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(qna_pairs, f, indent=4, ensure_ascii=False)
            print(f"✅ Saved: {json_path}")
        else:
            print(f"⚠️ No Q&A found in {pdf_file}, saving debug text...")
            with open(debug_txt_path, "w", encoding="utf-8") as f:
                f.write(raw_text)  # Save extracted text for debugging
            skipped_pdfs.append(pdf_file)

    with open(os.path.join(debug_dir, "skipped_files.log"), "w", encoding="utf-8") as f:
        for skipped_file in skipped_pdfs:
            f.write(skipped_file + "\n")

    print("\n🎉 All PDFs have been processed! Check debug logs for skipped files.")

if __name__ == "__main__":
    process_pdfs(PDF_DIR, JSON_DIR, DEBUG_DIR)
