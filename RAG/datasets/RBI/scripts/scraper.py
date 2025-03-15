import os
import requests
from bs4 import BeautifulSoup
from pyhtml2pdf import converter

BASE_URL = "https://www.rbi.org.in/commonman/English/Scripts/FAQs.aspx"
OUTPUT_DIR = "RBI_FAQs_PDFs"

def get_faq_links():
    """Scrapes FAQ section links and extracts non-PDF FAQ pages."""
    response = requests.get(BASE_URL)
    soup = BeautifulSoup(response.content, "html.parser")
    
    faq_links = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "FAQs" in href and not href.lower().endswith(".pdf"):  # Exclude PDFs
            faq_links.append((requests.compat.urljoin(BASE_URL, href), link.text.strip()))
    
    return faq_links

def save_faq_as_pdf(url, title):
    """Converts a given FAQ page into a PDF and saves it."""
    if not title:  # Skip empty titles
        return

    safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)
    pdf_path = os.path.join(OUTPUT_DIR, f"{safe_title}.pdf")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Saving FAQ: {safe_title}")
    converter.convert(url, pdf_path)

def main():
    faq_links = get_faq_links()
    for url, title in faq_links:
        save_faq_as_pdf(url, title)

if __name__ == "__main__":
    main()
