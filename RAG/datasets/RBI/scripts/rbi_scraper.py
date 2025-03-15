from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import os
from weasyprint import HTML

HEADERS = {"User-Agent": "Mozilla/5.0"}

# Set up Selenium WebDriver
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # Run in headless mode
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

def get_faq_sections(main_url):
    driver = webdriver.Chrome(service=webdriver.ChromeService(ChromeDriverManager().install()), options=options)
    driver.get(main_url)
    
    # Let JavaScript load content
    driver.implicitly_wait(5)
    page_html = driver.page_source
    driver.quit()
    
    soup = BeautifulSoup(page_html, "html.parser")

    print("✅ Page loaded successfully. Checking structure...")
    print(soup.prettify()[:1000])  # Print first 1000 characters to see structure

    faq_sections = {}
    for section in soup.find_all("div", class_=["panel-group", "panel-body"]):
        header = section.find("h4")
        if not header:
            continue
        section_title = header.text.strip()
        links = []
        for link in section.find_all("a", href=True):
            href = link["href"]
            if href.startswith("/Scripts/"):
                full_url = f"https://www.rbi.org.in{href}"
                section_links.append((header.text.strip(), full_url))
    
    return faq_sections

def main():
    main_url = "https://www.rbi.org.in/Scripts/FAQView.aspx?Id=100"
    
    print("🔄 Fetching FAQ sections...")
    faq_sections = get_faq_sections(main_url)
    
    if not faq_sections:
        print("⚠️ No FAQ sections found. Trying again with Selenium...")
        return

    for section, links in faq_sections.items():
        print(f"\n=== Section: {section} ===")
        for title, url in links:
            print(f"📄 {title} -> {url}")

if __name__ == "__main__":
    main()
