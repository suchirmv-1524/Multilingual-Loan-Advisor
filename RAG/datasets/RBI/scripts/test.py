import requests

main_url = "https://www.rbi.org.in/Scripts/FAQView.aspx?Id=100"
response = requests.get(main_url)

if response.status_code == 200:
    print("Page loaded successfully!")
    print(response.text[:1000])  # Print the first 1000 characters
else:
    print(f"Failed to load page. Status code: {response.status_code}")
