import csv
import json

# Function to convert CSV to JSON
def csv_to_json(csv_file, json_file):
    data = {"questions_answers": []}

    # Read the CSV file
    with open(csv_file, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        # Process each row
        for row in reader:
            entry = {
                "question": row["Question"],
                "answer": row["Answer"],
                "category": row["Class"]
            }
            data["questions_answers"].append(entry)

    # Write to JSON file
    with open(json_file, mode="w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

    print(f"Conversion successful! JSON file saved as {json_file}")

# Example usage
csv_to_json("BankFAQs.csv", "BankFAQs.json")
