def loan_eligibility_analysis():
    print("Welcome to the Loan Eligibility Chatbot!")
    print("This chatbot will help determine your loan eligibility based on several factors.")
    print("All information provided is used only for calculation and won't be stored.\n")
    
    # Initialize variables
    employment_score = None
    loan_amount = None
    annual_income = None
    loan_term = None
    cibil_score = None
    
    # Weights for calculation
    W1 = 15  # Employment
    W2 = 25  # Loan-to-Income Ratio
    W3 = 20  # Loan Term
    W4 = 30  # CIBIL Score
    
    # Collect Employment Information
    while employment_score is None:
        employment = input("What is your employment type? (Salaried/Self-Employed/Unemployed): ").strip().lower()
        if employment == "salaried":
            employment_score = 1.0
        elif employment == "self-employed":
            employment_score = 0.7
        elif employment == "unemployed":
            employment_score = 0.4
        else:
            print("Please enter either 'Salaried', 'Self-Employed', or 'Unemployed'.")
    
    # Collect Loan Amount
    while loan_amount is None:
        try:
            loan_amount = float(input("What loan amount are you seeking? (in currency units): ").replace(',', ''))
            if loan_amount <= 0:
                print("Please enter a positive loan amount.")
                loan_amount = None
        except ValueError:
            print("Please enter a valid numeric amount.")
    
    # Collect Annual Income (handle unemployed case)
    while annual_income is None:
        if employment == "unemployed":
            try:
                annual_income = float(input("Even though you're unemployed, do you have any annual income? (Enter 0 if none): ").replace(',', ''))
                if annual_income <= 0:
                    print("Please enter a valid amount (0 or positive).")
                    annual_income = None
            except ValueError:
                print("Please enter a valid numeric amount.")
        else:
            try:
                annual_income = float(input("What is your annual income? (in currency units): ").replace(',', ''))
                if annual_income <= 0:
                    print("Please enter a positive annual income.")
                    annual_income = None
            except ValueError:
                print("Please enter a valid numeric amount.")
    
    # Collect Loan Term
    while loan_term is None:
        try:
            loan_term = int(input("What is your preferred loan term? (in months): "))
            if loan_term <= 0 or loan_term > 360:
                print("Please enter a positive loan term between 1 and 360 months.")
                loan_term = None
        except ValueError:
            print("Please enter a valid number of months.")
    
    # Collect CIBIL Score
    while cibil_score is None:
        cibil_input = input("What is your CIBIL Score? (300-900, or enter 'unknown' if you don't know): ").strip().lower()
        if cibil_input == "unknown":
            print("For accurate eligibility assessment, knowing your CIBIL score is important.")
            print("You can check your CIBIL score at the official CIBIL website.")
            print("For this calculation, we'll use a neutral value of 650.")
            cibil_score = 650
        else:
            try:
                cibil_score = int(cibil_input)
                if cibil_score < 300 or cibil_score > 900:
                    print("CIBIL Score must be between 300 and 900.")
                    cibil_score = None
            except ValueError:
                print("Please enter a valid CIBIL Score or 'unknown'.")
    
    # Calculate the eligibility score
    loan_to_income_ratio = min(1.0, loan_amount / (annual_income if annual_income > 0 else 1))
    loan_term_normalized = loan_term / 360
    cibil_normalized = cibil_score / 900
    
    eligibility_score = (
        W1 * employment_score +
        W2 * (1 - loan_to_income_ratio) +  # Lower ratio is better
        W3 * (1 - loan_term_normalized) +  # Shorter term is better
        W4 * cibil_normalized
    )
    
    # Convert to 0-100 scale
    eligibility_score = min(100, max(0, eligibility_score))
    
    # Determine eligibility category
    if eligibility_score >= 85:
        category = "Platinum Borrower"
        approval = "Guaranteed"
        terms = [
            "0% Processing Fee",
            "Lowest Interest Rates (7-8%)",
            "Higher Loan Limits",
            "Pre-Approved Top-Up Loans"
        ]
        advice = [
            "You qualify for the best loan terms! Maintain your excellent financial habits to continue enjoying these benefits."
        ]
    elif eligibility_score >= 70:
        category = "Gold Borrower"
        approval = "High"
        terms = [
            "50% Processing Fee Waiver",
            "Competitive Interest Rates (8-9.5%)",
            "Faster Loan Disbursal"
        ]
        advice = [
            "You're in a strong position! Consider improving your CIBIL score or reducing your debt to qualify for even better terms."
        ]
    elif eligibility_score >= 55:
        category = "Silver Borrower"
        approval = "Moderate"
        terms = [
            "Partial (25%) Processing Fee Waiver",
            "Moderate Interest Rates (10-12%)",
            "Higher Down Payment Required"
        ]
        advice = [
            "You're on the right track! Focus on improving your credit score and maintaining a stable income to move up to a higher category."
        ]
    elif eligibility_score >= 40:
        category = "Basic Borrower"
        approval = "Low"
        terms = [
            "Higher Processing Fees",
            "High Interest Rates (12-16%)",
            "Collateral or Guarantor Required"
        ]
        advice = [
            "Consider improving your financial profile to qualify for better terms. Pay bills on time, reduce debt, and avoid multiple loan applications."
        ]
    else:
        category = "High-Risk Borrower"
        approval = "Very Low"
        terms = [
            "Very High Interest Rates (16-24%)",
            "Collateral & Guarantor Required",
            "Approval Subject to Additional Checks"
        ]
        advice = [
            "\nImprove your CIBIL score by paying bills on time and reducing outstanding debt.",
            "Consider adding a guarantor to increase your chances of approval.",
            "Explore secured loan options where collateral can be provided."
        ]
    
    # Display results
    print("\n" + "="*100)
    print(f"Your Loan Eligibility Score: {eligibility_score:.2f}")
    print(f"Category: {category}")
    print(f"Approval Chances: {approval}")
    print("\nLoan Terms and Conditions:")
    for term in terms:
        print(f"  - {term}")
    print("\nSuggestions to Improve Loan eligibility:")
    for tip in advice:
        print(f"  - {tip}")
    print("="*100)
    
    # Disclaimer
    print("Disclaimer: This is an indicative assessment only.")
    print("Actual loan approval and terms may vary based on the lender's policies.")
    print("No personal data has been stored during this calculation.")

if __name__ == "__main__":
    loan_eligibility_analysis()