from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

model = joblib.load("backend/loan_approval_model.joblib")

# Categorical mappings exactly same as in model training
gender_map = {"Male": 0, "Female": 1}
marital_status_map = {"Single": 0, "Married": 1, "Divorced": 2}
education_map = {"High School": 0, "Graduate": 1, "Postgraduate": 2}
employment_status_map = {"Unemployed": 0, "Employed": 1, "Self-Employed": 2}
occupation_type_map = {"Business": 0, "Freelancer": 1, "Professional": 2, "Salaried": 3}
residential_status_map = {"Rent": 0, "Own": 1, "Other": 2}
city_town_map = {"Rural": 0, "Suburban": 1, "Urban": 2}
loan_purpose_map = {"Education": 0, "Home": 1, "Personal": 2, "Vehicle": 3}
loan_type_map = {"Unsecured": 0, "Secured": 1}
co_applicant_map = {"Yes": 0, "No": 1}


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Map categorical to numeric, default -1 if missing/unrecognized
    gender = gender_map.get(data.get("gender"), -1)
    marital_status = marital_status_map.get(data.get("maritalStatus"), -1)
    education = education_map.get(data.get("education"), -1)
    employment_status = employment_status_map.get(data.get("employmentStatus"), -1)
    occupation_type = occupation_type_map.get(data.get("occupationType"), -1)
    residential_status = residential_status_map.get(data.get("residentialStatus"), -1)
    city_town = city_town_map.get(data.get("cityTown"), -1)
    loan_purpose = loan_purpose_map.get(data.get("loanPurpose"), -1)
    loan_type = loan_type_map.get(data.get("loanType"), -1)
    co_applicant = co_applicant_map.get(data.get("coApplicant"), -1)

    features = [
        data.get("creditScore", 0),
        gender,
        data.get("age", 0),
        marital_status,
        data.get("dependents", 0),
        education,
        employment_status,
        occupation_type,
        residential_status,
        city_town,
        data.get("annualIncome", 0),
        data.get("monthlyExpenses", 0),
        data.get("existingLoans", 0),
        data.get("totalExistingLoanAmount", 0),
        data.get("outstandingDebt", 0),
        data.get("loanHistory", 0),  
        data.get("loanAmountRequested", 0),
        data.get("loanTerm", 0),
        loan_purpose,
        data.get("interestRate", 0),
        loan_type,
        co_applicant,
        data.get("transactionFrequency", 0),  
        data.get("defaultRisk", 0), 
    ]

    prediction = model.predict([features])[0]
    result = "Approved" if prediction == 1 else "Rejected"

    return jsonify({"loanApprovalStatus": result})


if __name__ == "__main__":
    app.run(debug=True)
