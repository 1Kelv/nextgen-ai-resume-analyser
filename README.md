# NextGen AI – Fair & Transparent CV Analyser

NextGen AI is a smart CV analyser designed to bring fairness and transparency into recruitment decisions. Built during an Agile group University project, it uses explainable AI tools like **SHAP (SHapley Additive exPlanations)** and **LIME (Local Interpretable Model-agnostic Explanations)** to reveal how decisions are made and includes a **Fairness Audit** that checks for discrimination, especially in education or experience-based selections.

## Features

- **Resume Parsing**: Extracts information from CVs in varied formats.
- **Bias Detection**: Flags potential bias using SHAP and LIME explainability tools.
- **Fairness Audit**: Uses selection rate charts to assess and mitigate unfair outcomes.
- **Risk Management Plan**: Identifies and tracks project risks from Sprint 0 to 3.
- **Transparent Scoring**: Clear reasoning behind AI decisions for better trust.

## Tech Stack

- **Python**
- **Streamlit**
- **Pandas / NumPy**
- **scikit-learn**
- **SHAP**
- **LIME**
- **Matplotlib**
- **CSV-based dataset for CVs**

## How to Run the App

1. Clone the repository or download the code files.

2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   once all dependencies are installed, run streamlit run app.py


## To auto commit on github

git add app.py 
## change app.py to any file you update

git commit -m "Incude the commit message"

## change update message to the relevant message
git push
