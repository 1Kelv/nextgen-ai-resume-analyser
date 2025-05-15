import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
data = pd.read_csv('NextGen.csv')
data['combined_text'] = data['experience'].astype(str) + ' ' + data['skills'].astype(str) + ' ' + data['education'].astype(str) + ' ' + data['job_category'].astype(str)

# Vectorisation
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['combined_text'])
y = data['job_category']

# Model
model = LogisticRegression()
model.fit(X, y)

# Save model and vectorizer
with open('resume_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("✅ Model and vectorizer saved successfully.")
