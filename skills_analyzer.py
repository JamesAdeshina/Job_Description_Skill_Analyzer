import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# pip install pydantic==1.10.12

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Sample Job Description
job_description = """
We are looking for a Data Scientist proficient in Python, SQL, and Machine Learning.
Experience with cloud platforms like AWS or Azure is a plus. Knowledge of Tableau and data visualization is desired.
"""

# Predefined list of skills
skills_list = [
    "python", "sql", "machine learning", "aws", "azure",
    "data visualization", "tableau", "deep learning", "statistics"
]

# Preprocessing function
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

# Extract skills using keyword matching
def extract_skills(text, skills_list):
    text = preprocess_text(text)
    extracted_skills = [skill for skill in skills_list if skill in text]
    return extracted_skills

# Summarize using TF-IDF
def summarize_text(text):
    vectorizer = TfidfVectorizer(max_df=0.8, stop_words="english", max_features=5)
    tfidf_matrix = vectorizer.fit_transform([text])
    top_words = vectorizer.get_feature_names_out()
    return "Key terms: " + ", ".join(top_words)

# Preprocess and analyze
processed_text = preprocess_text(job_description)
skills = extract_skills(job_description, skills_list)
summary = summarize_text(processed_text)

# Results
print("Skills Extracted:", skills)
print(summary)