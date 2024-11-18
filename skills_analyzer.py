import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

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
