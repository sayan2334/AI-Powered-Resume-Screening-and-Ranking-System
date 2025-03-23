import streamlit as st
import spacy
import pandas as pd
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = " ".join(page.extract_text() or "" for page in pdf.pages)
    return text

# Function to clean and preprocess text
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    text = text.lower()  # Convert to lowercase
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    # Preprocess job description and resumes
    job_description = preprocess_text(job_description)
    resumes = [preprocess_text(resume) for resume in resumes]
    
    # TF-IDF Vectorization with bigrams & optimized parameters
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.85, min_df=1, stop_words='english')
    vectors = vectorizer.fit_transform([job_description] + resumes).toarray()

    # Compute cosine similarity
    cosine_similarities = cosine_similarity([vectors[0]], vectors[1:]).flatten()
    
    # Weighted scoring (Cosine Similarity + Keyword Matching)
    important_keywords = {"java", "python", "c++","oop", "data structure","sql","algorithm", "spring boot", "django","aws"}
    weighted_scores = [(cos_sim * 0.8 + len(set(resume.split()) & important_keywords) * 0.2) * 10
                       for cos_sim, resume in zip(cosine_similarities, resumes)]
    
    return weighted_scores

# Streamlit app UI
st.title("📄 AI Resume Screening & Candidate Ranking System")

# Job description input
st.header("Job Description")
job_description = st.text_area("Enter the job description")

# File uploader
st.header("Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF resumes", type=["pdf"], accept_multiple_files=True)

# Button to trigger ranking
if uploaded_files and job_description:
    if st.button("Rank Resumes"):
        st.header("Ranking Resumes")
        resumes = [extract_text_from_pdf(file) for file in uploaded_files]

        # Rank resumes with automatic scale of 10
        scores = rank_resumes(job_description, resumes)

        # Create DataFrame for table display with ranking based on score
        results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores})
        results = results.sort_values(by="Score", ascending=False)
        results.insert(0, "Rank", range(1, len(results) + 1))  # Adding Rank Column

        # Display results in a table format with full score in the header
        st.write(f"### 📊 Ranked Resumes (Total: {len(uploaded_files)}) - Scoring Scale: 10")
        st.dataframe(results.style.format({"Score": "{:.3f}"}))
