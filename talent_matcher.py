"""
Filename: talent_matcher
Author: Luca Poit
Date: 2025-04-29
Version: 1.0
Description: This script is the source of an app that uses NLP to match a CV with current job openings. See readme for more.

"""


import streamlit as st
import pdfplumber
import json
import re
import pandas as pd
import unidecode
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
from nltk.util import ngrams


model = SentenceTransformer('all-mpnet-base-v2')  


# Load job positions
def load_job_positions(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading job positions file: {e}")
        return []

# Extract text from uploaded PDF
def extract_text_from_pdf(file):
    try:
        with pdfplumber.open(file) as pdf:
            return "\n".join(
                page.extract_text() for page in pdf.pages if page.extract_text()
            )
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return ""

# Clean and tokenize text
def preprocess_text(text):
    text = unidecode.unidecode(text.lower())  # Convert to lowercase and remove accents
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return list(text.split())

# Match skills in resume with job descriptions
def match_skills(resume_tokens, job_positions):

    results = []

    for job in job_positions:

        resume_text = " ".join(resume_tokens) 

        hard_matches = semantic_match(job["hard_skills"], resume_text)
        soft_matches = semantic_match(job["soft_skills"], resume_text)

        score = len(hard_matches) * 1.5 + len(soft_matches)  # weight hard skills more
        results.append({
            "Job Title": job["title"],
            "Hard Skill Matches": list(hard_matches),
            "Soft Skill Matches": list(soft_matches),
            "Match Score": score
        })
    return sorted(results, key=lambda x: x["Match Score"], reverse=True)


def semantic_match(skills_list, resume_text, threshold=0.6, weak_penalty=0.6):

    WEAK_CONTEXT_INDICATORS = ['basic', 'beginner', 'introductory', 'familiar with', 'exposure to']

    tokens = nltk.word_tokenize(resume_text.lower())
    chunks = [' '.join(gram) for gram in ngrams(tokens, 4)]
    chunk_embeddings = model.encode(chunks)

    templates = [
        "large experience with {}",
        "worked with {} for a long time",
        "used {} on a daily basis",
        "developed in {}",
        "built solutions using {}",
        "implemented {}",
        "hands-on with {}",
        "projects involving {}"
    ]

    matched_skills = []

    for skill in skills_list:
        all_adjusted_similarities = []
        matched_chunks = []

        for template in templates:
            phrase = template.format(skill)
            skill_embedding = model.encode([phrase])[0]
            similarities = cosine_similarity([skill_embedding], chunk_embeddings)[0]

            for idx, sim in enumerate(similarities):
                chunk = chunks[idx]
                adjusted_sim = sim

                if any(weak_word in chunk for weak_word in WEAK_CONTEXT_INDICATORS):
                    adjusted_sim *= weak_penalty

                if adjusted_sim >= threshold:
                    all_adjusted_similarities.append(adjusted_sim)
                    matched_chunks.append(chunk)


        if all_adjusted_similarities:
            matched_skills.append({
                "skill": skill,
                "mention_count": len(all_adjusted_similarities),
                "average_score": np.round(float(np.mean(all_adjusted_similarities)), 3),
                "top_score": np.round(float(np.max(all_adjusted_similarities)), 3),
                "top_chunk": matched_chunks[np.argmax(all_adjusted_similarities)],
                "all_chunks": matched_chunks
            })

    return matched_skills



# Streamlit app layout
st.title("Smart Talent Matcher")
st.markdown("Upload a resume and find matching job openings based on skills!")

uploaded_file = st.file_uploader("Insert resume in PDF format", type="pdf")

if uploaded_file:

    resume_text = extract_text_from_pdf(uploaded_file)

    preprocessed_text = preprocess_text(resume_text)

    st.subheader("Job Descriptions Source")
    source_type = st.radio("Choose input method:", ["Upload JSON", "Paste JSON"])

    job_positions = []

    if source_type == "Upload JSON":
        job_file = st.file_uploader("Upload job positions JSON", type="json")
        if job_file:
            try:
                job_positions = json.load(job_file)
            except Exception as e:
                st.error(f"Could not parse uploaded file: {e}")

    elif source_type == "Paste JSON":
        pasted_text = st.text_area("Paste job positions JSON here")
        if pasted_text:
            try:
                job_positions = json.loads(pasted_text)
            except Exception as e:
                st.error(f"Invalid JSON: {e}")


    matches = match_skills(preprocessed_text, job_positions)

    st.subheader("Top Job Matches")

    for i, match in enumerate(matches):
        if i == 0:
            st.markdown(f"## 🔥 Top Match: {match['Job Title']}")
        else:
            st.markdown(f"### {match['Job Title']}")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(label="Match Score", value=match['Match Score'])

        st.write("**Hard Skills Matched:**")
        for hs in match["Hard Skill Matches"]:
            with st.expander(f"🛠️ {hs['skill']}"):
                st.write(f"**Top Score**: {hs['top_score']}")
                st.write(f"**Average Score**: {hs['average_score']}")
                st.write(f"**Mention Count**: {hs['mention_count']}")
                st.markdown(f"**Top Matched Text:**\n\n_{hs['top_chunk']}_")

                if hs.get("all_chunks"):
                    st.markdown("**All Matching Chunks:**")
                    for idx, chunk in enumerate(hs["all_chunks"], 1):
                        st.markdown(f"{idx}. _{chunk}_")

        st.write("**Soft Skills Matched:**")
        for ss in match["Soft Skill Matches"]:
            with st.expander(f"🤝 {ss['skill']}"):
                st.write(f"**Top Score**: {ss['top_score']}")
                st.write(f"**Average Score**: {ss['average_score']}")
                st.write(f"**Mention Count**: {ss['mention_count']}")
                st.markdown(f"**Top Matched Text:**\n\n_{ss['top_chunk']}_")

                if ss.get("all_chunks"):
                    st.markdown("**All Matching Chunks:**")
                    for idx, chunk in enumerate(ss["all_chunks"], 1):
                        st.markdown(f"{idx}. _{chunk}_")


        st.markdown("---")



