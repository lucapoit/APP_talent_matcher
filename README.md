# Smart Talent Matcher

A resume-to-job matching Streamlit application that uses semantic similarity to identify how well a candidate's resume aligns with various job positions. Designed as part of a technical evaluation for a Data Science role at a talent-matching company.

---

## 🚀 Features

- 📄 Upload a PDF resume and automatically extract its content.
- 🧠 Uses **sentence-transformers** to semantically match skills — not just exact keyword matching.
- 🧰 Differentiates between **hard** and **soft** skills and weights them accordingly.
- 🔍 Displays matched skills with:
  - Top matched sentence from the resume
  - Similarity scores
  - Number of contextual mentions
- 🧠 Penalizes low-confidence phrases (e.g., "basic", "beginner") to improve accuracy.
- 📋 Accepts job descriptions via file upload or text input.

---

## 📦 Tech Stack

- **Python**
- **Streamlit**
- **PDFPlumber** (for resume text extraction)
- **NLTK** (tokenization)
- **Sentence Transformers** (`all-mpnet-base-v2`)
- **Scikit-learn** (cosine similarity)

---

## 💡 Strengths

- **Semantic Matching**: Goes beyond simple keyword matching using pre-trained language models.
- **Explainability**: Clearly displays matched text snippets and scores to build trust.
- **Flexible Input**: Job positions can be uploaded or pasted manually.
- **Weighting Logic**: Hard skills are weighted more heavily, and low-confidence contexts are penalized.
- **Clean UI**: Expandable skill displays, match scores, and resume text review all presented clearly.

---

## ⚠️ Limitations

- **Local Embedding Only**: Matching is done in memory; not yet optimized for large-scale database search (e.g., FAISS or Pinecone).
- **Free model used**: Room to improvemore powerful models and bigger context windows. 
- **PDF Quality Dependency**: If resume text is poorly structured or scanned, extraction may fail.
- **Not Personalized**: No personalization or feedback loop for hiring teams (yet).
- **Lacks Role-Specific Models**: Generic model used for all domains — could be improved with domain-specific fine-tuning.
- **Some hard-coded content**: low confidence contexts and templates for context matching currently are hard coded.

---

## ✅ Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/lucapoit/APP_talent_matcher.git
   cd APP_talent_matcher
