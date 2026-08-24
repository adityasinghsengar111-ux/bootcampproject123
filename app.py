import streamlit as st
import fitz  # PyMuPDF
import ollama
import json
import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Optional
from io import BytesIO

# 1. Define Structured Data Schemes for Strict AI Extraction
class Experience(BaseModel):
    company: str = Field(description="Name of the company or organization")
    role: str = Field(description="Job title or position held")
    duration: str = Field(description="Time period, e.g., '2 years' or '2022-2024'")

class CandidateProfile(BaseModel):
    name: str = Field(description="Full name of the candidate")
    email: str = Field(description="Email address")
    skills: List[str] = Field(description="Key technical and soft skills listed")
    experience: List[Experience] = Field(description="List of previous jobs and roles")
    match_score: int = Field(description="An automated score from 0 to 100 indicating how well they match general technical roles based on skills.")
    justification: str = Field(description="A short 2-sentence explanation for the score assigned.")

# 2. Extract Clean Text from Uploaded PDF Files
def extract_text_from_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# 3. Use Local Ollama LLM to Parse Text Into Strict JSON Structure
def parse_resume_with_llm(resume_text):
    prompt = f"""
    You are an expert HR Data Extraction AI system. Analyze the following resume text and extract the details precisely into the requested JSON schema.
    
    Resume Content:
    {resume_text}
    """
    
    try:
        response = ollama.chat(
            model='llama3',  # Ensure you have 'llama3' or 'mistral' pulled locally via Ollama
            messages=[{'role': 'user', 'content': prompt}],
            format=CandidateProfile.model_json_schema() # Forces the local LLM to return valid structural JSON
        )
        
        # Parse output safely
        data = json.loads(response['message']['content'])
        return data
    except Exception as e:
        st.error(f"Error parsing resume details: {e}")
        return None

# 4. Interactive Streamlit Interface Construction
st.set_page_config(page_title="AI Resume Parser & Screener", layout="wide")
st.title("📄 AI Resume Parser & Candidate Screener Dashboard")
st.subheader("Extract structure and instantly score candidates locally using Open-Source AI models")

# Sidebar Configuration for HR Filters
st.sidebar.header("Target Candidate Requirements")
min_experience = st.sidebar.slider("Minimum Required Experience (Years)", 0, 15, 2)
target_skills = st.sidebar.multiselect(
    "Preferred Keyword Filtering", 
    ["Python", "React", "SQL", "Machine Learning", "Project Management", "Docker", "Java", "Kubernetes"]
)

# Document Uploader Segment
uploaded_files = st.file_uploader("Upload Candidate Resumes (PDF Format)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    all_candidates_data = []
    
    # Initialization of tracking bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for index, file in enumerate(uploaded_files):
        status_text.text(f"Processing candidate: {file.name}...")
        
        # Extraction & Conversion Pipeline
        pdf_bytes = file.read()
        extracted_text = extract_text_from_pdf(pdf_bytes)
        parsed_json = parse_resume_with_llm(extracted_text)
        
        if parsed_json:
            # Flatten profile for simple presentation grid display
            flat_profile = {
                "Candidate Name": parsed_json.get("name", "Unknown"),
                "Email": parsed_json.get("email", "N/A"),
                "Skills": ", ".join(parsed_json.get("skills", [])),
                "Experience (Years)": len(parsed_json.get("experience", [])),
                "Match Score (%)": parsed_json.get("match_score", 0),
                "Justification": parsed_json.get("justification", "")
            }
            all_candidates_data.append(flat_profile)
        
        progress_bar.progress((index + 1) / len(uploaded_files))
    
    status_text.text("✨ Batch processing completed successfully!")
    st.session_state['master_df'] = pd.DataFrame(all_candidates_data)

# Interactive Presentation Data Table
if 'master_df' in st.session_state and not st.session_state['master_df'].empty:
    df_working = st.session_state['master_df'].copy()
    
    # Filter operations based on human parameters in sidebar
    filtered_df = df_working[df_working["Experience (Years)"] >= min_experience]
    filtered_df = filtered_df.sort_values(by="Match Score (%)", ascending=False)
    
    st.write("---")
    st.subheader(f"📊 Screened Candidate Shortlist ({len(filtered_df)} Profiles Match Criteria)")
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)
    
    # Data Export Operations
    col_csv, col_docx = st.columns(2)
    with col_csv:
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export Shortlist as CSV", data=csv_data, file_name="shortlisted_candidates.csv", mime="text/csv")