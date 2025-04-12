import streamlit as st
import pandas as pd
import io
import os
import tempfile
from resume_parser import parse_resume
from job_analyzer import extract_job_requirements
from relevance_scorer import calculate_relevance_scores, rank_resumes
from utils import allowed_file, display_resume_feedback

# Page config
st.set_page_config(
    page_title="AI-Powered Resume Scanner",
    page_icon="📄",
    layout="wide"
)

# Application title and description
st.title("📄 AI-Powered Resume Scanner")
st.markdown("""
This application helps streamline the recruitment process by automatically parsing, 
analyzing, and ranking resumes based on their relevance to specific job descriptions.
""")

# Session state initialization
if 'uploaded_resumes' not in st.session_state:
    st.session_state.uploaded_resumes = []
if 'job_description' not in st.session_state:
    st.session_state.job_description = ""
if 'parsed_resumes' not in st.session_state:
    st.session_state.parsed_resumes = []
if 'job_requirements' not in st.session_state:
    st.session_state.job_requirements = {}
if 'ranked_resumes' not in st.session_state:
    st.session_state.ranked_resumes = []
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'processing_started' not in st.session_state:
    st.session_state.processing_started = False
if 'step' not in st.session_state:
    st.session_state.step = 1

# Sidebar for application steps
with st.sidebar:
    st.header("Application Steps")
    step_1 = st.sidebar.button("1. Upload Resumes", 
                              disabled=st.session_state.step == 1,
                              type="primary" if st.session_state.step == 1 else "secondary")
    step_2 = st.sidebar.button("2. Enter Job Description", 
                              disabled=st.session_state.step == 2,
                              type="primary" if st.session_state.step == 2 else "secondary")
    step_3 = st.sidebar.button("3. View Results", 
                              disabled=st.session_state.step == 3,
                              type="primary" if st.session_state.step == 3 else "secondary")
    
    if step_1:
        st.session_state.step = 1
        st.rerun()
    if step_2:
        st.session_state.step = 2
        st.rerun()
    if step_3 and st.session_state.processing_complete:
        st.session_state.step = 3
        st.rerun()

# Step 1: Upload Resumes
if st.session_state.step == 1:
    st.header("Step 1: Upload Resumes")
    
    uploaded_files = st.file_uploader(
        "Upload resumes (PDF, DOCX, DOC)", 
        accept_multiple_files=True,
        type=["pdf", "docx", "doc"]
    )
    
    if uploaded_files:
        valid_files = []
        for uploaded_file in uploaded_files:
            if allowed_file(uploaded_file.name):
                valid_files.append(uploaded_file)
            else:
                st.error(f"Invalid file format: {uploaded_file.name}. Please upload PDF, DOCX, or DOC files only.")
        
        if valid_files:
            if st.button("Continue to Job Description"):
                st.session_state.uploaded_resumes = valid_files
                st.session_state.step = 2
                st.rerun()
    else:
        st.info("Please upload at least one resume to continue.")

# Step 2: Enter Job Description
elif st.session_state.step == 2:
    st.header("Step 2: Enter Job Description")
    
    job_description = st.text_area(
        "Paste the job description here:",
        height=300,
        value=st.session_state.job_description
    )
    
    jd_file = st.file_uploader("Or upload a job description file (TXT)", type=["txt"])
    
    if jd_file is not None:
        job_description = jd_file.getvalue().decode("utf-8")
        st.text_area("Job Description from File:", value=job_description, height=200)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("Back to Resume Upload"):
            st.session_state.step = 1
            st.rerun()
    
    with col2:
        if job_description and st.button("Analyze Resumes"):
            st.session_state.job_description = job_description
            st.session_state.processing_started = True
            
            # Process job description
            with st.spinner("Analyzing job description..."):
                job_requirements = extract_job_requirements(job_description)
                st.session_state.job_requirements = job_requirements
            
            # Process resumes
            parsed_resumes = []
            with st.spinner("Parsing resumes..."):
                for resume_file in st.session_state.uploaded_resumes:
                    # Create a temporary file to store the uploaded file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(resume_file.name)[1]) as tmp_file:
                        tmp_file.write(resume_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Parse the resume
                    parsed_resume = parse_resume(tmp_path, resume_file.name)
                    parsed_resumes.append(parsed_resume)
                    
                    # Delete the temporary file
                    os.unlink(tmp_path)
                
                st.session_state.parsed_resumes = parsed_resumes
            
            # Calculate scores and rank resumes
            with st.spinner("Calculating relevance scores and ranking resumes..."):
                relevance_scores = calculate_relevance_scores(
                    st.session_state.parsed_resumes, 
                    st.session_state.job_requirements
                )
                
                ranked_resumes = rank_resumes(
                    st.session_state.parsed_resumes, 
                    relevance_scores
                )
                
                st.session_state.ranked_resumes = ranked_resumes
                st.session_state.processing_complete = True
                st.session_state.step = 3
                st.rerun()

# Step 3: View Results
elif st.session_state.step == 3 and st.session_state.processing_complete:
    st.header("Step 3: View Results")
    
    # Display job requirements
    with st.expander("Job Requirements Analysis", expanded=True):
        st.subheader("Required Skills")
        
        # Convert the skills to a DataFrame for better display
        skills_df = pd.DataFrame({
            "Skills": list(st.session_state.job_requirements.get("skills", [])),
            "Category": ["Technical" for _ in st.session_state.job_requirements.get("skills", [])]
        })
        
        if not skills_df.empty:
            st.dataframe(skills_df, hide_index=True)
        else:
            st.info("No specific skills identified in the job description.")
        
        st.subheader("Required Experience")
        st.write(", ".join(st.session_state.job_requirements.get("experience", ["Not specified"])))
        
        st.subheader("Required Education")
        st.write(", ".join(st.session_state.job_requirements.get("education", ["Not specified"])))
    
    # Display ranked resumes
    st.subheader("Ranked Resumes")
    
    if not st.session_state.ranked_resumes:
        st.warning("No resumes were processed. Please go back and try again.")
    else:
        for i, resume in enumerate(st.session_state.ranked_resumes):
            with st.expander(f"{i+1}. {resume['filename']} - Match Score: {resume['match_score']:.2f}%", expanded=i == 0):
                
                # Display resume content in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### Contact Information")
                    st.write(resume.get("contact_info", "Not available"))
                    
                    st.markdown("##### Skills")
                    if resume.get("skills"):
                        skills_list = resume.get("skills", [])
                        st.write(", ".join(skills_list))
                    else:
                        st.info("No skills identified")
                
                with col2:
                    st.markdown("##### Education")
                    if resume.get("education"):
                        for edu in resume.get("education", []):
                            st.write(edu)
                    else:
                        st.info("No education information identified")
                    
                    st.markdown("##### Experience")
                    if resume.get("experience"):
                        for exp in resume.get("experience", []):
                            st.write(exp)
                    else:
                        st.info("No experience information identified")
                
                # Display feedback
                st.markdown("##### Feedback")
                display_resume_feedback(
                    resume, 
                    st.session_state.job_requirements
                )
    
    # Buttons for navigation
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("Start Over"):
            # Reset session state
            st.session_state.uploaded_resumes = []
            st.session_state.job_description = ""
            st.session_state.parsed_resumes = []
            st.session_state.job_requirements = {}
            st.session_state.ranked_resumes = []
            st.session_state.processing_complete = False
            st.session_state.processing_started = False
            st.session_state.step = 1
            st.rerun()
    
    with col2:
        if st.button("Edit Job Description"):
            st.session_state.step = 2
            st.rerun()

# Display message when processing has started but not completed
elif st.session_state.processing_started and not st.session_state.processing_complete:
    st.info("Processing your resumes. Please wait...")
