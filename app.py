import streamlit as st
import pandas as pd
import io
import os
import tempfile
from resume_parser import parse_resume
from job_analyzer import extract_job_requirements
from relevance_scorer import calculate_relevance_scores, rank_resumes
from utils import allowed_file, display_resume_feedback
from database import get_all_resumes, get_resume_by_id

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
                # Extract job title from the description (first line or default)
                job_title = ""
                if job_description:
                    lines = job_description.strip().split('\n')
                    if lines:
                        job_title = lines[0][:50]  # First line, truncated to 50 chars
                
                # Calculate relevance scores using AI
                relevance_scores = calculate_relevance_scores(
                    st.session_state.parsed_resumes, 
                    st.session_state.job_requirements,
                    job_title=job_title,
                    job_description=job_description
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
                
                # Get parsed data (either from AI or traditional)
                parsed_data = resume.get("parsed_data", {})
                resume_id = resume.get("resume_id")
                
                # Display resume content in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### Contact Information")
                    contact_info = parsed_data.get("contact_info", {})
                    if contact_info:
                        st.write(f"**Name:** {contact_info.get('name', 'Not available')}")
                        st.write(f"**Email:** {contact_info.get('email', 'Not available')}")
                        st.write(f"**Phone:** {contact_info.get('phone', 'Not available')}")
                        st.write(f"**Location:** {contact_info.get('location', 'Not available')}")
                    else:
                        st.write(resume.get("contact_info", "Not available"))
                    
                    st.markdown("##### Skills")
                    skills = parsed_data.get("skills", {})
                    if isinstance(skills, dict) and (skills.get("technical") or skills.get("soft")):
                        if skills.get("technical"):
                            st.write("**Technical Skills:**")
                            st.write(", ".join(skills.get("technical", [])))
                        if skills.get("soft"):
                            st.write("**Soft Skills:**")
                            st.write(", ".join(skills.get("soft", [])))
                    elif resume.get("skills"):
                        # Fallback to old format
                        skills_list = resume.get("skills", [])
                        st.write(", ".join(skills_list))
                    else:
                        st.info("No skills identified")
                
                with col2:
                    st.markdown("##### Education")
                    education = parsed_data.get("education", [])
                    if education and isinstance(education, list):
                        for edu in education:
                            if isinstance(edu, dict):
                                st.write(f"**Degree:** {edu.get('degree', 'N/A')}")
                                st.write(f"**Institution:** {edu.get('institution', 'N/A')}")
                                if edu.get('graduation_date'):
                                    st.write(f"**Graduation Date:** {edu.get('graduation_date')}")
                                if edu.get('gpa'):
                                    st.write(f"**GPA:** {edu.get('gpa')}")
                                st.write("---")
                            else:
                                st.write(edu)
                    elif resume.get("education"):
                        # Fallback to old format
                        for edu in resume.get("education", []):
                            st.write(edu)
                    else:
                        st.info("No education information identified")
                    
                    st.markdown("##### Experience")
                    experience = parsed_data.get("experience", [])
                    if experience and isinstance(experience, list):
                        for exp in experience:
                            if isinstance(exp, dict):
                                st.write(f"**Title:** {exp.get('title', 'N/A')}")
                                st.write(f"**Company:** {exp.get('company', 'N/A')}")
                                if exp.get('start_date') and exp.get('end_date'):
                                    st.write(f"**Period:** {exp.get('start_date')} to {exp.get('end_date')}")
                                if exp.get('description'):
                                    st.write(f"**Description:** {exp.get('description')}")
                                if exp.get('achievements'):
                                    st.write("**Achievements:**")
                                    for achievement in exp.get('achievements', []):
                                        st.write(f"- {achievement}")
                                st.write("---")
                            else:
                                st.write(exp)
                    elif resume.get("experience"):
                        # Fallback to old format
                        for exp in resume.get("experience", []):
                            st.write(exp)
                    else:
                        st.info("No experience information identified")
                
                # Display match analysis and feedback
                st.markdown("##### Match Analysis")
                
                match_analysis = resume.get("match_analysis", {})
                if match_analysis:
                    scores = match_analysis.get("scores", {})
                    
                    # Create columns for different score types
                    score_col1, score_col2, score_col3, score_col4 = st.columns(4)
                    
                    with score_col1:
                        st.metric("Overall", f"{scores.get('overall_match', 0)}%")
                    with score_col2:
                        st.metric("Skills", f"{scores.get('skills_match', 0)}%")
                    with score_col3:
                        st.metric("Experience", f"{scores.get('experience_match', 0)}%")
                    with score_col4:
                        st.metric("Education", f"{scores.get('education_match', 0)}%")
                    
                    # Display matching and missing skills
                    matching_skills = match_analysis.get("matching_skills", [])
                    missing_skills = match_analysis.get("missing_skills", [])
                    
                    skill_col1, skill_col2 = st.columns(2)
                    
                    with skill_col1:
                        st.write("**Matching Skills:**")
                        if matching_skills:
                            st.write(", ".join(matching_skills))
                        else:
                            st.info("No matching skills found")
                    
                    with skill_col2:
                        st.write("**Missing Skills:**")
                        if missing_skills:
                            st.write(", ".join(missing_skills))
                        else:
                            st.info("No missing skills")
                    
                    # Display recommendations
                    st.write("**Recommendations:**")
                    recommendations = match_analysis.get("recommendations", [])
                    if recommendations:
                        for rec in recommendations:
                            st.write(f"- {rec}")
                    
                    # Display summary
                    st.write("**Summary:**")
                    st.write(match_analysis.get("feedback_summary", "No summary available"))
                    
                    # Display DB ID for reference if available
                    if resume_id:
                        st.caption(f"Resume ID: {resume_id}")
                else:
                    # Fallback to simple feedback
                    display_resume_feedback(resume, st.session_state.job_requirements)
    
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
