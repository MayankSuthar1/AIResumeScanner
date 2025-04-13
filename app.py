import streamlit as st
import pandas as pd
import io
import os
import tempfile
from resume_parser import parse_resume
from job_analyzer import extract_job_requirements
from relevance_scorer import calculate_relevance_scores, rank_resumes
from utils import allowed_file, display_resume_feedback

# Use in-memory storage instead of database
if 'stored_resumes' not in st.session_state:
    st.session_state.stored_resumes = []

# Page config
st.set_page_config(
    page_title="AI-Powered Resume Scanner",
    page_icon="📄",
    layout="wide"
)

# Application title, description and navigation buttons
col1 = st.columns([1])[0]
with col1:
    st.title("AI-Powered Resume Scanner")

# Sidebar navigation
with st.sidebar:
    st.header("AI Resume Scanner")
    st.markdown("---")
    
    if st.button("Start Over", key="start_over_btn", use_container_width=True):
        # Reset session state
        st.session_state.uploaded_resumes = []
        st.session_state.job_description = ""
        st.session_state.parsed_resumes = []
        st.session_state.job_requirements = {}
        st.session_state.ranked_resumes = []
        st.session_state.processing_complete = False
        st.session_state.processing_started = False
        st.session_state.resume_parsed = False
        st.session_state.messages = []
        st.session_state.user_name = ""
        st.session_state.ui_state = "upload_resume"
        st.rerun()
    
    if st.session_state.get('resume_parsed', False):
        if st.button("New Resume", key="new_resume_btn", use_container_width=True):
            # Reset to upload new resume
            st.session_state.ui_state = "upload_resume"
            st.session_state.resume_parsed = False
            st.session_state.parsed_resumes = []
            st.session_state.job_requirements = {}
            st.session_state.job_description = ""
            st.session_state.ranked_resumes = []
            st.session_state.messages = []
            st.rerun()
    
    if st.session_state.get('resume_parsed', False):
        if st.button("New Job", key="new_job_btn", use_container_width=True):
            # Reset to enter new job description
            st.session_state.ui_state = "job_description"
            st.session_state.job_requirements = {}
            st.session_state.job_description = ""
            st.session_state.ranked_resumes = []
            st.session_state.messages = []
            st.rerun()

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
if 'show_database_view' not in st.session_state:
    st.session_state.show_database_view = False
# Chat related session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'resume_parsed' not in st.session_state:
    st.session_state.resume_parsed = False
if 'user_name' not in st.session_state:
    st.session_state.user_name = ""

# Main UI with Chat Interface
# Define the steps of the UI flow
if 'ui_state' not in st.session_state:
    st.session_state.ui_state = "upload_resume"

# Step 1: Upload Resume Section
if st.session_state.ui_state == "upload_resume":
    st.header("Upload Your Resume")
    
    uploaded_file = st.file_uploader(
        "Upload your resume (PDF, DOCX, DOC)", 
        accept_multiple_files=False,
        type=["pdf", "docx", "doc"]
    )
    
    if uploaded_file and allowed_file(uploaded_file.name):
        if st.button("Process Resume"):
            with st.spinner("Parsing your resume..."):
                # Create a temporary file to store the uploaded file
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Parse the resume
                parsed_resume = parse_resume(tmp_path, uploaded_file.name)
                st.session_state.parsed_resumes = [parsed_resume]
                
                # Delete the temporary file
                os.unlink(tmp_path)
                
                # Get user name from the parsed resume
                contact_info = parsed_resume.get("parsed_data", {}).get("contact_info", {})
                user_name = contact_info.get("name", "")
                if not user_name and isinstance(parsed_resume.get("contact_info"), str):
                    user_name = parsed_resume.get("contact_info", "").split('\n')[0].strip()
                
                # Store the user name in session state
                st.session_state.user_name = user_name if user_name else "there"
                
                # Store the uploaded resume in session state
                st.session_state.uploaded_resumes = [uploaded_file]
                
                # Store in the historical resumes list (in-memory storage)
                st.session_state.stored_resumes.append(parsed_resume)
                
                # Move to the next UI state
                st.session_state.ui_state = "job_description"
                st.session_state.resume_parsed = True
                
                st.rerun()

# Step 2: Job Description Input
elif st.session_state.ui_state == "job_description":
    st.header("Enter Job Description")
    
    # Show resume summary
    with st.expander("Resume Summary", expanded=False):
        resume = st.session_state.parsed_resumes[0]
        parsed_data = resume.get("parsed_data", {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Contact Information")
            contact_info = parsed_data.get("contact_info", {})
            if contact_info.get("name"):
                st.write(f"**Name:** {contact_info.get('name')}")
            if contact_info.get("email"):
                st.write(f"**Email:** {contact_info.get('email')}")
                
            st.markdown("##### Skills")
            skills = parsed_data.get("skills", {})
            if isinstance(skills, dict):
                if skills.get("technical"):
                    st.write("**Technical Skills:**")
                    st.write(", ".join(skills.get("technical", [])[:5]) + ("..." if len(skills.get("technical", [])) > 5 else ""))
                if skills.get("soft"):
                    st.write("**Soft Skills:**")
                    st.write(", ".join(skills.get("soft", [])[:5]) + ("..." if len(skills.get("soft", [])) > 5 else ""))
    
    # Job description text area
    job_description = st.text_area(
        "Enter the job description below:", 
        height=300,
        placeholder="Paste the full job description here..."
    )
    
    # Create columns for the analyze button and processing status
    col1, col2 = st.columns([1, 3])
    
    # Process job description button
    analyze_button = col1.button("Analyze Match", type="primary", use_container_width=True)
    
    # Store the analyze status in session state
    if 'analyze_clicked' not in st.session_state:
        st.session_state.analyze_clicked = False
    
    if analyze_button:
        st.session_state.analyze_clicked = True
    
    # Show processing animation when button is clicked
    if st.session_state.analyze_clicked and job_description:
        with col2:
            with st.spinner("Analyzing job description and calculating match..."):
                # Store job description
                st.session_state.job_description = job_description
                
                # Extract job requirements
                job_requirements = extract_job_requirements(job_description)
                st.session_state.job_requirements = job_requirements
                
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
                
                # Store job match in the in-memory storage with the resume
                if ranked_resumes and len(st.session_state.stored_resumes) > 0:
                    resume = ranked_resumes[0]
                    match_analysis = resume.get("match_analysis", {})
                    
                    # Find the last added resume and add job match info
                    last_resume = st.session_state.stored_resumes[-1]
                    if "job_matches" not in last_resume:
                        last_resume["job_matches"] = []
                    
                    # Add the match details
                    job_match = {
                        "job_title": job_title,
                        "job_description": job_description,
                        "skills_match_score": match_analysis.get("scores", {}).get("skills_match", 0),
                        "experience_match_score": match_analysis.get("scores", {}).get("experience_match", 0),
                        "education_match_score": match_analysis.get("scores", {}).get("education_match", 0),
                        "overall_match_score": match_analysis.get("scores", {}).get("overall_match", 0),
                        "matching_skills": match_analysis.get("matching_skills", []),
                        "missing_skills": match_analysis.get("missing_skills", []),
                        "recommendations": match_analysis.get("recommendations", []),
                        "feedback_summary": match_analysis.get("feedback_summary", "")
                    }
                    
                    last_resume["job_matches"].append(job_match)
                
                # Generate initial response message
                if ranked_resumes:
                    resume = ranked_resumes[0]
                    match_score = resume.get("match_score", 0)
                    match_analysis = resume.get("match_analysis", {})
                    
                    # Create response message
                    response = f"### Resume Analysis for {job_title if job_title else 'this position'}\n\n"
                    
                    # Overall match
                    response += f"**Overall Match:** {match_score:.1f}%\n\n"
                    
                    # Skills match
                    skills_match = match_analysis.get("scores", {}).get("skills_match", 0)
                    response += f"**Skills Match:** {skills_match}%\n\n"
                    
                    # Experience match
                    experience_match = match_analysis.get("scores", {}).get("experience_match", 0) 
                    response += f"**Experience Match:** {experience_match}%\n\n"
                    
                    # Education match
                    education_match = match_analysis.get("scores", {}).get("education_match", 0)
                    response += f"**Education Match:** {education_match}%\n\n"
                    
                    # Required experience
                    job_experience = st.session_state.job_requirements.get("experience", [])
                    if job_experience:
                        response += "**Required Experience:**\n"
                        for exp in job_experience:
                            if isinstance(exp, dict):
                                years = exp.get("years", "")
                                domain = exp.get("domain", "")
                                if years or domain:
                                    response += f"- {years}{' in ' if domain else ''}{domain}\n"
                        response += "\n"
                    
                    # Required education
                    job_education = st.session_state.job_requirements.get("education", [])
                    if job_education:
                        response += "**Required Education:**\n"
                        for edu in job_education:
                            if isinstance(edu, dict):
                                degree = edu.get("degree", "")
                                field = edu.get("field", "")
                                if degree or field:
                                    response += f"- {degree}{' in ' if field else ''}{field}\n"
                        response += "\n"
                    
                    # Matching skills
                    matching_skills = match_analysis.get("matching_skills", [])
                    if matching_skills:
                        response += "**Matching Skills:**\n"
                        response += ", ".join(matching_skills)  # Show all matching skills
                        response += "\n\n"
                    
                    # Missing skills - show ALL of them
                    missing_skills = match_analysis.get("missing_skills", [])
                    if missing_skills:
                        response += "**Missing Skills:**\n"
                        response += ", ".join(missing_skills)  # Show ALL missing skills
                        response += "\n\n"
                    
                    # Missing experience
                    missing_experience = match_analysis.get("missing_experience", [])
                    if missing_experience:
                        response += "**Missing Experience:**\n"
                        for exp in missing_experience:
                            response += f"- {exp}\n"
                        response += "\n"
                    
                    # Missing education
                    missing_education = match_analysis.get("missing_education", [])
                    if missing_education:
                        response += "**Missing Education:**\n"
                        for edu in missing_education:
                            response += f"- {edu}\n"
                        response += "\n"
                    
                    # Recommendations - show ALL of them
                    recommendations = match_analysis.get("recommendations", [])
                    if recommendations:
                        response += "**Recommendations:**\n"
                        for rec in recommendations:  # Show ALL recommendations
                            response += f"- {rec}\n"
                        response += "\n"
                    
                    # Summary
                    feedback_summary = match_analysis.get("feedback_summary", "")
                    if feedback_summary:
                        response += f"**Summary:** {feedback_summary}\n\n"
                    
                    # Add suggested follow-up questions
                    response += "You can ask me specific questions about this match, such as:\n"
                    response += "- How can I improve my education match?\n"
                    response += "- What skills should I prioritize learning?\n"
                    response += "- Why is my experience score low?\n"
                    response += "- What can I add to my resume to better match this job?"
                else:
                    response = "I couldn't analyze your resume against this job description. Please try again with a more detailed job description."
                
                # Store initial bot response
                greeting = f"Hi {st.session_state.user_name}! 👋" if st.session_state.user_name != "there" else "Hi there! 👋"
                st.session_state.messages = [{"role": "assistant", "content": f"{greeting} Here's my analysis of your resume match:\n\n{response}"}]
                
                # Reset the analyze_clicked state
                st.session_state.analyze_clicked = False
                
                # Move to chat UI
                st.session_state.ui_state = "chat"
                st.rerun()

# Step 3: Chat UI
elif st.session_state.ui_state == "chat":
    # st.header("Resume Match Analysis & Chat")
    
    # # Show match score
    # if st.session_state.ranked_resumes:
    #     resume = st.session_state.ranked_resumes[0]
    #     match_score = resume.get("match_score", 0)
    #     st.metric("Overall Match", f"{match_score:.1f}%")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Get user input
    if prompt := st.chat_input(f"Ask a question about your resume match..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Process user question
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your question..."):
                # This is a follow-up question about the job match analysis
                resume = st.session_state.ranked_resumes[0]
                resume_info = resume.get("parsed_data", {})
                job_info = st.session_state.job_requirements
                match_analysis = resume.get("match_analysis", {})
                
                # Use AI to answer the question
                from ai_helper import answer_chat_question
                answer = answer_chat_question(prompt, resume_info, job_info, match_analysis)
                
                # Add response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                # Display the response
                st.write(answer)
