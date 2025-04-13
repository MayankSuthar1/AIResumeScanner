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
        # Reset session state but preserve API connection info
        api_key = st.session_state.api_key
        api_connected = st.session_state.api_connected
        
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
        
        # Preserve API connection state
        st.session_state.api_key = api_key
        st.session_state.api_connected = api_connected
        
        # Go directly to upload resume screen if API is connected
        st.session_state.ui_state = "upload_resume" if api_connected else "api_key_input"
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
# API key related session state
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'api_connected' not in st.session_state:
    st.session_state.api_connected = False

# Main UI with Chat Interface
# Define the steps of the UI flow
if 'ui_state' not in st.session_state:
    st.session_state.ui_state = "api_key_input" if not st.session_state.get('api_connected', False) else "upload_resume"

# Step 0: API Key Input
if st.session_state.ui_state == "api_key_input":
    st.header("Welcome to AI-Powered Resume Scanner")
    
    # Check if there's an API key in the environment variables
    env_api_key = os.environ.get("GOOGLE_API_KEY")
    if env_api_key:
        st.success("Found Gemini API key in environment variables.")
        st.session_state.api_key = env_api_key
        st.session_state.api_connected = True
        st.session_state.ui_state = "upload_resume"
        st.rerun()
    
    st.write("To use the AI-powered features, please provide your Google Gemini API key.")
    
    with st.expander("How to get a Gemini API key", expanded=False):
        st.write("""
        1. Go to [Google AI Studio](https://ai.google.dev/)
        2. Sign in with your Google account
        3. Navigate to 'API keys' in your account settings
        4. Create a new API key
        5. Copy the API key and paste it below
        """)
    
    # Create a form for API key input
    with st.form("api_key_form"):
        api_key = st.text_input("Enter your Gemini API key:", type="password")
        submitted = st.form_submit_button("Connect", type="primary", use_container_width=True)
        
        if submitted and api_key:
            with st.spinner("Connecting to Gemini API..."):
                # Test the API key
                try:
                    from ai_helper import configure_gemini_api
                    configure_gemini_api(api_key)
                    # Save the working API key to session state
                    st.session_state.api_key = api_key
                    st.session_state.api_connected = True
                    st.success("Successfully connected to Gemini API!")
                    st.session_state.ui_state = "upload_resume"
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to connect to Gemini API: {e}")
                    st.session_state.api_connected = False
    
    # Additional information about the app
    st.markdown("---")
    st.write("""
    ### About this Application
    
    This AI-powered resume scanner helps you analyze how well your resume matches specific job descriptions. 
    The application will:
    
    - Extract information from your resume
    - Analyze the requirements from job descriptions
    - Calculate match scores for skills, experience, and education
    - Provide recommendations to improve your match
    
    Your API key is only stored temporarily in your session and is not saved permanently.
    """)

# Step 1: Upload Resume Section
elif st.session_state.ui_state == "upload_resume":
    st.header("Upload Your Resume")
    
    # Initialize processing state in session state if not exists
    if 'resume_processing' not in st.session_state:
        st.session_state.resume_processing = False
    
    # Only show file uploader if not in processing state
    if not st.session_state.resume_processing:
        uploaded_file = st.file_uploader(
            "Upload your resume (PDF, DOCX, DOC)", 
            accept_multiple_files=False,
            type=["pdf", "docx", "doc"]
        )
        
        if uploaded_file and allowed_file(uploaded_file.name):
            # Store uploaded file in session state so we can access it during processing
            st.session_state._uploaded_file = uploaded_file
            
            if st.button("Process Resume"):
                # Set processing state to true
                st.session_state.resume_processing = True
                st.rerun()  # Rerun to update UI
    else:
        # Show disabled file uploader during processing
        st.file_uploader(
            "Upload your resume (PDF, DOCX, DOC)",
            accept_multiple_files=False,
            type=["pdf", "docx", "doc"],
            disabled=True
        )
    
    # Process the resume if in processing state
    if st.session_state.resume_processing:
        with st.spinner("Parsing your resume..."):
            # Get the uploaded file from session state
            uploaded_file = st.session_state.get('_uploaded_file')
            
            if uploaded_file and allowed_file(uploaded_file.name):
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
                
                # Reset processing state
                st.session_state.resume_processing = False
                
                # Move to the next UI state
                st.session_state.ui_state = "job_description"
                st.session_state.resume_parsed = True
                
                st.rerun()
            else:
                st.error("No resume file found. Please upload a valid resume file.")
                st.session_state.resume_processing = False
                st.rerun()

# Step 2: Job Description Input
elif st.session_state.ui_state == "job_description":
    st.header("Enter Job Description")
    
    # Initialize job processing state if not exists
    if 'job_processing' not in st.session_state:
        st.session_state.job_processing = False
        
    # Store job description in session state if entered
    if 'current_job_description' not in st.session_state:
        st.session_state.current_job_description = ""
    
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
    
    # Job description text area - enabled when not processing
    if not st.session_state.job_processing:
        job_description = st.text_area(
            "Enter the job description below:", 
            height=300,
            placeholder="Paste the full job description here...",
            value=st.session_state.current_job_description
        )
        # Store the current job description
        if job_description:
            st.session_state.current_job_description = job_description
    else:
        # Show disabled text area during processing
        st.text_area(
            "Enter the job description below:", 
            height=300,
            value=st.session_state.current_job_description,
            disabled=True
        )
        job_description = st.session_state.current_job_description
    
    # Create columns for the analyze button and processing status
    col1, col2 = st.columns([1, 3])
    
    # Process job description button - only show when not processing
    if not st.session_state.job_processing:
        analyze_button = col1.button("Analyze Match", type="primary", use_container_width=True)
        if analyze_button and job_description:
            st.session_state.job_processing = True
            st.rerun()  # Rerun to update UI
    else:
        # Disabled button during processing
        col1.button("Analyzing...", disabled=True, use_container_width=True)
    
    # Process job if in processing state
    if st.session_state.job_processing and job_description:
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
                
                # Reset the processing state
                st.session_state.job_processing = False
                
                # Move to chat UI
                st.session_state.ui_state = "chat"
                st.rerun()

# Step 3: Chat UI
elif st.session_state.ui_state == "chat":
    # Create columns for the match score and view resume details button
    col1, col2 = st.columns([1, 1])
    
    # Show match score
    if st.session_state.ranked_resumes:
        resume = st.session_state.ranked_resumes[0]
        match_score = resume.get("match_score", 0)
        col1.metric("Overall Match", f"{match_score:.1f}%")
    
    # Show resume details in an expander
    with st.expander("View Extracted Resume Details", expanded=False):
        if st.session_state.ranked_resumes:
            resume = st.session_state.ranked_resumes[0]
            parsed_data = resume.get("parsed_data", {})
            
            # Contact Information
            st.markdown("#### Contact Information")
            contact_info = parsed_data.get("contact_info", {})
            if contact_info:
                for key, value in contact_info.items():
                    if value:
                        st.write(f"**{key.capitalize()}:** {value}")
            
            # Education
            st.markdown("#### Education")
            education = parsed_data.get("education", [])
            if education:
                for edu in education:
                    if isinstance(edu, dict):
                        edu_parts = []
                        if edu.get("degree"):
                            edu_parts.append(f"**Degree:** {edu.get('degree')}")
                        if edu.get("institution"):
                            edu_parts.append(f"**Institution:** {edu.get('institution')}")
                        if edu.get("graduation_date"):
                            edu_parts.append(f"**Graduation Date:** {edu.get('graduation_date')}")
                        if edu_parts:
                            st.markdown(" | ".join(edu_parts))
                            st.markdown("---")
            else:
                st.write("No education information extracted")
            
            # Experience
            st.markdown("#### Work Experience")
            experience = parsed_data.get("experience", [])
            if experience:
                for exp in experience:
                    if isinstance(exp, dict):
                        if exp.get("company") or exp.get("title"):
                            st.markdown(f"**{exp.get('title', '')}** at **{exp.get('company', '')}**")
                        
                        date_range = []
                        if exp.get("start_date"):
                            date_range.append(exp.get("start_date"))
                        if exp.get("end_date"):
                            date_range.append(exp.get("end_date"))
                        
                        if date_range:
                            st.write(f"*{' - '.join(date_range)}*")
                        
                        if exp.get("description"):
                            st.write(exp.get("description"))
                        
                        if exp.get("achievements"):
                            st.markdown("**Achievements:**")
                            for achievement in exp.get("achievements"):
                                st.markdown(f"- {achievement}")
                        
                        st.markdown("---")
            else:
                st.write("No experience information extracted")
            
            # Skills
            st.markdown("#### Skills")
            skills = parsed_data.get("skills", {})
            
            # Technical skills
            if isinstance(skills, dict) and skills.get("technical"):
                st.markdown("**Technical Skills:**")
                st.write(", ".join(skills.get("technical")))
            
            # Soft skills
            if isinstance(skills, dict) and skills.get("soft"):
                st.markdown("**Soft Skills:**")
                st.write(", ".join(skills.get("soft")))
            
            if not isinstance(skills, dict) or (not skills.get("technical") and not skills.get("soft")):
                st.write("No skills information extracted")
            
            # Certifications
            certifications = parsed_data.get("certifications", [])
            if certifications:
                st.markdown("#### Certifications")
                for cert in certifications:
                    st.markdown(f"- {cert}")
    
    # Add horizontal separator
    st.markdown("---")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Variable to track if we're in processing state
    is_processing = st.session_state.get("is_processing", False)
    
    # Get user input (disable when processing)
    if not is_processing:
        prompt = st.chat_input(f"Ask a question about your resume match...")
        
        if prompt:
            # Set processing state to true
            st.session_state.is_processing = True
            
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.rerun()  # Rerun to disable the input
    else:
        # Just show the disabled input placeholder during processing
        st.chat_input(f"Processing your question...", disabled=True)
    
    # Process user question if we have a new prompt
    if st.session_state.get("is_processing", False) and not st.session_state.get("processing_complete", False):
        # Get the last user message
        last_user_message = next((msg["content"] for msg in reversed(st.session_state.messages) 
                                if msg["role"] == "user"), None)
        
        if last_user_message:
            # Display user message
            with st.chat_message("user"):
                st.write(last_user_message)
            
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
                    answer = answer_chat_question(last_user_message, resume_info, job_info, match_analysis)
                    
                    # Add response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    # Display the response
                    st.write(answer)
                    
                    # Reset processing states
                    st.session_state.is_processing = False
                    st.session_state.processing_complete = False
                    st.rerun()  # Rerun to re-enable the input
