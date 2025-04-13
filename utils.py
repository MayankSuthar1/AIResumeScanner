import streamlit as st

def allowed_file(filename):
    """
    Check if the uploaded file has an allowed extension
    """
    ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_chat_resume_feedback(resume, job_requirements):
    """
    Generate chat-friendly feedback about skill gaps and matches between resume and job requirements.
    Returns a dictionary with structured feedback data for chat UI.
    """
    feedback = {
        "overall_match": 0,
        "matching_skills": [],
        "missing_skills": [],
        "recommendations": [],
        "feedback_summary": ""
    }
    
    # Check if we have the AI-based match analysis already
    if resume.get("match_analysis"):
        match_analysis = resume.get("match_analysis")
        
        # Get scores
        scores = match_analysis.get("scores", {})
        match_score = scores.get("overall_match", 0)
        feedback["overall_match"] = match_score
        
        # Get matching skills
        feedback["matching_skills"] = match_analysis.get("matching_skills", [])
        
        # Get missing skills
        feedback["missing_skills"] = match_analysis.get("missing_skills", [])
        
        # Get recommendations
        feedback["recommendations"] = match_analysis.get("recommendations", [])
        
        # Generate summary based on match score
        if match_score >= 80:
            feedback["feedback_summary"] = f"Strong Match ({match_score}%): Your profile aligns very well with this position."
        elif match_score >= 60:
            feedback["feedback_summary"] = f"Good Match ({match_score}%): Your profile aligns well with this position with some areas for improvement."
        elif match_score >= 40:
            feedback["feedback_summary"] = f"Moderate Match ({match_score}%): Consider addressing the skill gaps to improve your chances."
        else:
            feedback["feedback_summary"] = f"Low Match ({match_score}%): Your current profile may not be the best fit for this specific position."
        
        return feedback
    
    # Fallback to traditional analysis if no AI analysis is available
    
    # Extract job requirements skills
    job_skills = set()
    if isinstance(job_requirements.get("skills"), dict):
        # New AI format with technical and soft skills
        tech_skills = job_requirements.get("skills", {}).get("technical", [])
        soft_skills = job_requirements.get("skills", {}).get("soft", [])
        job_skills = set([s.lower() for s in tech_skills + soft_skills])
    else:
        # Old format with just a list of skills
        job_skills = set([s.lower() for s in job_requirements.get("skills", [])])
    
    # Extract resume skills
    resume_skills = set()
    parsed_data = resume.get("parsed_data", {})
    if isinstance(parsed_data.get("skills"), dict):
        # New AI format
        tech_skills = parsed_data.get("skills", {}).get("technical", [])
        soft_skills = parsed_data.get("skills", {}).get("soft", [])
        resume_skills = set([s.lower() for s in tech_skills + soft_skills])
    elif resume.get("skills"):
        # Old format
        resume_skills = set([skill.lower() for skill in resume.get("skills", [])])
    
    # Find matching skills
    matching_skills = resume_skills.intersection(job_skills)
    feedback["matching_skills"] = list(matching_skills)
    
    # Find missing skills
    missing_skills = job_skills - resume_skills
    feedback["missing_skills"] = list(missing_skills)
    
    # Provide recommendations
    recommendations = []
    
    if len(missing_skills) > 3:
        # If many skills are missing, suggest focusing on the most important ones
        recommendations.append("Consider focusing on developing the following key skills: " + 
                               ", ".join(list(missing_skills)[:3]))
    else:
        # For each missing skill, provide a specific recommendation
        for skill in missing_skills:
            recommendations.append(f"Add experience or certification related to {skill}")
    
    # Add a general recommendation
    if len(missing_skills) / len(job_skills) > 0.5 if job_skills else 0:
        recommendations.append("Consider tailoring your resume more specifically to this job description")
    
    feedback["recommendations"] = recommendations
    
    # Calculate and set match score
    match_score = resume.get("match_score", 0)
    feedback["overall_match"] = match_score
    
    # Generate summary based on match score
    if match_score >= 80:
        feedback["feedback_summary"] = f"Strong Match ({match_score:.1f}%): Your profile aligns very well with this position."
    elif match_score >= 60:
        feedback["feedback_summary"] = f"Good Match ({match_score:.1f}%): Your profile aligns well with this position with some areas for improvement."
    elif match_score >= 40:
        feedback["feedback_summary"] = f"Moderate Match ({match_score:.1f}%): Consider addressing the skill gaps to improve your chances."
    else:
        feedback["feedback_summary"] = f"Low Match ({match_score:.1f}%): Your current profile may not be the best fit for this specific position."
    
    return feedback

def display_resume_feedback(resume, job_requirements):
    """
    Display feedback about skill gaps and matches between resume and job requirements
    """
    # Check if we have the AI-based match analysis already
    if resume.get("match_analysis"):
        match_analysis = resume.get("match_analysis")
        
        # Display overall score
        scores = match_analysis.get("scores", {})
        st.markdown("##### Overall Assessment")
        match_score = scores.get("overall_match", 0)
        
        if match_score >= 80:
            st.success(f"Strong Match ({match_score}%): Your profile aligns very well with this position.")
        elif match_score >= 60:
            st.info(f"Good Match ({match_score}%): Your profile aligns well with this position with some areas for improvement.")
        elif match_score >= 40:
            st.warning(f"Moderate Match ({match_score}%): Consider addressing the skill gaps to improve your chances.")
        else:
            st.error(f"Low Match ({match_score}%): Your current profile may not be the best fit for this specific position.")
        
        # Create tabs for different match categories
        skill_tab, exp_tab, edu_tab = st.tabs(["Skills Match", "Experience Match", "Education Match"])
        
        # SKILLS TAB
        with skill_tab:
            # Display skills match score
            skill_score = scores.get("skills_match", 0)
            st.metric("Skills Match", f"{skill_score}%")
            
            # Display matching skills
            matching_skills = match_analysis.get("matching_skills", [])
            if matching_skills:
                st.success(f"Matching Skills ({len(matching_skills)}): {', '.join(matching_skills)}")
            else:
                st.warning("No direct skill matches found with job requirements.")
            
            # Display missing skills
            missing_skills = match_analysis.get("missing_skills", [])
            if missing_skills:
                st.warning(f"Missing Skills ({len(missing_skills)}): {', '.join(missing_skills)}")
        
        # EXPERIENCE TAB
        with exp_tab:
            # Display experience match score
            exp_score = scores.get("experience_match", 0)
            st.metric("Experience Match", f"{exp_score}%")
            
            # Display matching experience
            matching_experience = match_analysis.get("matching_experience", [])
            if matching_experience:
                st.success("Matching Experience:")
                for item in matching_experience:
                    st.success(f"✓ {item}")
            else:
                st.info("No specific experience matches identified.")
            
            # Display missing experience
            missing_experience = match_analysis.get("missing_experience", [])
            if missing_experience:
                st.warning("Missing Experience:")
                for item in missing_experience:
                    st.warning(f"⚠ {item}")
        
        # EDUCATION TAB
        with edu_tab:
            # Display education match score
            edu_score = scores.get("education_match", 0)
            st.metric("Education Match", f"{edu_score}%")
            
            # Display matching education
            matching_education = match_analysis.get("matching_education", [])
            if matching_education:
                st.success("Matching Education:")
                for item in matching_education:
                    st.success(f"✓ {item}")
            else:
                st.info("No specific education matches identified.")
            
            # Display missing education
            missing_education = match_analysis.get("missing_education", [])
            if missing_education:
                st.warning("Missing Education:")
                for item in missing_education:
                    st.warning(f"⚠ {item}")
        
        # Display recommendations
        recommendations = match_analysis.get("recommendations", [])
        if recommendations:
            st.markdown("##### Recommendations")
            for i, recommendation in enumerate(recommendations):
                st.markdown(f"{i+1}. {recommendation}")
        
        return
    
    # Fallback to traditional analysis if no AI analysis is available
    
    # Extract job requirements skills
    job_skills = set()
    if isinstance(job_requirements.get("skills"), dict):
        # New AI format with technical and soft skills
        tech_skills = job_requirements.get("skills", {}).get("technical", [])
        soft_skills = job_requirements.get("skills", {}).get("soft", [])
        job_skills = set([s.lower() for s in tech_skills + soft_skills])
    else:
        # Old format with just a list of skills
        job_skills = set([s.lower() for s in job_requirements.get("skills", [])])
    
    # Extract resume skills
    resume_skills = set()
    parsed_data = resume.get("parsed_data", {})
    if isinstance(parsed_data.get("skills"), dict):
        # New AI format
        tech_skills = parsed_data.get("skills", {}).get("technical", [])
        soft_skills = parsed_data.get("skills", {}).get("soft", [])
        resume_skills = set([s.lower() for s in tech_skills + soft_skills])
    elif resume.get("skills"):
        # Old format
        resume_skills = set([skill.lower() for skill in resume.get("skills", [])])
    
    # Find matching skills
    matching_skills = resume_skills.intersection(job_skills)
    
    # Find missing skills
    missing_skills = job_skills - resume_skills
    
    # Display matching skills
    if matching_skills:
        st.success(f"Matching Skills ({len(matching_skills)}): {', '.join(matching_skills)}")
    else:
        st.warning("No direct skill matches found with job requirements.")
    
    # Display missing skills
    if missing_skills:
        st.warning(f"Missing Skills ({len(missing_skills)}): {', '.join(missing_skills)}")
        
        # Provide recommendations
        st.markdown("##### Recommendations")
        recommendations = []
        
        if len(missing_skills) > 3:
            # If many skills are missing, suggest focusing on the most important ones
            recommendations.append("Consider focusing on developing the following key skills: " + 
                                  ", ".join(list(missing_skills)[:3]))
        else:
            # For each missing skill, provide a specific recommendation
            for skill in missing_skills:
                recommendations.append(f"Add experience or certification related to {skill}")
        
        # Add a general recommendation
        if len(missing_skills) / len(job_skills) > 0.5 if job_skills else 0:
            recommendations.append("Consider tailoring your resume more specifically to this job description")
        
        for i, recommendation in enumerate(recommendations):
            st.markdown(f"{i+1}. {recommendation}")
    else:
        st.success("Your resume includes all the required skills for this position.")
    
    # Overall feedback based on match score
    st.markdown("##### Overall Assessment")
    match_score = resume.get("match_score", 0)
    
    if match_score >= 80:
        st.success(f"Strong Match ({match_score:.1f}%): Your profile aligns very well with this position.")
    elif match_score >= 60:
        st.info(f"Good Match ({match_score:.1f}%): Your profile aligns well with this position with some areas for improvement.")
    elif match_score >= 40:
        st.warning(f"Moderate Match ({match_score:.1f}%): Consider addressing the skill gaps to improve your chances.")
    else:
        st.error(f"Low Match ({match_score:.1f}%): Your current profile may not be the best fit for this specific position.")
