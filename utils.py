import streamlit as st

def allowed_file(filename):
    """
    Check if the uploaded file has an allowed extension
    """
    ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def display_resume_feedback(resume, job_requirements):
    """
    Display feedback about skill gaps and matches between resume and job requirements
    """
    # Extract requirements
    required_skills = set(job_requirements.get("skills", []))
    
    # Extract resume skills
    resume_skills = set([skill.lower() for skill in resume.get("skills", [])])
    
    # Find matching skills
    matching_skills = resume_skills.intersection(required_skills)
    
    # Find missing skills
    missing_skills = required_skills - resume_skills
    
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
        if len(missing_skills) / len(required_skills) > 0.5:
            recommendations.append("Consider tailoring your resume more specifically to this job description")
        
        for i, recommendation in enumerate(recommendations):
            st.markdown(f"{i+1}. {recommendation}")
    else:
        st.success("Your resume includes all the required skills for this position.")
    
    # Overall feedback based on match score
    st.markdown("##### Overall Assessment")
    match_score = resume.get("match_score", 0)
    
    if match_score >= 80:
        st.success("Strong Match: Your profile aligns very well with this position.")
    elif match_score >= 60:
        st.info("Good Match: Your profile aligns well with this position with some areas for improvement.")
    elif match_score >= 40:
        st.warning("Moderate Match: Consider addressing the skill gaps to improve your chances.")
    else:
        st.error("Low Match: Your current profile may not be the best fit for this specific position.")
