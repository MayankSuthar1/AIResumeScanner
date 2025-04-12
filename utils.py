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
