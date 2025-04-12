import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ai_helper import calculate_match_score
from database import save_job_match

def calculate_relevance_scores(parsed_resumes, job_requirements, job_title="", job_description=""):
    """
    Calculate relevance scores between resumes and job requirements using Gemini AI
    """
    relevance_scores = []
    match_data = []  # Store detailed match information for each resume
    
    for resume in parsed_resumes:
        # Skip resumes that failed to parse
        if resume.get("status") != "success":
            relevance_scores.append(0)
            match_data.append(None)
            continue
        
        try:
            # Use Gemini AI to calculate match score
            print(f"Using Gemini AI to calculate match score for {resume.get('filename', 'resume')}")
            resume_id = resume.get("resume_id")
            resume_info = resume.get("parsed_data", {})
            
            # Calculate match using AI
            match_analysis = calculate_match_score(resume_info, job_requirements)
            
            # Save match data to database if resume_id is available
            if resume_id and job_title:
                save_job_match(
                    resume_id=resume_id,
                    job_title=job_title,
                    job_description=job_description,
                    match_analysis=match_analysis
                )
            
            # Extract overall score
            overall_score = match_analysis.get("scores", {}).get("overall_match", 0)
            relevance_scores.append(overall_score)
            match_data.append(match_analysis)
            
        except Exception as e:
            print(f"AI match scoring failed: {e}, falling back to traditional scoring")
            # Fall back to traditional scoring
            
            # Extract job skills
            job_skills = set()
            if isinstance(job_requirements.get("skills"), dict):
                # New AI format with technical and soft skills
                tech_skills = job_requirements.get("skills", {}).get("technical", [])
                soft_skills = job_requirements.get("skills", {}).get("soft", [])
                job_skills = set([s.lower() for s in tech_skills + soft_skills])
            else:
                # Old format with just a list of skills
                job_skills = set([s.lower() for s in job_requirements.get("skills", [])])
                
            job_experience = job_requirements.get("experience", [])
            job_education = job_requirements.get("education", [])
            
            # Get resume skills
            resume_skills = set()
            if isinstance(resume.get("parsed_data", {}).get("skills"), dict):
                # New AI format
                resume_data = resume.get("parsed_data", {})
                tech_skills = resume_data.get("skills", {}).get("technical", [])
                soft_skills = resume_data.get("skills", {}).get("soft", [])
                resume_skills = set([s.lower() for s in tech_skills + soft_skills])
            else:
                # Old format
                resume_skills = set([s.lower() for s in resume.get("skills", [])])
            
            # Calculate skills match
            skills_match = 0
            matching_skills = []
            missing_skills = []
            if job_skills and resume_skills:
                matching_skills = list(resume_skills.intersection(job_skills))
                missing_skills = list(job_skills - resume_skills)
                skills_match = len(matching_skills) / len(job_skills) if job_skills else 0
            
            # Calculate experience match (simplified)
            experience_match = 0.5  # Default mid-range score for experience
            
            # Calculate education match (simplified)
            education_match = 0.5  # Default mid-range score for education
            
            # Calculate overall match score with weighted components
            overall_score = (
                0.5 * skills_match + 
                0.3 * experience_match + 
                0.2 * education_match
            ) * 100  # Convert to percentage
            
            # Create a match analysis structure similar to AI output for consistency
            fallback_match = {
                "scores": {
                    "skills_match": round(skills_match * 100),
                    "experience_match": round(experience_match * 100),
                    "education_match": round(education_match * 100),
                    "overall_match": round(overall_score)
                },
                "matching_skills": matching_skills,
                "missing_skills": missing_skills,
                "recommendations": [
                    "Add missing skills to your resume",
                    "Highlight relevant experience more clearly"
                ],
                "feedback_summary": "Basic match analysis performed without AI assistance"
            }
            
            # Save fallback match data to database if resume_id is available
            if resume.get("resume_id") and job_title:
                save_job_match(
                    resume_id=resume.get("resume_id"),
                    job_title=job_title,
                    job_description=job_description,
                    match_analysis=fallback_match
                )
            
            relevance_scores.append(overall_score)
            match_data.append(fallback_match)
    
    # Store the match data in the parsed_resumes for later use
    for i, resume in enumerate(parsed_resumes):
        if i < len(match_data) and match_data[i] is not None:
            resume["match_analysis"] = match_data[i]
    
    return relevance_scores

def rank_resumes(parsed_resumes, relevance_scores):
    """
    Rank resumes based on their relevance scores
    """
    # Combine resumes with their scores
    resume_with_scores = []
    for i, resume in enumerate(parsed_resumes):
        # Skip resumes that failed to parse
        if resume.get("status") != "success":
            continue
        
        # Add score to resume data
        resume_copy = resume.copy()
        resume_copy["match_score"] = relevance_scores[i]
        resume_with_scores.append(resume_copy)
    
    # Sort by score in descending order
    ranked_resumes = sorted(resume_with_scores, key=lambda x: x["match_score"], reverse=True)
    
    return ranked_resumes
