import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_relevance_scores(parsed_resumes, job_requirements):
    """
    Calculate relevance scores between resumes and job requirements
    """
    relevance_scores = []
    
    # Extract job skills
    job_skills = set(job_requirements.get("skills", []))
    job_experience = set(job_requirements.get("experience", []))
    job_education = set(job_requirements.get("education", []))
    
    for resume in parsed_resumes:
        # Skip resumes that failed to parse
        if resume.get("status") != "success":
            relevance_scores.append(0)
            continue
        
        # Calculate skills match
        resume_skills = set([skill.lower() for skill in resume.get("skills", [])])
        skills_match = 0
        if job_skills and resume_skills:
            matching_skills = resume_skills.intersection(job_skills)
            skills_match = len(matching_skills) / len(job_skills) if job_skills else 0
        
        # Calculate experience match using text similarity
        experience_match = 0
        if job_experience and resume.get("experience"):
            # Combine all experience text
            resume_exp_text = " ".join(resume.get("experience", []))
            job_exp_text = " ".join(job_experience)
            
            # Vectorize and calculate similarity
            vectorizer = TfidfVectorizer()
            try:
                tfidf_matrix = vectorizer.fit_transform([resume_exp_text, job_exp_text])
                experience_match = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            except:
                # Handle case when vectorizer fails (e.g., empty strings)
                experience_match = 0
        
        # Calculate education match using text similarity
        education_match = 0
        if job_education and resume.get("education"):
            # Combine all education text
            resume_edu_text = " ".join(resume.get("education", []))
            job_edu_text = " ".join(job_education)
            
            # Vectorize and calculate similarity
            vectorizer = TfidfVectorizer()
            try:
                tfidf_matrix = vectorizer.fit_transform([resume_edu_text, job_edu_text])
                education_match = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            except:
                # Handle case when vectorizer fails
                education_match = 0
        
        # Calculate overall match score with weighted components
        # Skills are most important, followed by experience, then education
        overall_score = (
            0.5 * skills_match + 
            0.3 * experience_match + 
            0.2 * education_match
        ) * 100  # Convert to percentage
        
        relevance_scores.append(overall_score)
    
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
