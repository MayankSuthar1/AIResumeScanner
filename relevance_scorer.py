import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ai_helper import calculate_match_score
# from database import save_job_match
import re

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
            # resume_id = resume.get("resume_id")
            resume_info = resume.get("parsed_data", {})
            
            # Calculate match using AI
            match_analysis = calculate_match_score(resume_info, job_requirements)
            
            # Save match data to database if resume_id is available
            # if resume_id and job_title:
            #     save_job_match(
            #         resume_id=resume_id,
            #         job_title=job_title,
            #         job_description=job_description,
            #         match_analysis=match_analysis
            #     )
            
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
                skill_match_result = extract_matching_items(resume_skills, job_skills, matcher_type="skills")
                matching_skills = skill_match_result["matching_items"]
                missing_skills = skill_match_result["missing_items"]
                skills_match = len(matching_skills) / len(job_skills) if job_skills else 0
            
            # Calculate experience match (enhanced)
            experience_match = calculate_experience_match(
                resume.get("parsed_data", {}).get("experience", []), 
                job_experience
            )
            
            # Calculate education match (enhanced)
            education_match = calculate_education_match(
                resume.get("parsed_data", {}).get("education", []), 
                job_education
            )
            
            # Calculate overall match score with weighted components
            # FIX: Properly compute the weighted average (removing the parenthesis bug)
            overall_score = (
                0.5 * skills_match + 
                0.3 * experience_match + 
                0.2 * education_match
            ) * 100  # Convert to percentage
            
            # Create recommendations based on the gaps
            recommendations = generate_recommendations(
                skills_match, experience_match, education_match,
                matching_skills, missing_skills,
                resume.get("parsed_data", {}), job_requirements
            )
            
            # Generate feedback summary
            feedback_summary = generate_feedback_summary(
                skills_match, experience_match, education_match, overall_score
            )
            
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
                "recommendations": recommendations,
                "feedback_summary": feedback_summary
            }
            
            # Save fallback match data to database if resume_id is available
            # if resume.get("resume_id") and job_title:
            #     save_job_match(
            #         resume_id=resume.get("resume_id"),
            #         job_title=job_title,
            #         job_description=job_description,
            #         match_analysis=fallback_match
            #     )
            
            relevance_scores.append(overall_score)
            match_data.append(fallback_match)
    
    # Store the match data in the parsed_resumes for later use
    for i, resume in enumerate(parsed_resumes):
        if i < len(match_data) and match_data[i] is not None:
            resume["match_analysis"] = match_data[i]
    
    return relevance_scores

def calculate_experience_match(resume_experience, job_experience):
    """
    Calculate how well the candidate's experience matches the job requirements
    
    Evaluation criteria:
    1. Years of Experience Match (70% weight):
       - Compares total years of candidate's relevant work experience against required years
       - 100% match if candidate meets or exceeds the required years
       - Partial match based on ratio of candidate's years to required years
       - Years are calculated from start/end dates of each position
    
    2. Domain Relevance Match (30% weight):
       - Evaluates if candidate has experience in the specific domains required
       - Domains are matched by checking job titles, company names, and descriptions
       - 100% match if all required domains are matched
       - Partial match based on percentage of matched domains
    
    Args:
        resume_experience: List of candidate's experience items
        job_experience: List of job experience requirements
        
    Returns:
        Float between 0-1 representing match percentage
    """
    if not job_experience:
        return 0.7  # Default to moderately positive if no specific requirements
    
    # Extract required years from job requirements
    required_years = 0
    required_domains = set()
    
    for exp in job_experience:
        # Extract years requirement
        if isinstance(exp, dict):
            years_text = exp.get("years", "")
            if years_text:
                # Extract numeric values from strings like "3+ years" or "2-4 years"
                years_match = re.search(r'(\d+)', years_text)
                if years_match:
                    years = int(years_match.group(1))
                    required_years = max(required_years, years)
            
            # Extract domain requirement
            domain = exp.get("domain", "")
            if domain:
                required_domains.add(domain.lower())
    
    if not resume_experience:
        return 0  # No experience listed
    
    # Calculate total years of experience from resume
    total_years = 0
    domain_matches = set()
    
    for exp in resume_experience:
        if isinstance(exp, dict):
            # Try to calculate duration for this position
            start_date = exp.get("start_date", "")
            end_date = exp.get("end_date", "")
            
            # Extract years from dates
            years = extract_years_from_dates(start_date, end_date)
            total_years += years
            
            # Check if this experience matches required domains
            title = exp.get("title", "").lower()
            company = exp.get("company", "").lower()
            description = exp.get("description", "").lower()
            
            # Check if any required domains appear in this experience
            for domain in required_domains:
                if (domain in title or domain in company or domain in description):
                    domain_matches.add(domain)
    
    # Calculate years match (100% if meets or exceeds requirements, scaled otherwise)
    years_match = min(1.0, total_years / required_years) if required_years > 0 else 0.7
    
    # Calculate domain match
    domain_match_result = extract_matching_items(domain_matches, required_domains, matcher_type="domain")
    domain_match = len(domain_match_result["matching_items"]) / len(required_domains) if required_domains else 0.7
    
    # Combine years and domain matches (years is more important)
    experience_match = (0.7 * years_match) + (0.3 * domain_match)
    
    return experience_match

def extract_years_from_dates(start_date, end_date):
    """
    Extract years of experience from date strings
    """
    if not start_date:
        return 0
    
    # Extract years from date strings
    try:
        # Pattern for dates like "January 2020" or "Jan 2020" or "2020"
        start_year_match = re.search(r'(\d{4})', start_date)
        
        # Handle end date - could be "Present" or actual date
        end_year = None
        if end_date and end_date.lower() != "present" and end_date.lower() != "current":
            end_year_match = re.search(r'(\d{4})', end_date) 
            if end_year_match:
                end_year = int(end_year_match.group(1))
        else:
            # If "Present", use current year
            from datetime import datetime
            end_year = datetime.now().year
        
        if start_year_match and end_year:
            start_year = int(start_year_match.group(1))
            return max(0, end_year - start_year)  # Ensure non-negative
    except Exception:
        pass
    
    # Default to 1 year if we couldn't parse
    return 1

def calculate_education_match(resume_education, job_education):
    """
    Calculate how well the candidate's education matches the job requirements
    
    Evaluation criteria:
    1. Education Level Match (60% weight):
       - Compares the highest education level of the candidate against the required level
       - Education levels in ascending order: high school, associate, bachelor, master, doctorate
       - 100% match if candidate's level meets or exceeds the required level
       - Partial match if candidate has lower but related education
    
    2. Field of Study Match (40% weight):
       - Evaluates if the candidate's field of study matches the required fields
       - 100% match if all required fields are matched
       - Partial match based on percentage of matched fields
    
    Args:
        resume_education: List of candidate's education items
        job_education: List of job education requirements
        
    Returns:
        Float between 0-1 representing match percentage
    """
    if not job_education:
        return 0.7  # Default to moderately positive if no specific requirements
    
    if not resume_education:
        return 0.0  # No education listed
    
    # Define education levels for comparison
    edu_levels = {
        "high school": 1,
        "associate": 2, 
        "bachelor": 3,
        "bachelors": 3, 
        "undergraduate": 3,
        "bs": 3, 
        "ba": 3,
        "master": 4,
        "masters": 4,
        "ms": 4, 
        "ma": 4,
        "mba": 4,
        "phd": 5, 
        "doctorate": 5
    }
    
    # Extract required degree level and field
    required_level = 0
    required_fields = set()
    
    for edu in job_education:
        if isinstance(edu, dict):
            degree = edu.get("degree", "").lower()
            
            # Determine the education level
            for level_name, level_value in edu_levels.items():
                if level_name in degree:
                    required_level = max(required_level, level_value)
                    break
            
            # Extract field of study
            field = edu.get("field", "")
            if field:
                required_fields.add(field.lower())
        elif isinstance(edu, str):
            # Handle case where education is a string
            edu_lower = edu.lower()
            for level_name, level_value in edu_levels.items():
                if level_name in edu_lower:
                    required_level = max(required_level, level_value)
                    break
    
    # Find candidate's highest education level and field
    highest_level = 0
    field_matches = set()
    
    for edu in resume_education:
        if isinstance(edu, dict):
            degree = edu.get("degree", "").lower()
            
            # Determine education level
            for level_name, level_value in edu_levels.items():
                if level_name in degree:
                    highest_level = max(highest_level, level_value)
                    break
            
            # Check if field matches any required fields
            field = edu.get("institution", "").lower() + " " + edu.get("degree", "").lower()
            for required_field in required_fields:
                if required_field in field:
                    field_matches.add(required_field)
    
    # Calculate level match
    level_match = 0
    if required_level > 0:
        level_match = min(1.0, highest_level / required_level)
    else:
        level_match = 0.7  # No specific level required
    
    # Calculate field match
    field_match_result = extract_matching_items(field_matches, required_fields, matcher_type="education")
    field_match = len(field_match_result["matching_items"]) / len(required_fields) if required_fields else 0.7
    
    # Combine level and field matches
    education_match = (0.6 * level_match) + (0.4 * field_match)
    
    return education_match

def generate_recommendations(skills_match, experience_match, education_match, 
                             matching_skills, missing_skills, resume_data, job_requirements):
    """
    Generate personalized recommendations based on the gaps in the match
    """
    recommendations = []
    
    # Skills recommendations
    if skills_match < 0.7:
        if missing_skills:
            top_missing = missing_skills[:3]  # Get top 3 missing skills
            recommendations.append(f"Add these key skills to your resume: {', '.join(top_missing)}")
            
        if skills_match < 0.5:
            recommendations.append("Consider upskilling in areas relevant to this role")
    
    # Experience recommendations
    if experience_match < 0.7:
        job_exp = job_requirements.get("experience", [])
        required_years = 0
        required_domains = []
        
        for exp in job_exp:
            if isinstance(exp, dict):
                years_text = exp.get("years", "")
                years_match = re.search(r'(\d+)', years_text) if years_text else None
                if years_match:
                    required_years = max(required_years, int(years_match.group(1)))
                
                domain = exp.get("domain", "")
                if domain:
                    required_domains.append(domain)
        
        if required_years > 0:
            recommendations.append(f"Highlight experience that demonstrates {required_years}+ years in this field")
        
        if required_domains:
            domain_str = ", ".join(required_domains[:2])  # Limit to first 2 domains
            recommendations.append(f"Emphasize experience in {domain_str}")
    
    # Education recommendations
    if education_match < 0.7:
        job_edu = job_requirements.get("education", [])
        for edu in job_edu:
            if isinstance(edu, dict):
                degree = edu.get("degree", "")
                field = edu.get("field", "")
                if degree and field:
                    recommendations.append(f"Include your {degree} in {field} or equivalent experience")
                    break
                elif degree:
                    recommendations.append(f"Include your {degree} or equivalent experience")
                    break
    
    # General recommendations
    if len(recommendations) < 2:
        recommendations.append("Tailor your resume to highlight relevant experience for this role")
    
    if len(recommendations) < 3 and matching_skills:
        recommendations.append(f"Emphasize your expertise in {', '.join(matching_skills[:3])}")
    
    return recommendations

def generate_feedback_summary(skills_match, experience_match, education_match, overall_score):
    """
    Generate a personalized feedback summary based on match scores
    """
    # Convert decimal scores to percentages
    skills_pct = skills_match * 100
    experience_pct = experience_match * 100
    education_pct = education_match * 100
    
    # Determine the strongest and weakest areas
    scores = {
        "Skills": skills_pct,
        "Experience": experience_pct, 
        "Education": education_pct
    }
    
    strongest = max(scores, key=scores.get)
    weakest = min(scores, key=scores.get)
    
    # Generate appropriate feedback based on overall score
    if overall_score >= 85:
        summary = f"Excellent match! Your profile is very well aligned with this role. "
        summary += f"Your {strongest.lower()} is particularly strong for this position."
    elif overall_score >= 70:
        summary = f"Good match with this position. Your {strongest.lower()} matches the requirements well. "
        summary += f"With some improvements to your {weakest.lower()}, you could be an excellent candidate."
    elif overall_score >= 55:
        summary = f"Decent match with some gaps. Your {strongest.lower()} is relevant, but there are "
        summary += f"significant gaps in your {weakest.lower()} that you should address."
    else:
        summary = f"This position might not be the best fit for your current profile. Consider roles "
        summary += f"that better align with your skills or address the gaps in your {weakest.lower()}."
    
    return summary

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

def get_skill_synonyms():
    """
    Returns a dictionary of skill synonyms to help match related terms
    between job descriptions and resumes.
    
    Example: When a job requires "SQL queries" but resume has "SQL"
    """
    return {
        # Programming languages
        "sql": ["sql", "sql queries", "sql database", "mysql", "postgresql", "tsql", "pl/sql", "oracle sql"],
        "python": ["python", "python programming", "python development", "python scripting", "django", "flask"],
        "java": ["java", "java programming", "java development", "j2ee", "spring", "java ee", "java se"],
        "javascript": ["javascript", "js", "ecmascript", "node.js", "nodejs", "react.js", "vue.js", "angular.js"],
        "typescript": ["typescript", "ts"],
        "c++": ["c++", "cpp", "c plus plus", "cplusplus"],
        "c#": ["c#", "csharp", "c sharp", ".net"],
        
        # Web technologies
        "html": ["html", "html5", "markup", "web development"],
        "css": ["css", "css3", "cascading style sheets", "scss", "sass", "less"],
        "react": ["react", "reactjs", "react.js", "react native"],
        "angular": ["angular", "angularjs", "angular.js", "angular 2+"],
        "vue": ["vue", "vuejs", "vue.js"],
        
        # Cloud platforms
        "aws": ["aws", "amazon web services", "amazon cloud", "ec2", "s3", "lambda"],
        "azure": ["azure", "microsoft azure", "azure cloud", "azure devops"],
        "gcp": ["gcp", "google cloud platform", "google cloud", "cloud platform"],
        
        # Data science/ML
        "machine learning": ["machine learning", "ml", "deep learning", "ai", "artificial intelligence", "neural networks"],
        "data science": ["data science", "data analytics", "data analysis", "big data"],
        "tensorflow": ["tensorflow", "tf", "keras"],
        "pytorch": ["pytorch", "torch"],
        "pandas": ["pandas", "numpy", "data frames"],
        
        # DevOps
        "docker": ["docker", "containerization", "containers"],
        "kubernetes": ["kubernetes", "k8s", "container orchestration"],
        "jenkins": ["jenkins", "ci/cd", "continuous integration"],
        "git": ["git", "github", "version control", "gitlab", "bitbucket"],
        
        # Soft skills
        "communication": ["communication", "verbal communication", "written communication"],
        "leadership": ["leadership", "team leadership", "people management", "team management"],
        "problem solving": ["problem solving", "analytical thinking", "critical thinking", "troubleshooting"],
        "teamwork": ["teamwork", "collaboration", "team player", "cross-functional teams"],
    }

def get_education_synonyms():
    """
    Returns a dictionary of education level synonyms to help match related terms
    between job descriptions and resumes.
    """
    return {
        "bachelor": ["bachelor", "bachelors", "bachelor's", "bs", "ba", "b.s.", "b.a.", "undergraduate", "btech", "b.tech"],
        "master": ["master", "masters", "master's", "ms", "ma", "m.s.", "m.a.", "mtech", "m.tech", "mba", "m.b.a."],
        "doctorate": ["phd", "ph.d.", "doctorate", "doctoral", "doctor of philosophy"],
        "associate": ["associate", "associates", "associate's", "a.a.", "a.s."],
        "high school": ["high school", "hs", "secondary", "secondary school", "high school diploma"],
    }

def get_domain_synonyms():
    """
    Returns a dictionary of experience domain synonyms to help match related terms
    between job descriptions and resumes.
    """
    return {
        "software development": ["software development", "software engineering", "programming", "coding", "application development"],
        "data science": ["data science", "machine learning", "data mining", "statistical analysis", "predictive modeling"],
        "web development": ["web development", "front end", "frontend", "front-end", "back end", "backend", "back-end", "full stack", "fullstack", "full-stack"],
        "devops": ["devops", "cloud operations", "site reliability", "platform engineering", "infrastructure"],
        "project management": ["project management", "program management", "product management", "scrum master", "agile management"],
        "marketing": ["marketing", "digital marketing", "market research", "brand management", "seo", "sem"],
        "finance": ["finance", "accounting", "financial analysis", "bookkeeping", "budget management"],
        "sales": ["sales", "business development", "account management", "client acquisition"],
        "customer service": ["customer service", "customer support", "client services", "help desk"],
        "healthcare": ["healthcare", "medical", "clinical", "health services", "patient care"],
    }

def extract_matching_items(resume_items, job_items, matcher_type="skills"):
    """
    Extract matching and missing items between resume and job requirements
    using synonym matching for better accuracy.
    
    Args:
        resume_items: List of items from the resume
        job_items: List of items from the job description
        matcher_type: Type of matching to perform (skills, education, domain)
        
    Returns:
        Dictionary with matching_items and missing_items
    """
    if not job_items:
        return {"matching_items": [], "missing_items": []}
    
    # Get the appropriate synonym dictionary based on matcher_type
    if matcher_type == "skills":
        synonyms = get_skill_synonyms()
    elif matcher_type == "education":
        synonyms = get_education_synonyms()
    elif matcher_type == "domain":
        synonyms = get_domain_synonyms()
    else:
        synonyms = {}
    
    # Normalize all items to lowercase
    resume_items_lower = [item.lower() if isinstance(item, str) else "" for item in resume_items]
    job_items_lower = [item.lower() if isinstance(item, str) else "" for item in job_items]
    
    matching_items = []
    missing_items = []
    
    for job_item in job_items_lower:
        # Skip empty items
        if not job_item:
            continue
            
        # Check if the exact item or any of its synonyms is in the resume
        item_found = False
        
        # Direct match
        if job_item in resume_items_lower:
            matching_items.append(job_item)
            item_found = True
            continue
            
        # Synonym match
        for base_term, synonym_list in synonyms.items():
            # Check if job_item is related to this base term
            job_item_matches_base = False
            for syn in synonym_list:
                if syn in job_item or job_item in syn:
                    job_item_matches_base = True
                    break
            
            # If job_item is related to this base term, check if any synonym is in resume
            if job_item_matches_base:
                for resume_item in resume_items_lower:
                    for syn in synonym_list:
                        if syn in resume_item or resume_item in syn:
                            matching_items.append(job_item)  # Use the job's term in the match list
                            item_found = True
                            break
                    if item_found:
                        break
            
            if item_found:
                break
                
        # If no match found, add to missing items
        if not item_found:
            missing_items.append(job_item)
    
    # Return unique items (no duplicates)
    return {
        "matching_items": list(set(matching_items)), 
        "missing_items": list(set(missing_items))
    }
