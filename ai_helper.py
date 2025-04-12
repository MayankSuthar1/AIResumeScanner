import os
import google.generativeai as genai
from typing import Dict, List, Any, Optional
import json
import re

# Configure the Gemini API with your key
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY not set. AI features will not work.")

try:
    # Configure the Gemini API
    genai.configure(api_key=GOOGLE_API_KEY)
    
    # Use the specific model provided by the user
    model_name = 'gemini-2.5-pro-preview-03-25'
    model = None
    
    try:
        model = genai.GenerativeModel(model_name)
        # Test with a simple prompt to verify it works
        test_response = model.generate_content("Hello")
        print(f"Successfully connected to Gemini AI using model: {model_name}")
    except Exception as e:
        print(f"Error connecting to model {model_name}: {e}")
        # Try fallback models if the specified one doesn't work
        fallback_models = ['gemini-pro', 'gemini-1.0-pro', 'gemini-1.5-pro']
        for fallback_model in fallback_models:
            try:
                model = genai.GenerativeModel(fallback_model)
                # Test with a simple prompt to verify it works
                test_response = model.generate_content("Hello")
                print(f"Successfully connected to Gemini AI using fallback model: {fallback_model}")
                break
            except Exception as e2:
                print(f"Error connecting to fallback model {fallback_model}: {e2}")
                continue
    
    if model is None:
        print("Error: Could not connect to any Gemini model. AI features will not work.")
        # Provide a dummy model that will be properly handled by the exception blocks
        class DummyModel:
            def generate_content(self, prompt):
                raise Exception("No Gemini model available")
        model = DummyModel()

except Exception as e:
    print(f"Error configuring Gemini AI: {e}")
    # Provide a dummy model that will be properly handled by the exception blocks
    class DummyModel:
        def generate_content(self, prompt):
            raise Exception("Gemini AI configuration failed")
    model = DummyModel()

def extract_resume_info(text: str) -> Dict[str, Any]:
    """
    Extract structured information from resume text using Gemini AI
    
    Args:
        text: The full text of the resume
        
    Returns:
        Dictionary with extracted information
    """
    prompt = f"""
    You are an expert AI assistant specialized in parsing and extracting information from resumes.
    Based on the resume text provided, extract the following information in JSON format:
    
    1. Contact information (name, email, phone, location)
    2. Education history (degree, institution, graduation date, GPA if available)
    3. Work experience (company, job title, dates, descriptions, achievements)
    4. Skills (both technical and soft skills)
    5. Certifications and licenses
    
    Return the output as a valid JSON object with the following structure:
    {{
        "contact_info": {{
            "name": "Full Name",
            "email": "email@example.com",
            "phone": "phone number",
            "location": "City, State"
        }},
        "education": [
            {{
                "degree": "Degree Name",
                "institution": "Institution Name",
                "graduation_date": "YYYY-MM",
                "gpa": "GPA if available"
            }}
        ],
        "experience": [
            {{
                "company": "Company Name",
                "title": "Job Title",
                "start_date": "YYYY-MM",
                "end_date": "YYYY-MM or 'Present'",
                "description": "Job description",
                "achievements": ["Achievement 1", "Achievement 2"]
            }}
        ],
        "skills": {{"technical": ["skill1", "skill2"], "soft": ["skill1", "skill2"]}},
        "certifications": ["Certification 1", "Certification 2"]
    }}
    
    Only respond with the JSON, nothing else. If you cannot find certain information, use null or empty arrays/objects as appropriate.
    
    Here is the resume text:
    
    {text}
    """
    
    try:
        response = model.generate_content(prompt)
        
        # Extract the JSON from the response
        response_text = response.text
        
        # Handle case where response might have markdown code block
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].strip()
        else:
            json_str = response_text.strip()
            
        # Parse the JSON string into a dictionary
        parsed_info = json.loads(json_str)
        return parsed_info
    except Exception as e:
        print(f"Error extracting resume information with Gemini: {e}")
        # Use a rules-based approach as fallback
        return extract_resume_info_fallback(text)

def analyze_job_description(text: str) -> Dict[str, Any]:
    """
    Analyze job description using Gemini AI
    
    Args:
        text: The full text of the job description
        
    Returns:
        Dictionary with extracted job requirements
    """
    prompt = f"""
    You are an expert AI assistant specialized in analyzing job descriptions.
    Based on the job description provided, extract the following information in JSON format:
    
    1. Required skills (both technical and soft skills)
    2. Required experience (years, specific domain experience)
    3. Required education (degrees, certifications)
    4. Job responsibilities
    5. Preferred qualifications (nice-to-have but not required)
    
    Return the output as a valid JSON object with the following structure:
    {{
        "skills": {{
            "technical": ["skill1", "skill2"],
            "soft": ["skill1", "skill2"]
        }},
        "experience": [
            {{
                "years": "Number of years or range",
                "domain": "Domain or field"
            }}
        ],
        "education": [
            {{
                "degree": "Degree Name",
                "field": "Field of Study"
            }}
        ],
        "responsibilities": ["Responsibility 1", "Responsibility 2"],
        "preferred_qualifications": ["Qualification 1", "Qualification 2"]
    }}
    
    Only respond with the JSON, nothing else. If you cannot find certain information, use null or empty arrays/objects as appropriate.
    
    Here is the job description:
    
    {text}
    """
    
    try:
        response = model.generate_content(prompt)
        
        # Extract the JSON from the response
        response_text = response.text
        
        # Handle case where response might have markdown code block
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].strip()
        else:
            json_str = response_text.strip()
            
        # Parse the JSON string into a dictionary
        parsed_info = json.loads(json_str)
        return parsed_info
    except Exception as e:
        print(f"Error analyzing job description with Gemini: {e}")
        # Use a rules-based approach as fallback
        return analyze_job_description_fallback(text)

# Fallback functions for when AI is not available

def extract_resume_info_fallback(text: str) -> Dict[str, Any]:
    """
    Extract structured information from resume text using rule-based approach
    
    Args:
        text: The full text of the resume
        
    Returns:
        Dictionary with extracted information
    """
    print("Using fallback function for resume parsing")
    
    # Basic structure for result
    result = {
        "contact_info": {},
        "education": [],
        "experience": [],
        "skills": {"technical": [], "soft": []},
        "certifications": []
    }
    
    # Extract email using regex
    import re
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    emails = re.findall(email_pattern, text)
    if emails:
        result["contact_info"]["email"] = emails[0]
    
    # Extract phone using regex
    phone_pattern = r'(\+\d{1,3}[-.\s]?)?(\d{3}[-.\s]?\d{3}[-.\s]?\d{4}|\(\d{3}\)[-.\s]?\d{3}[-.\s]?\d{4}|\d{10})'
    phones = re.findall(phone_pattern, text)
    if phones:
        # Join the phone number parts and clean up
        phone = ''.join(phones[0]).strip()
        phone = re.sub(r'[\s.-]', '', phone)  # Remove spaces, dots, and dashes
        if phone:
            result["contact_info"]["phone"] = phone
    
    # Try to extract skills by looking for common skill keywords
    common_tech_skills = [
        "python", "java", "javascript", "html", "css", "react", "angular", 
        "node", "express", "flask", "django", "sql", "mongodb", "aws", "azure",
        "docker", "kubernetes", "git", "ci/cd", "jenkins", "rest", "api", 
        "nosql", "graphql", "typescript", "c++", "c#", "php", "ruby", "scala",
        "swift", "kotlin", "r", "matlab", "tableau", "excel", "word", "powerpoint",
        "tensorflow", "pytorch", "machine learning", "artificial intelligence",
        "data science", "data analysis", "statistics", "algorithms", "networking"
    ]
    
    common_soft_skills = [
        "communication", "teamwork", "leadership", "problem solving", 
        "critical thinking", "time management", "adaptability", "creativity",
        "project management", "collaboration", "presentation", "organization",
        "analytical", "decision making", "flexibility", "research", "detail oriented",
        "conflict resolution", "mentoring", "training", "customer service", 
        "interpersonal", "negotiation", "multitasking", "agile", "scrum"
    ]
    
    # Check for tech skills
    for skill in common_tech_skills:
        if re.search(r'\b' + skill + r'\b', text.lower()):
            result["skills"]["technical"].append(skill)
    
    # Check for soft skills
    for skill in common_soft_skills:
        if re.search(r'\b' + skill + r'\b', text.lower()):
            result["skills"]["soft"].append(skill)
    
    # Look for education keywords
    edu_keywords = ["bachelor", "master", "phd", "doctorate", "bs", "ms", "ba", "ma", 
                   "degree", "university", "college", "school", "institute", "gpa", "graduated"]
    
    # Split text into paragraphs and look for education information
    paragraphs = text.split('\n\n')
    for paragraph in paragraphs:
        # Check if paragraph might be about education
        if any(keyword in paragraph.lower() for keyword in edu_keywords):
            # Try to extract degree and institution
            degree = None
            institution = None
            
            # Check for common degrees
            degree_patterns = [
                r'bachelor[\'s]* (?:of|in) ([^,\n\.]+)',
                r'master[\'s]* (?:of|in) ([^,\n\.]+)',
                r'phd (?:of|in) ([^,\n\.]+)',
                r'doctorate (?:of|in) ([^,\n\.]+)',
                r'b\.?s\.? (?:of|in)? ([^,\n\.]+)',
                r'm\.?s\.? (?:of|in)? ([^,\n\.]+)',
                r'b\.?a\.? (?:of|in)? ([^,\n\.]+)',
                r'm\.?a\.? (?:of|in)? ([^,\n\.]+)'
            ]
            
            for pattern in degree_patterns:
                match = re.search(pattern, paragraph.lower())
                if match:
                    degree = match.group(0).strip()
                    break
            
            # Try to find institution name (just a simple heuristic)
            lines = paragraph.split('\n')
            for line in lines:
                if "university" in line.lower() or "college" in line.lower() or "institute" in line.lower():
                    institution = line.strip()
                    break
            
            if degree or institution:
                education_entry = {"degree": degree if degree else "Degree not specified"}
                if institution:
                    education_entry["institution"] = institution
                result["education"].append(education_entry)
    
    return result

def analyze_job_description_fallback(text: str) -> Dict[str, Any]:
    """
    Analyze job description using rule-based approach
    
    Args:
        text: The full text of the job description
        
    Returns:
        Dictionary with extracted job requirements
    """
    print("Using fallback function for job description analysis")
    
    # Basic structure for result
    result = {
        "skills": {"technical": [], "soft": []},
        "experience": [],
        "education": [],
        "responsibilities": [],
        "preferred_qualifications": []
    }
    
    # Common technical skills to look for
    common_tech_skills = [
        "python", "java", "javascript", "html", "css", "react", "angular", 
        "node", "express", "flask", "django", "sql", "mongodb", "aws", "azure",
        "docker", "kubernetes", "git", "ci/cd", "jenkins", "rest", "api", 
        "nosql", "graphql", "typescript", "c++", "c#", "php", "ruby", "scala",
        "swift", "kotlin", "r", "matlab", "tableau", "excel", "word", "powerpoint",
        "tensorflow", "pytorch", "machine learning", "artificial intelligence",
        "data science", "data analysis", "statistics", "algorithms", "networking"
    ]
    
    # Common soft skills to look for
    common_soft_skills = [
        "communication", "teamwork", "leadership", "problem solving", 
        "critical thinking", "time management", "adaptability", "creativity",
        "project management", "collaboration", "presentation", "organization",
        "analytical", "decision making", "flexibility", "research", "detail oriented",
        "conflict resolution", "mentoring", "training", "customer service", 
        "interpersonal", "negotiation", "multitasking", "agile", "scrum"
    ]
    
    # Check for tech skills
    for skill in common_tech_skills:
        if re.search(r'\b' + skill + r'\b', text.lower()):
            result["skills"]["technical"].append(skill)
    
    # Check for soft skills
    for skill in common_soft_skills:
        if re.search(r'\b' + skill + r'\b', text.lower()):
            result["skills"]["soft"].append(skill)
    
    # Look for experience requirements (years)
    exp_patterns = [
        r'(\d+)[+]? years? experience',
        r'(\d+)[+]? years? of experience',
        r'experience: (\d+)[+]? years?',
        r'minimum (\d+)[+]? years?',
        r'at least (\d+)[+]? years?'
    ]
    
    for pattern in exp_patterns:
        matches = re.finditer(pattern, text.lower())
        for match in matches:
            years = match.group(1)
            result["experience"].append({
                "years": f"{years}+ years",
                "domain": "General"
            })
    
    # Look for education requirements
    edu_keywords = ["bachelor", "master", "phd", "doctorate", "bs", "ms", "ba", "ma", "degree"]
    edu_fields = ["computer science", "engineering", "business", "information technology", 
                 "data science", "mathematics", "statistics", "economics", "finance"]
    
    for keyword in edu_keywords:
        if keyword in text.lower():
            # Try to find associated field
            field_found = False
            for field in edu_fields:
                if field in text.lower():
                    result["education"].append({
                        "degree": keyword.capitalize(),
                        "field": field.capitalize()
                    })
                    field_found = True
                    break
            
            if not field_found:
                result["education"].append({
                    "degree": keyword.capitalize(),
                    "field": "Not specified"
                })
    
    # Look for responsibilities sections
    resp_section = None
    
    # Check for common section headers
    resp_headers = ["responsibilities", "duties", "what you'll do", "job duties", "key responsibilities"]
    
    # Split text into paragraphs
    paragraphs = text.split('\n\n')
    
    for i, paragraph in enumerate(paragraphs):
        if any(header in paragraph.lower() for header in resp_headers):
            resp_section = i
            break
    
    # If responsibilities section is found, extract bullet points
    if resp_section is not None and resp_section + 1 < len(paragraphs):
        resp_text = paragraphs[resp_section + 1]
        bullet_points = re.findall(r'•\s*(.*?)(?=•|\Z)', resp_text, re.DOTALL)
        
        if not bullet_points:  # Try another pattern if no bullet points found
            bullet_points = resp_text.split('\n')
        
        for point in bullet_points:
            point = point.strip()
            if point and len(point) > 10:  # Only add non-empty, meaningful points
                result["responsibilities"].append(point)
    
    return result

def calculate_match_score_fallback(resume_info: Dict[str, Any], job_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate the match score between a resume and job description using rule-based approach
    
    Args:
        resume_info: Structured resume information
        job_info: Structured job requirements
        
    Returns:
        Dictionary with match scores and feedback
    """
    print("Using fallback function for match scoring")
    
    # Initialize scores
    skills_match = 0
    experience_match = 0
    education_match = 0
    
    # Extract lists of skills
    resume_tech_skills = set([skill.lower() for skill in resume_info.get("skills", {}).get("technical", [])])
    resume_soft_skills = set([skill.lower() for skill in resume_info.get("skills", {}).get("soft", [])])
    job_tech_skills = set([skill.lower() for skill in job_info.get("skills", {}).get("technical", [])])
    job_soft_skills = set([skill.lower() for skill in job_info.get("skills", {}).get("soft", [])])
    
    # Calculate matching skills
    matching_tech_skills = resume_tech_skills.intersection(job_tech_skills)
    matching_soft_skills = resume_soft_skills.intersection(job_soft_skills)
    missing_tech_skills = job_tech_skills - resume_tech_skills
    missing_soft_skills = job_soft_skills - resume_soft_skills
    
    matching_skills = list(matching_tech_skills) + list(matching_soft_skills)
    missing_skills = list(missing_tech_skills) + list(missing_soft_skills)
    
    # Calculate skills match score
    if job_tech_skills or job_soft_skills:  # Avoid division by zero
        total_job_skills = len(job_tech_skills) + len(job_soft_skills)
        total_matching_skills = len(matching_tech_skills) + len(matching_soft_skills)
        skills_match = min(100, round((total_matching_skills / total_job_skills) * 100)) if total_job_skills > 0 else 50
    else:
        skills_match = 50  # Default if no skills are specified in job
    
    # Calculate experience match (simple)
    if job_info.get("experience"):
        # Get the minimum years of experience required
        min_years_required = 0
        for exp in job_info.get("experience", []):
            years_text = exp.get("years", "")
            years_match = re.search(r'(\d+)', years_text)
            if years_match:
                years = int(years_match.group(1))
                min_years_required = max(min_years_required, years)
        
        # Count the years of experience in the resume
        resume_years = 0
        for exp in resume_info.get("experience", []):
            if isinstance(exp, dict) and "start_date" in exp and "end_date" in exp:
                start_year_match = re.search(r'(\d{4})', exp.get("start_date", ""))
                end_year_match = re.search(r'(\d{4})', exp.get("end_date", ""))
                
                if start_year_match and end_year_match:
                    start_year = int(start_year_match.group(1))
                    end_year = 2023 if "present" in exp.get("end_date", "").lower() else int(end_year_match.group(1))
                    resume_years += (end_year - start_year)
        
        # Calculate match
        if min_years_required > 0:
            experience_match = min(100, round((resume_years / min_years_required) * 100))
        else:
            experience_match = 70  # Default if years not specified
    else:
        experience_match = 70  # Default if no experience requirements
    
    # Calculate education match (simple)
    if job_info.get("education"):
        # Check if degrees match
        job_degrees = [edu.get("degree", "").lower() for edu in job_info.get("education", [])]
        resume_degrees = [edu.get("degree", "").lower() for edu in resume_info.get("education", [])]
        
        matched_degrees = 0
        for job_degree in job_degrees:
            for resume_degree in resume_degrees:
                if job_degree in resume_degree or resume_degree in job_degree:
                    matched_degrees += 1
                    break
        
        education_match = min(100, round((matched_degrees / len(job_degrees)) * 100)) if job_degrees else 70
    else:
        education_match = 70  # Default if no education requirements
    
    # Calculate overall match
    overall_match = round((skills_match * 0.5) + (experience_match * 0.3) + (education_match * 0.2))
    
    # Generate recommendations
    recommendations = []
    
    if missing_skills:
        if len(missing_skills) > 3:
            recommendations.append(f"Consider learning the following key skills: {', '.join(missing_skills[:3])}")
        else:
            for skill in missing_skills:
                recommendations.append(f"Add experience with {skill}")
    
    if experience_match < 70:
        recommendations.append("Gain more years of relevant work experience")
    
    if education_match < 70:
        recommendations.append("Consider additional education or certifications")
    
    # Generate feedback summary
    if overall_match >= 80:
        feedback_summary = "Strong candidate match for this position"
    elif overall_match >= 60:
        feedback_summary = "Good candidate match with some areas for improvement"
    elif overall_match >= 40:
        feedback_summary = "Moderate match with several skill gaps"
    else:
        feedback_summary = "Not a strong match for this position"
    
    # Return match analysis
    return {
        "scores": {
            "skills_match": skills_match,
            "experience_match": experience_match,
            "education_match": education_match,
            "overall_match": overall_match
        },
        "matching_skills": matching_skills,
        "missing_skills": missing_skills,
        "recommendations": recommendations if recommendations else ["Improve match with more relevant experience"],
        "feedback_summary": feedback_summary
    }

def calculate_match_score(resume_info: Dict[str, Any], job_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate the match score between a resume and job description using Gemini AI
    
    Args:
        resume_info: Structured resume information
        job_info: Structured job requirements
        
    Returns:
        Dictionary with match scores and feedback
    """
    
    # Convert inputs to JSON strings for the prompt
    resume_json = json.dumps(resume_info)
    job_json = json.dumps(job_info)
    
    prompt = f"""
    You are an expert AI assistant specialized in evaluating candidate profiles against job requirements.
    
    Please analyze the following resume information and job description to calculate match scores and provide feedback.
    
    Resume Information:
    {resume_json}
    
    Job Requirements:
    {job_json}
    
    Calculate the following scores and provide feedback:
    
    1. Skills Match: Score from 0-100 based on how well the candidate's skills match the job's required skills
    2. Experience Match: Score from 0-100 based on how well the candidate's experience matches the job's requirements
    3. Education Match: Score from 0-100 based on how well the candidate's education matches the job's requirements
    4. Overall Match: Weighted average of the above scores (skills: 50%, experience: 30%, education: 20%)
    
    Also provide:
    1. Matching Skills: List of skills that match between resume and job requirements
    2. Missing Skills: List of required skills not found in the resume
    3. Recommendations: Specific suggestions for how the candidate could improve their match
    
    Return the output as a valid JSON object with the following structure:
    {{
        "scores": {{
            "skills_match": 85,
            "experience_match": 70,
            "education_match": 90,
            "overall_match": 82
        }},
        "matching_skills": ["skill1", "skill2"],
        "missing_skills": ["skill3", "skill4"],
        "recommendations": ["Recommendation 1", "Recommendation 2"],
        "feedback_summary": "A brief summary of the candidate's fit for this role"
    }}
    
    Only respond with the JSON, nothing else.
    """
    
    try:
        response = model.generate_content(prompt)
        
        # Extract the JSON from the response
        response_text = response.text
        
        # Handle case where response might have markdown code block
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].strip()
        else:
            json_str = response_text.strip()
            
        # Parse the JSON string into a dictionary
        match_analysis = json.loads(json_str)
        return match_analysis
    except Exception as e:
        print(f"Error calculating match score with Gemini: {e}")
        # Use a rules-based approach as fallback
        return calculate_match_score_fallback(resume_info, job_info)