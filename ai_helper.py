import os
import google.generativeai as genai
from typing import Dict, List, Any, Optional
import json

# Configure the Gemini API with your key
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Use Gemini Pro model
model = genai.GenerativeModel('gemini-pro')

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
        # Fallback to a basic structure
        return {
            "contact_info": {},
            "education": [],
            "experience": [],
            "skills": {"technical": [], "soft": []},
            "certifications": []
        }

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
        # Fallback to a basic structure
        return {
            "skills": {"technical": [], "soft": []},
            "experience": [],
            "education": [],
            "responsibilities": [],
            "preferred_qualifications": []
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
        # Fallback to a basic structure
        return {
            "scores": {
                "skills_match": 0,
                "experience_match": 0,
                "education_match": 0,
                "overall_match": 0
            },
            "matching_skills": [],
            "missing_skills": [],
            "recommendations": ["Error analyzing match score"],
            "feedback_summary": "Unable to analyze match"
        }