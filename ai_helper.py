import os
import google.generativeai as genai
from typing import Dict, List, Any, Optional
import json
import re
import tempfile
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from pdf2image import convert_from_path, convert_from_bytes
from datetime import datetime
import streamlit as st

# Configure the Gemini API with your key - prioritize session state API key if available
# GOOGLE_API_KEY = st.session_state.get('api_key') or os.environ.get("GOOGLE_API_KEY")
# global model
# if GOOGLE_API_KEY:
#     # print("Warning: GOOGLE_API_KEY not set. AI features will not work.")


def configure_gemini_api(GOOGLE_API_KEY: Optional[str] = None):
    """
    Configure the Gemini API with the provided API key and set up the model for use.
    """
    try:
        # Configure the Gemini API
        genai.configure(api_key=GOOGLE_API_KEY)
        
        # Use the specific model provided by the user
        model_name = 'gemini-2.5-pro-exp-03-25'
        model = None
        
        try:
            model = genai.GenerativeModel(model_name)
            # Test with a simple prompt to verify it works
            test_response = model.generate_content("Hello")
            print(f"Successfully connected to Gemini AI using model: {model_name} \n {test_response.text}")
            st.session_state.model = model
            return True
            
        except Exception as e:
            print(f"Error connecting to model {model_name}: {e}")
            
            # Capture the original error message before trying fallbacks
            original_error = str(e)
            if "API key" in original_error or "authentication" in original_error.lower() or "invalid" in original_error.lower():
                # This appears to be an API key issue - raise error immediately
                raise Exception(f"Invalid API key: {original_error}")
            
            # Try fallback models if not an API key issue
            fallback_models = [
                                "gemini-2.5-pro-exp-03-25",
                                "gemini-2.5-pro-preview-03-25",
                                "gemini-2.0-flash-exp",
                                "gemini-2.0-flash",
                                "gemini-2.0-flash-001",
                                "gemini-2.0-flash-exp-image-generation",
                                "gemini-2.0-flash-lite-001",
                                "gemini-2.0-flash-lite",
                                "gemini-2.0-flash-lite-preview-02-05",
                                "gemini-2.0-flash-lite-preview",
                                "gemini-2.0-pro-exp",
                                "gemini-2.0-pro-exp-02-05",
                                "gemini-2.0-flash-thinking-exp-01-21",
                                "gemini-2.0-flash-thinking-exp",
                                "gemini-2.0-flash-thinking-exp-1219",
                                "gemini-1.5-pro-latest",
                                "gemini-1.5-pro-001",
                                "gemini-1.5-pro-002",
                                "gemini-1.5-pro",
                                "gemini-1.5-flash-latest",
                                "gemini-1.5-flash-001",
                                "gemini-1.5-flash-001-tuning",
                                "gemini-1.5-flash",
                                "gemini-1.5-flash-002",
                                "gemini-1.5-flash-8b",
                                "gemini-1.5-flash-8b-001",
                                "gemini-1.5-flash-8b-latest",
                                "gemini-1.5-flash-8b-exp-0827",
                                "gemini-1.5-flash-8b-exp-0924"
                            ]
            
            for fallback_model in fallback_models:
                try:
                    model = genai.GenerativeModel(fallback_model)
                    # Test with a simple prompt to verify it works
                    test_response = model.generate_content("Hello")
                    print(f"Successfully connected to Gemini AI using fallback model: {fallback_model}")
                    if model:
                        st.session_state.model = model
                    return True
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
        # Re-raise the exception to be handled by the caller
        raise e

def extract_text_from_pdf(pdf_path_or_data, use_ocr=True):
    """
    Extract text from PDF using both native text extraction and OCR when necessary
    
    Args:
        pdf_path_or_data: Path to PDF file or PDF data as bytes
        use_ocr: Whether to use OCR for text extraction (default: True)
        
    Returns:
        Extracted text from the PDF
    """
    try:
        # Check if input is a file path or bytes
        is_bytes = isinstance(pdf_path_or_data, bytes)
        
        # Open the PDF using PyMuPDF (fitz)
        if is_bytes:
            pdf_document = fitz.open(stream=pdf_path_or_data, filetype="pdf")
        else:
            pdf_document = fitz.open(pdf_path_or_data)
        
        # First attempt: Extract text directly from PDF
        text = ""
        text_extraction_success = False
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            page_text = page.get_text()
            if page_text.strip():
                text += page_text + "\n\n"
                text_extraction_success = True
        
        # If native text extraction succeeded and OCR is not required
        if text_extraction_success and not use_ocr:
            pdf_document.close()
            return text.strip()
        
        # Second attempt: Apply OCR if native extraction failed or OCR is explicitly requested
        if use_ocr:
            print("Using OCR to extract text from PDF...")
            ocr_text = ""
            
            # Convert PDF to images
            if is_bytes:
                # Create a temporary file to save the PDF data
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
                    temp_pdf.write(pdf_path_or_data)
                    temp_path = temp_pdf.name
                images = convert_from_path(temp_path)
                os.unlink(temp_path)  # Remove the temporary file
            else:
                images = convert_from_path(pdf_path_or_data)
            
            # Apply OCR to each page with enhanced image quality
            for i, img in enumerate(images):
                # Enhance image quality for better OCR
                from PIL import ImageEnhance
                gray_img = img.convert('L')  # Convert to grayscale
                enhancer = ImageEnhance.Contrast(gray_img)
                enhanced_img = enhancer.enhance(1.5)  # Increase contrast by 50%
                
                # Use pytesseract with optimal settings for resume text
                page_text = pytesseract.image_to_string(
                    enhanced_img, 
                    lang='eng',  # English language
                    config='--psm 6'  # Assume a single uniform block of text
                )
                ocr_text += page_text + "\n\n"
            
            # Combine native text and OCR text if both are available,
            # or use OCR text if native extraction failed
            if text_extraction_success:
                # Combine both texts, prioritizing the one with more content
                if len(ocr_text) > len(text) * 1.2:  # OCR has 20% more content
                    final_text = ocr_text
                else:
                    final_text = text
            else:
                final_text = ocr_text
            
            pdf_document.close()
            return final_text.strip()
        
        pdf_document.close()
        return text.strip()
    
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def extract_resume_info_fallback(text: str) -> Dict[str, Any]:
    """
    Fallback function to extract resume information using rule-based parsing
    when AI-based extraction fails
    
    Args:
        text: The full text of the resume
        
    Returns:
        Dictionary with basic extracted information
    """
    print("Using fallback function for resume information extraction")
    
    # Initialize the structure for extracted information
    resume_info = {
        "contact_info": {
            "name": None,
            "email": None,
            "phone": None,
            "location": None
        },
        "education": [],
        "experience": [],
        "skills": {"technical": [], "soft": []},
        "certifications": []
    }
    
    # Extract email using regex
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    if emails:
        resume_info["contact_info"]["email"] = emails[0]
    
    # Extract phone using regex
    phone_pattern = r'(?:\+\d{1,3}[-\.\s]?)?\(?\d{3}\)?[-\.\s]?\d{3}[-\.\s]?\d{4}'
    phones = re.findall(phone_pattern, text)
    if phones:
        resume_info["contact_info"]["phone"] = phones[0]
    
    # Extract skills (simplified approach)
    common_tech_skills = ["python", "java", "javascript", "html", "css", "sql", 
                         "aws", "docker", "kubernetes", "react", "angular", "node.js"]
    common_soft_skills = ["leadership", "communication", "teamwork", "problem solving", 
                         "time management", "critical thinking"]
    
    for skill in common_tech_skills:
        if re.search(r'\b' + skill + r'\b', text.lower()):
            resume_info["skills"]["technical"].append(skill)
    
    for skill in common_soft_skills:
        if re.search(r'\b' + skill + r'\b', text.lower()):
            resume_info["skills"]["soft"].append(skill)
    
    # Return the extracted information
    return resume_info

def extract_resume_info(text_or_pdf_path: str, is_pdf=False) -> Dict[str, Any]:
    """
    Extract structured information from resume text or PDF using Gemini AI
    
    Args:
        text_or_pdf_path: Either the full text of the resume or path to PDF file
        is_pdf: Whether the input is a PDF file path (default: False)
        
    Returns:
        Dictionary with extracted information
    """
    # If input is a PDF file, extract text with enhanced PDF extraction
    if is_pdf:
        text = extract_text_from_pdf(text_or_pdf_path, use_ocr=True)
        if not text:
            print("Failed to extract text from PDF")
            return {}
    else:
        text = text_or_pdf_path
    
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
            "phone": "123-456-7890",
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
        model = st.session_state.get('model', None)

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

def analyze_job_description_fallback(text: str) -> Dict[str, Any]:
    """
    Fallback function to analyze job description using rule-based parsing
    when AI-based analysis fails
    
    Args:
        text: The full text of the job description
        
    Returns:
        Dictionary with basic extracted job requirements
    """
    print("Using fallback function for job description analysis")
    
    # Initialize the structure for extracted information
    job_info = {
        "skills": {
            "technical": [],
            "soft": []
        },
        "experience": [],
        "education": [],
        "responsibilities": [],
        "preferred_qualifications": []
    }
    
    # Extract skills (simplified approach)
    common_tech_skills = ["python", "java", "javascript", "html", "css", "sql", 
                         "aws", "docker", "kubernetes", "react", "angular", "node.js"]
    common_soft_skills = ["leadership", "communication", "teamwork", "problem solving", 
                         "time management", "critical thinking"]
    
    for skill in common_tech_skills:
        if re.search(r'\b' + skill + r'\b', text.lower()):
            job_info["skills"]["technical"].append(skill)
    
    for skill in common_soft_skills:
        if re.search(r'\b' + skill + r'\b', text.lower()):
            job_info["skills"]["soft"].append(skill)
    
    # Look for common education requirements
    education_patterns = [
        r'\b(?:bachelor|master|phd|doctorate|bs|ms|ba|ma|mba)\b.*?\b(?:degree|education)\b',
        r'\bdegree\s+in\s+[^.]*'
    ]
    
    for pattern in education_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            job_info["education"].append({"degree": match.strip(), "field": None})
    
    # Look for experience requirements
    experience_pattern = r'\b(\d+)[+]?\s+years?\s+(?:of\s+)?(?:experience|exp)\b'
    experience_matches = re.findall(experience_pattern, text, re.IGNORECASE)
    
    if experience_matches:
        job_info["experience"].append({
            "years": experience_matches[0],
            "domain": "General"
        })
    
    return job_info

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
                "years": "X",
                "domain": "Domain Name"
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
    
    Only respond with the JSON, nothing else.
    
    Here is the job description:
    
    {text}
    """
    
    try:
        model = st.session_state.get('model', None)

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
        job_info = json.loads(json_str)
        return job_info
    except Exception as e:
        print(f"Error analyzing job description with Gemini: {e}")
        # Use a rules-based approach as fallback
        return analyze_job_description_fallback(text)

def calculate_experience_match(resume_experience, job_experience):
    """Helper function to calculate experience match score"""
    # Simple implementation
    if not job_experience:
        return 0.8  # Default score if no experience requirements
    
    # Check if candidate has any experience
    if not resume_experience:
        return 0.2  # Low score if no experience
        
    # Basic matching - calculate based on years of experience
    job_years_required = 0
    for exp in job_experience:
        if isinstance(exp, dict) and 'years' in exp:
            try:
                job_years_required = int(str(exp['years']).split('-')[0])
                break
            except (ValueError, TypeError):
                job_years_required = 1  # Default if parsing fails
    
    # Estimate candidate's total years of experience
    candidate_years = 0
    for exp in resume_experience:
        if isinstance(exp, dict) and 'start_date' in exp:
            try:
                # Calculate duration between start and end dates
                start_year = int(exp['start_date'].split('-')[0])
                
                if 'end_date' in exp and exp['end_date'] and exp['end_date'].lower() != 'present':
                    end_year = int(exp['end_date'].split('-')[0])
                else:
                    end_year = datetime.now().year
                
                candidate_years += (end_year - start_year)
            except (ValueError, IndexError):
                candidate_years += 1  # Default if parsing fails
    
    # Calculate match ratio
    if job_years_required > 0:
        experience_ratio = min(1.0, candidate_years / job_years_required)
    else:
        experience_ratio = 0.7  # Default
    
    return experience_ratio

def calculate_education_match(resume_education, job_education):
    """Helper function to calculate education match score"""
    # Simple implementation
    if not job_education:
        return 0.9  # High default score if no education requirements
    
    if not resume_education:
        return 0.3  # Low score if no education
    
    # Very basic matching
    # Assume match if candidate has any degree
    return 0.7  # Default education match

def generate_recommendations(skills_match, experience_match, education_match, 
                           matching_skills, missing_skills, resume_info, job_info):
    """Generate recommendations based on match analysis"""
    recommendations = []
    
    # Skills recommendations
    if skills_match < 0.7:
        recommendations.append(f"Consider adding skills in: {', '.join(missing_skills[:3])}")
    
    # Experience recommendations
    if experience_match < 0.6:
        recommendations.append("Highlight more relevant work experience for this role")
    
    # Education recommendations
    if education_match < 0.5:
        recommendations.append("Consider additional education or certifications relevant to this role")
    
    # Default recommendation if none generated
    if not recommendations:
        recommendations.append("Your profile appears to be a good match for this role")
    
    return recommendations

def generate_feedback_summary(skills_match, experience_match, education_match, overall_match):
    """Generate overall feedback summary"""
    if overall_match >= 80:
        return "Strong match! Your profile aligns well with this job's requirements."
    elif overall_match >= 60:
        return "Good match. You meet many of the job requirements but could improve in some areas."
    elif overall_match >= 40:
        return "Moderate match. You have some relevant qualifications but may need additional skills or experience."
    else:
        return "Limited match. Consider developing more skills or experience for this type of role."

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
    
    # Skills matching
    # Extract lists of skills
    resume_tech_skills = set([skill.lower() for skill in resume_info.get("skills", {}).get("technical", [])])
    resume_soft_skills = set([skill.lower() for skill in resume_info.get("skills", {}).get("soft", [])])
    job_tech_skills = set([skill.lower() for skill in job_info.get("skills", {}).get("technical", [])])
    job_soft_skills = set([skill.lower() for skill in job_info.get("skills", {}).get("soft", [])])
    
    # Use the enhanced matching function with synonym support
    from relevance_scorer import extract_matching_items
    skill_match_result = extract_matching_items(
        list(resume_tech_skills) + list(resume_soft_skills),
        list(job_tech_skills) + list(job_soft_skills),
        matcher_type="skills"
    )
    
    matching_skills = skill_match_result["matching_items"]
    missing_skills = skill_match_result["missing_items"]
    
    # Calculate skill match percentage
    total_job_skills = len(job_tech_skills) + len(job_soft_skills)
    if total_job_skills > 0:
        skills_match = min(1.0, len(matching_skills) / total_job_skills)
    else:
        skills_match = 0.7  # Default if no skills specified
    
    # Experience match (improved)
    experience_match, matching_experience, missing_experience = calculate_enhanced_experience_match(
        resume_info.get("experience", []), 
        job_info.get("experience", [])
    )
    
    # Education match (improved)
    education_match, matching_education, missing_education = calculate_enhanced_education_match(
        resume_info.get("education", []), 
        job_info.get("education", [])
    )
    
    # Calculate overall match with weighted components
    overall_match = (skills_match * 0.5) + (experience_match * 0.3) + (education_match * 0.2)
    overall_match = round(overall_match * 100)  # Convert to percentage
    
    # Generate recommendations
    recommendations = generate_recommendations(
        skills_match, experience_match, education_match,
        matching_skills, missing_skills,
        resume_info, job_info
    )
    
    # Generate feedback summary
    feedback_summary = generate_feedback_summary(
        skills_match, experience_match, education_match, overall_match
    )
    
    # Return structured match analysis
    return {
        "scores": {
            "skills_match": round(skills_match * 100),
            "experience_match": round(experience_match * 100),
            "education_match": round(education_match * 100),
            "overall_match": overall_match
        },
        "matching_skills": matching_skills,
        "missing_skills": missing_skills,
        "matching_experience": matching_experience,
        "missing_experience": missing_experience,
        "matching_education": matching_education,
        "missing_education": missing_education,
        "recommendations": recommendations,
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
    # Convert input data to JSON strings for the prompt
    resume_json = json.dumps(resume_info, indent=2)
    job_json = json.dumps(job_info, indent=2)
    
    prompt = f"""
    You are an expert AI assistant specialized in evaluating how well a candidate's resume matches a job description.
    
    Analyze the provided resume information and job description information (both in JSON format) and calculate:
    
    1. Skills match score (0-100): How well the candidate's technical and soft skills match the job requirements
    2. Experience match score (0-100): How well the candidate's experience matches the job requirements
    3. Education match score (0-100): How well the candidate's education matches the job requirements
    4. Overall match score (0-100): A weighted calculation with skills (50%), experience (30%), and education (20%)
    
    Also provide:
    - A list of matching skills found in both the resume and job description
    - A list of missing skills that are in the job description but not in the resume
    - 2-3 specific recommendations for the candidate to improve their fit for this role
    - A brief summary of the candidate's fit for this role (2-3 sentences)
    
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
    
    Resume Information:
    {resume_json}
    
    Job Description Information:
    {job_json}
    """
    
    try:
        model = st.session_state.get('model', None)

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

def calculate_enhanced_experience_match(resume_experience, job_experience):
    """
    Calculate how well the candidate's experience matches the job requirements and 
    returns both the score and the matching/missing items
    
    Args:
        resume_experience: List of candidate's experience items
        job_experience: List of job experience requirements
        
    Returns:
        Tuple of (match_score, matching_experience, missing_experience)
    """
    if not job_experience:
        return 0.7, [], []  # Default score if no experience requirements
    
    # Check if candidate has any experience
    if not resume_experience:
        return 0.2, [], [f"{exp.get('domain', 'experience')} ({exp.get('years', '')} years)" 
                          for exp in job_experience if isinstance(exp, dict)]
    
    # Extract required years and domains from job requirements
    required_years = 0
    required_domains = []
    
    for exp in job_experience:
        if isinstance(exp, dict):
            # Extract years requirement
            years_text = exp.get("years", "")
            if years_text:
                # Extract numeric values from strings like "3+ years" or "2-4 years"
                years_match = re.search(r'(\d+)', str(years_text))
                if years_match:
                    years = int(years_match.group(1))
                    required_years = max(required_years, years)
            
            # Extract domain requirement
            domain = exp.get("domain", "")
            if domain:
                required_domains.append(domain.lower())
    
    # Calculate total years of experience from resume
    total_years = 0
    candidate_domains = []
    
    for exp in resume_experience:
        if isinstance(exp, dict):
            # Try to calculate duration for this position
            start_date = exp.get("start_date", "")
            end_date = exp.get("end_date", "")
            
            # Extract years from dates
            years = extract_years_from_dates(start_date, end_date)
            total_years += years
            
            # Extract domains from experience
            title = exp.get("title", "").lower() if exp.get("title") else ""
            company = exp.get("company", "").lower() if exp.get("company") else ""
            description = exp.get("description", "").lower() if exp.get("description") else ""
            
            # Create a combined text to search for domains
            combined_text = f"{title} {company} {description}"
            
            # Use title as domain if it's a senior position
            if title and any(senior_term in title.lower() for senior_term in 
                            ["senior", "lead", "manager", "director", "head", "principal"]):
                # Extract the domain part (e.g., "Senior Software Engineer" -> "Software Engineer")
                domain_parts = title.split()
                if len(domain_parts) > 1:
                    candidate_domains.append(" ".join(domain_parts[1:]))
                else:
                    candidate_domains.append(title)
            
            # Add other relevant domains from the experience
            for domain in ["software", "data", "marketing", "sales", "finance", "design", 
                          "research", "management", "engineering", "development"]:
                if domain in combined_text and domain not in candidate_domains:
                    candidate_domains.append(domain)
    
    # Match domains using synonym matching
    from relevance_scorer import extract_matching_items
    domain_match_result = extract_matching_items(
        candidate_domains, 
        required_domains,
        matcher_type="domain"
    )
    
    matching_domains = domain_match_result["matching_items"]
    missing_domains = domain_match_result["missing_items"]
    
    # Format matching and missing experience items
    matching_experience = []
    missing_experience = []
    
    # Create formatted matching experience items
    if total_years > 0 and required_years > 0 and total_years >= required_years:
        matching_experience.append(f"{total_years} years of experience (meets {required_years}+ requirement)")
    
    # Add matching domains
    for domain in matching_domains:
        matching_experience.append(f"Experience in {domain}")
    
    # Create formatted missing experience items
    if total_years > 0 and required_years > 0 and total_years < required_years:
        missing_experience.append(f"{required_years - total_years} more years of experience")
    
    # Add missing domains
    for domain in missing_domains:
        missing_experience.append(f"Experience in {domain}")
    
    # Calculate experience match score
    years_match = min(1.0, total_years / required_years) if required_years > 0 else 0.7
    domains_match = len(matching_domains) / len(required_domains) if required_domains else 0.7
    
    # Combine years and domain matches (years is more important)
    experience_match = (0.7 * years_match) + (0.3 * domains_match)
    
    return experience_match, matching_experience, missing_experience

def calculate_enhanced_education_match(resume_education, job_education):
    """
    Calculate how well the candidate's education matches the job requirements and 
    returns both the score and the matching/missing items
    
    Args:
        resume_education: List of candidate's education items
        job_education: List of job education requirements
        
    Returns:
        Tuple of (match_score, matching_education, missing_education)
    """
    if not job_education:
        return 0.7, [], []  # Default score if no education requirements
    
    if not resume_education:
        return 0.3, [], [f"{edu.get('degree', '')} in {edu.get('field', '')}" 
                         for edu in job_education if isinstance(edu, dict)]
    
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
    
    # Extract required degree level and fields from job requirements
    required_level = 0
    required_fields = []
    
    for edu in job_education:
        if isinstance(edu, dict):
            degree = edu.get("degree", "").lower() if edu.get("degree") else ""
            field = edu.get("field", "").lower() if edu.get("field") else ""
            
            # Determine the education level
            for level_name, level_value in edu_levels.items():
                if level_name in degree:
                    required_level = max(required_level, level_value)
                    break
            
            # Add field requirement
            if field:
                required_fields.append(field)
            elif degree:
                # If no specific field but degree is specified
                required_fields.append(f"{degree}")
        elif isinstance(edu, str):
            # Handle case where education is a string
            edu_lower = edu.lower()
            for level_name, level_value in edu_levels.items():
                if level_name in edu_lower:
                    required_level = max(required_level, level_value)
                    # Extract potential field
                    if "in" in edu_lower:
                        field_part = edu_lower.split("in", 1)[1].strip()
                        if field_part:
                            required_fields.append(field_part)
                    else:
                        required_fields.append(level_name)
                    break
    
    # Extract candidate's education information
    highest_level = 0
    candidate_fields = []
    
    for edu in resume_education:
        if isinstance(edu, dict):
            degree = edu.get("degree", "").lower() if edu.get("degree") else ""
            institution = edu.get("institution", "").lower() if edu.get("institution") else ""
            
            # Determine education level
            for level_name, level_value in edu_levels.items():
                if level_name in degree:
                    highest_level = max(highest_level, level_value)
                    break
            
            # Extract field from degree
            field_parts = degree.split("in", 1)
            if len(field_parts) > 1:
                field = field_parts[1].strip()
                if field:
                    candidate_fields.append(field)
            else:
                # Use whole degree as field
                candidate_fields.append(degree)
                
            # Add institution as potential field match
            if institution:
                candidate_fields.append(institution)
    
    # Match fields using synonym matching
    from relevance_scorer import extract_matching_items
    field_match_result = extract_matching_items(
        candidate_fields, 
        required_fields,
        matcher_type="education"
    )
    
    matching_fields = field_match_result["matching_items"]
    missing_fields = field_match_result["missing_items"]
    
    # Format matching and missing education items
    matching_education = []
    missing_education = []
    
    # Check if required education level is met
    level_names = {v: k for k, v in edu_levels.items()}
    
    if required_level > 0 and highest_level >= required_level:
        matching_education.append(f"{level_names.get(highest_level, 'Education')} level (meets {level_names.get(required_level, '')} requirement)")
    
    # Add matching fields
    for field in matching_fields:
        matching_education.append(f"Education in {field}")
    
    # Add missing education level if not met
    if required_level > 0 and highest_level < required_level:
        missing_education.append(f"{level_names.get(required_level, 'Higher education')} level required")
    
    # Add missing fields
    for field in missing_fields:
        missing_education.append(f"Education in {field}")
    
    # Calculate match scores
    level_match = min(1.0, highest_level / required_level) if required_level > 0 else 0.7
    field_match = len(matching_fields) / len(required_fields) if required_fields else 0.7
    
    # Combine level and field matches
    education_match = (0.6 * level_match) + (0.4 * field_match)
    
    return education_match, matching_education, missing_education

def extract_years_from_dates(start_date, end_date):
    """
    Extract years of experience from date strings
    """
    if not start_date:
        return 0
    
    # Extract years from date strings
    try:
        # Pattern for dates like "January 2020" or "Jan 2020" or "2020"
        start_year_match = re.search(r'(\d{4})', str(start_date))
        
        # Handle end date - could be "Present" or actual date
        end_year = None
        if end_date and str(end_date).lower() != "present" and str(end_date).lower() != "current":
            end_year_match = re.search(r'(\d{4})', str(end_date)) 
            if end_year_match:
                end_year = int(end_year_match.group(1))
        else:
            # If "Present", use current year
            end_year = datetime.now().year
        
        if start_year_match and end_year:
            start_year = int(start_year_match.group(1))
            return max(0, end_year - start_year)  # Ensure non-negative
    except Exception as e:
        print(f"Error extracting years: {e}")
    
    # Default to 1 year if we couldn't parse
    return 1

def answer_chat_question(question, resume_info, job_info, match_analysis):
    """
    Answer a chat question about the resume match using Gemini AI
    
    Args:
        question: The user's question
        resume_info: Structured resume information
        job_info: Structured job requirements
        match_analysis: Match analysis results
        
    Returns:
        String with the answer to the question
    """
    # Convert input data to JSON strings for the prompt
    resume_json = json.dumps(resume_info, indent=2)
    job_json = json.dumps(job_info, indent=2)
    match_json = json.dumps(match_analysis, indent=2)
    
    prompt = f"""
    You are an AI resume advisor helping a job seeker understand how their resume matches a job description.
    
    Answer the user's question based on the detailed resume information, job requirements, and match analysis provided below.
    Be specific, comprehensive, and helpful in your response. If the user is asking about gaps or how to improve,
    provide detailed and actionable advice.
    
    When discussing missing skills, experience, or education requirements, list ALL of them, not just a few examples.
    Provide comprehensive details about what the job requires and what the candidate is missing.
    
    Format your response with Markdown for readability.
    
    User's question: {question}
    
    Resume information:
    {resume_json}
    
    Job requirements:
    {job_json}
    
    Match analysis:
    {match_json}
    """
    
    try:
        model = st.session_state.get('model', None)

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating chat response with Gemini: {e}")
        # Fallback response
        return generate_fallback_answer(question, resume_info, job_info, match_analysis)

def generate_fallback_answer(question, resume_info, job_info, match_analysis):
    """
    Generate a fallback answer when AI fails
    """
    # Extract key information for the fallback response
    scores = match_analysis.get("scores", {})
    overall_match = scores.get("overall_match", 0)
    skills_match = scores.get("skills_match", 0)
    experience_match = scores.get("experience_match", 0)
    education_match = scores.get("education_match", 0)
    
    # Get specific data points
    matching_skills = match_analysis.get("matching_skills", [])
    missing_skills = match_analysis.get("missing_skills", [])
    
    # Get all missing experience details
    missing_experience = match_analysis.get("missing_experience", [])
    
    # Get all missing education details
    missing_education = match_analysis.get("missing_education", [])
    
    # Get recommendations
    recommendations = match_analysis.get("recommendations", [])
    
    # Create a detailed response based on the question keywords
    question_lower = question.lower()
    
    if "skill" in question_lower or "missing" in question_lower:
        response = f"### Skills Analysis\n\n"
        response += f"**Skills Match Score:** {skills_match}%\n\n"
        
        response += f"**Matching Skills ({len(matching_skills)}):**\n"
        if matching_skills:
            response += ", ".join(matching_skills)
        else:
            response += "None found"
        response += "\n\n"
        
        response += f"**Missing Skills ({len(missing_skills)}):**\n"
        if missing_skills:
            response += ", ".join(missing_skills)
        else:
            response += "None identified"
        response += "\n\n"
        
        response += "**Recommendations for skills improvement:**\n"
        for rec in [r for r in recommendations if "skill" in r.lower()]:
            response += f"- {rec}\n"
    
    elif "experience" in question_lower:
        response = f"### Experience Analysis\n\n"
        response += f"**Experience Match Score:** {experience_match}%\n\n"
        
        if job_info.get("experience"):
            response += "**Required Experience:**\n"
            for exp in job_info.get("experience"):
                if isinstance(exp, dict):
                    years = exp.get("years", "")
                    domain = exp.get("domain", "")
                    response += f"- {years} in {domain}\n" if years and domain else ""
        
        if missing_experience:
            response += "\n**Missing Experience:**\n"
            for exp in missing_experience:
                response += f"- {exp}\n"
    
    elif "education" in question_lower:
        response = f"### Education Analysis\n\n"
        response += f"**Education Match Score:** {education_match}%\n\n"
        
        if job_info.get("education"):
            response += "**Required Education:**\n"
            for edu in job_info.get("education"):
                if isinstance(edu, dict):
                    degree = edu.get("degree", "")
                    field = edu.get("field", "")
                    response += f"- {degree} in {field}\n" if degree else ""
        
        if missing_education:
            response += "\n**Missing Education:**\n"
            for edu in missing_education:
                response += f"- {edu}\n"
    
    elif "improve" in question_lower or "better" in question_lower:
        response = f"### Improvement Recommendations\n\n"
        response += "Here are ways to improve your resume for this job:\n\n"
        
        for rec in recommendations:
            response += f"- {rec}\n"
        
        # Add specific advice based on lowest score
        lowest_category = "skills"
        lowest_score = skills_match
        
        if experience_match < lowest_score:
            lowest_category = "experience"
            lowest_score = experience_match
        
        if education_match < lowest_score:
            lowest_category = "education"
        
        response += f"\nFocus most on improving your **{lowest_category}** as it's your lowest match category."
    
    else:
        # General response
        response = f"### Resume Match Analysis\n\n"
        response += f"**Overall Match:** {overall_match}%\n\n"
        
        response += "**Breakdown:**\n"
        response += f"- Skills Match: {skills_match}%\n"
        response += f"- Experience Match: {experience_match}%\n"
        response += f"- Education Match: {education_match}%\n\n"
        
        response += "**Key Recommendations:**\n"
        for rec in recommendations[:3]:
            response += f"- {rec}\n"
    
    return response