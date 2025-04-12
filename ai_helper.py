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
    
    # Extract phone using regex - improved pattern to catch more formats
    phone_pattern = r'(\+?1[-.\s]?)?(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\d{10})'
    phones = re.findall(phone_pattern, text)
    if phones:
        # Join the phone number parts and clean up
        phone = ''.join(''.join(phone_parts) for phone_parts in phones[0] if phone_parts).strip()
        phone = re.sub(r'[\s.-]', '', phone)  # Remove spaces, dots, and dashes
        if phone:
            result["contact_info"]["phone"] = phone
    
    # Extract name - look at the first few lines of text
    lines = text.split('\n')
    non_empty_lines = [line.strip() for line in lines if line.strip()]
    
    # Look for a name at the beginning of the resume
    if non_empty_lines:
        # Exclude lines with common non-name patterns
        excludes = ['resume', 'cv', 'curriculum', 'vitae', '@', '.com', 'http', 'www']
        for i in range(min(3, len(non_empty_lines))):
            line = non_empty_lines[i]
            if (
                len(line.split()) <= 5 and            # Most names are short
                not any(x in line.lower() for x in excludes) and
                not re.search(r'\d{3}', line) and     # Names typically don't have 3 consecutive digits
                not re.search(email_pattern, line)    # Not an email
            ):
                result["contact_info"]["name"] = line
                break
    
    # Extract location/address
    address_indicators = ['street', 'avenue', 'road', 'blvd', 'lane', 'drive', 'circle', 'apt', 'apartment', '#']
    city_state_pattern = r'\b[A-Z][a-zA-Z\s]+,\s*[A-Z]{2}\s*\d{5}\b'  # City, ST ZIP
    
    # First check for city, state pattern
    location_match = re.search(city_state_pattern, text)
    if location_match:
        result["contact_info"]["location"] = location_match.group(0)
    else:
        # Look for address indicators in first few lines
        for i in range(min(5, len(non_empty_lines))):
            line = non_empty_lines[i].lower()
            if any(ind in line for ind in address_indicators) and len(line.split()) <= 8:
                result["contact_info"]["location"] = non_empty_lines[i]
                break
    
    # Enhanced technical skills list with more keywords
    common_tech_skills = [
        # Programming Languages
        "python", "java", "javascript", "typescript", "c\\+\\+", "c#", "ruby", "php", "go", "rust", 
        "swift", "kotlin", "r", "matlab", "perl", "scala", "bash", "powershell", "sql", "pl/sql",
        
        # Web Development
        "html", "css", "sass", "less", "bootstrap", "tailwind", "jquery", "json", "xml", "rest", 
        "soap", "graphql", "api", "webgl", "svg", "webpack", "babel", "gatsby", "next.js", "nuxt.js",
        
        # Frameworks & Libraries
        "react", "angular", "vue", "svelte", "ember", "node", "express", "django", "flask", 
        "spring", "asp.net", "laravel", "rails", "symfony", ".net", "flutter", "xamarin",
        
        # Databases
        "sql", "mysql", "postgresql", "oracle", "mongodb", "sqlite", "redis", "cassandra", "dynamodb", 
        "couchdb", "neo4j", "mariadb", "firebase", "elasticsearch", "nosql", "hbase", "supabase",
        
        # Cloud & DevOps
        "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "terraform", "ansible", 
        "chef", "puppet", "prometheus", "grafana", "git", "github", "gitlab", "bitbucket", 
        "ci/cd", "devops", "cloudflare", "nginx", "apache", "systemd", "serverless",
        
        # AI/ML/Data Science
        "machine learning", "deep learning", "artificial intelligence", "ai", "nlp", "computer vision",
        "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn",
        "data science", "data analysis", "data engineering", "data visualization", "big data", "spark",
        
        # Business Tools & Analytics
        "tableau", "power bi", "excel", "spss", "sas", "looker", "domo", "qlik", "snowflake", "redshift",
        
        # Mobile Development
        "android", "ios", "react native", "ionic", "swift", "objective-c", "kotlin", "flutter",
        
        # Network & Infrastructure
        "networking", "tcp/ip", "dns", "dhcp", "vpn", "firewall", "load balancer", "linux", "unix", 
        "windows", "macos", "active directory", "ldap", "oauth", "saml"
    ]
    
    # Enhanced soft skills list
    common_soft_skills = [
        # Communication & Interpersonal
        "communication", "teamwork", "collaboration", "interpersonal", "presentation", "public speaking",
        "writing", "technical writing", "reporting", "negotiation", "conflict resolution", "persuasion",
        
        # Leadership & Management
        "leadership", "management", "team management", "people management", "project management", 
        "product management", "strategic planning", "decision making", "mentoring", "coaching", 
        "delegation", "performance management", "team building", "stakeholder management",
        
        # Problem Solving & Thinking
        "problem solving", "critical thinking", "analytical skills", "research", "troubleshooting",
        "debugging", "root cause analysis", "systems thinking", "creative thinking", "innovation",
        
        # Work Habits & Attributes
        "time management", "organization", "prioritization", "multitasking", "attention to detail",
        "adaptability", "flexibility", "reliability", "dependability", "initiative", "self-motivated",
        "proactive", "resourceful", "resilience", "stress management", "work ethic", "accountability",
        
        # Customer & Business Focus
        "customer service", "user experience", "client relations", "relationship building", 
        "business acumen", "market research", "budgeting", "cost analysis", "strategy"
    ]
    
    # Function to handle skills extraction with different section headers
    def extract_skills_from_section(section_name, text):
        skills_found = []
        section_patterns = [
            rf'(?i){section_name}s?[:\s]*\n(.*?)(?=\n\n|\n[A-Z]|\Z)',  # "Skills:" or "SKILLS:" etc.
            rf'(?i){section_name}s?[:\s]*\s(.*?)(?=\n\n|\n[A-Z]|\Z)'   # Inline skills section
        ]
        
        for pattern in section_patterns:
            section_match = re.search(pattern, text, re.DOTALL)
            if section_match:
                section_text = section_match.group(1)
                # Split by common delimiters
                for delimiter in [',', '•', '·', '-', '|', '\n', '/', '\\', ':', ';']:
                    if delimiter in section_text:
                        skill_items = [s.strip() for s in section_text.split(delimiter) if s.strip()]
                        for skill in skill_items:
                            # Clean up and add reasonable-length skills
                            skill = re.sub(r'\s+', ' ', skill).strip()
                            if 2 <= len(skill) <= 50:  # Reasonable skill name length
                                skills_found.append(skill.lower())
        
        return skills_found
    
    # Extract skills from potential skills sections
    skills_section_found = False
    for section_name in ["skill", "proficienc", "technolog", "tool", "expert", "competenc"]:
        section_skills = extract_skills_from_section(section_name, text)
        if section_skills:
            skills_section_found = True
            for skill in section_skills:
                # Try to categorize the skill
                if any(tech.lower() in skill.lower() for tech in common_tech_skills):
                    result["skills"]["technical"].append(skill)
                elif any(soft.lower() in skill.lower() for soft in common_soft_skills):
                    result["skills"]["soft"].append(skill)
                else:
                    # Default to technical if unclear
                    result["skills"]["technical"].append(skill)
    
    # If no skills section found, fall back to keyword search
    if not skills_section_found or len(result["skills"]["technical"]) + len(result["skills"]["soft"]) < 5:
        # Check for tech skills
        for skill in common_tech_skills:
            if re.search(r'\b' + re.escape(skill) + r'\b', text.lower()):
                result["skills"]["technical"].append(skill)
        
        # Check for soft skills
        for skill in common_soft_skills:
            if re.search(r'\b' + re.escape(skill) + r'\b', text.lower()):
                result["skills"]["soft"].append(skill)
    
    # Enhanced education extraction
    edu_keywords = [
        "bachelor", "master", "phd", "doctorate", "bs", "ms", "ba", "ma", "mba", "b.tech", "m.tech",
        "degree", "university", "college", "school", "institute", "gpa", "graduated", "diploma",
        "certification", "certificate", "major", "minor"
    ]
    
    # Look for education section
    edu_section_match = re.search(r'(?i)education[:|\s]*\n(.*?)(?=\n\n|\n[A-Z]|\Z)', text, re.DOTALL)
    if edu_section_match:
        edu_section = edu_section_match.group(1)
        # Split by double newlines to get each education entry
        edu_entries = re.split(r'\n\s*\n', edu_section)
        for entry in edu_entries:
            if entry.strip():
                # Create new education entry
                edu_entry = {}
                
                # Try to extract degree
                degree_pattern = r'(?i)(bachelor|master|phd|doctorate|mba|bs|ms|ba|ma|b\.tech|m\.tech)[\'\s]*(of|in|degree)?\s*([a-z\s,]+)'
                degree_match = re.search(degree_pattern, entry)
                if degree_match:
                    degree_type = degree_match.group(1)
                    field = degree_match.group(3).strip() if degree_match.group(3) else ""
                    edu_entry["degree"] = f"{degree_type.title()} {field.title()}".strip()
                
                # Try to extract institution
                uni_pattern = r'(?i)(university|college|institute|school) (?:of )?([a-z\s,]+)'
                uni_match = re.search(uni_pattern, entry)
                if uni_match:
                    edu_entry["institution"] = uni_match.group(0).strip()
                else:
                    # Look for capitalized lines which might be institution names
                    entry_lines = entry.split('\n')
                    for line in entry_lines:
                        if re.match(r'^[A-Z][a-zA-Z\s]+$', line.strip()) and len(line.strip().split()) <= 5:
                            edu_entry["institution"] = line.strip()
                            break
                
                # Try to extract graduation date
                date_pattern = r'(?i)(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|may|june|july|august|september|october|november|december)[,\s-]*\d{4}'
                dates = re.findall(date_pattern, entry)
                if dates:
                    edu_entry["graduation_date"] = dates[-1]  # Use the last date found as graduation date
                
                # Try to extract GPA
                gpa_pattern = r'(?i)gpa[:\s]*([0-9.]+)'
                gpa_match = re.search(gpa_pattern, entry)
                if gpa_match:
                    edu_entry["gpa"] = gpa_match.group(1)
                
                # Add education entry if we found any information
                if edu_entry:
                    result["education"].append(edu_entry)
    
    # If no structured education section found, try to extract from full text
    if not result["education"]:
        # Look for degree patterns in the full text
        degree_patterns = [
            r'(?i)(bachelor|master|phd|doctorate)[\'s]*\s+(of|in)\s+([a-z\s,]+)',
            r'(?i)(bs|ms|ba|ma|mba|b\.tech|m\.tech)\s+(in|of)?\s*([a-z\s,]+)',
            r'(?i)([a-z\s]+) (university|college|institute|school)',
        ]
        
        for pattern in degree_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                # Get some context around the match
                start_pos = max(0, match.start() - 50)
                end_pos = min(len(text), match.end() + 100)
                context = text[start_pos:end_pos]
                
                edu_entry = {}
                if match.lastindex >= 3:  # Full degree pattern with field
                    degree_type = match.group(1)
                    field = match.group(3) if match.group(3) else ""
                    edu_entry["degree"] = f"{degree_type.title()} {field.title()}".strip()
                else:  # Partial match
                    edu_entry["degree"] = match.group(0).strip()
                
                # Look for institution in context
                uni_pattern = r'(?i)([A-Z][a-zA-Z\s,]+) (university|college|institute|school)'
                uni_match = re.search(uni_pattern, context)
                if uni_match:
                    edu_entry["institution"] = uni_match.group(0).strip()
                
                # Add if we have at least degree info
                if "degree" in edu_entry:
                    result["education"].append(edu_entry)
    
    # Extract work experience
    experience_entries = []
    
    # Look for experience section
    exp_section_match = re.search(r'(?i)(experience|employment|work history|professional background)[:|\s]*\n(.*?)(?=\n\n\n|\n[A-Z][a-z]+:\n|\Z)', text, re.DOTALL)
    if exp_section_match:
        exp_section = exp_section_match.group(2)
        
        # Split by patterns that likely indicate separate job entries
        job_entries = re.split(r'\n\s*\n|\n(?=[A-Z][a-zA-Z\s]+\s*\|)', exp_section)
        
        for entry in job_entries:
            if len(entry.strip()) > 20:  # Minimum meaningful content
                exp_entry = {}
                
                # Try to extract company name (often at the beginning of entry or after title)
                lines = entry.split('\n')
                for i, line in enumerate(lines[:3]):  # Check first few lines
                    if re.match(r'^[A-Z][a-zA-Z0-9\s&.,]+$', line.strip()) and len(line.strip().split()) <= 5:
                        exp_entry["company"] = line.strip()
                        break
                
                # Try to extract job title (often at beginning, has keywords like "engineer", "manager", etc.)
                title_keywords = ["engineer", "developer", "manager", "director", "analyst", "specialist", 
                                 "coordinator", "consultant", "assistant", "associate", "lead", "head", 
                                 "architect", "designer", "administrator"]
                
                for i, line in enumerate(lines[:4]):  # Check first few lines
                    if any(keyword.lower() in line.lower() for keyword in title_keywords):
                        exp_entry["title"] = line.strip()
                        break
                
                # Try to extract dates
                date_pattern = r'(?i)(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|may|june|july|august|september|october|november|december)[,\s-]*\d{4}'
                dates = re.findall(date_pattern, entry)
                
                if len(dates) >= 2:  # Start and end dates
                    exp_entry["start_date"] = dates[0]
                    exp_entry["end_date"] = dates[-1]
                elif len(dates) == 1:  # At least one date
                    exp_entry["start_date"] = dates[0]
                    if "present" in entry.lower() or "current" in entry.lower():
                        exp_entry["end_date"] = "Present"
                
                # Try to extract description - rest of the text after title/company/dates
                if "company" in exp_entry or "title" in exp_entry:
                    # Find the line after title or company
                    start_line = 0
                    for i, line in enumerate(lines):
                        if line.strip() == exp_entry.get("company", "") or line.strip() == exp_entry.get("title", ""):
                            start_line = i + 1
                            break
                    
                    description_lines = []
                    for i in range(start_line, len(lines)):
                        # Stop if we hit what looks like a new section
                        if i > start_line and re.match(r'^[A-Z][a-zA-Z\s]+:$', lines[i]):
                            break
                        description_lines.append(lines[i])
                    
                    if description_lines:
                        exp_entry["description"] = " ".join(line.strip() for line in description_lines)
                
                # Extract achievements - often bullet points
                achievements = []
                achievement_pattern = r'•\s*(.*?)(?=•|\n\n|\Z)'
                achievement_matches = re.findall(achievement_pattern, entry)
                if achievement_matches:
                    for match in achievement_matches:
                        if len(match.strip()) > 10:  # Minimum meaningful content
                            achievements.append(match.strip())
                
                if achievements:
                    exp_entry["achievements"] = achievements
                
                # Add to experience if we have at least basic info
                if "company" in exp_entry or "title" in exp_entry:
                    experience_entries.append(exp_entry)
    
    # If still no experience, try to extract from full text based on date patterns
    if not experience_entries:
        # Look for date ranges which often indicate job periods
        date_range_pattern = r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)[\s,]+\d{4})\s*[-–—]\s*(Present|Current|Now|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)[\s,]+\d{4})'
        
        date_ranges = re.finditer(date_range_pattern, text, re.IGNORECASE)
        for date_range in date_ranges:
            # Get context around the date range
            start_pos = max(0, date_range.start() - 100)
            end_pos = min(len(text), date_range.end() + 200)
            context = text[start_pos:end_pos]
            
            exp_entry = {
                "start_date": date_range.group(1),
                "end_date": date_range.group(2)
            }
            
            # Look for job title and company in context
            lines = context.split('\n')
            for line in lines:
                line = line.strip()
                # Look for company name (capitalized, not too long)
                if not "company" in exp_entry and re.match(r'^[A-Z][a-zA-Z0-9\s&.,]+$', line) and len(line.split()) <= 5:
                    exp_entry["company"] = line
                
                # Look for job title
                title_keywords = ["engineer", "developer", "manager", "director", "analyst", "specialist"]
                if not "title" in exp_entry and any(keyword.lower() in line.lower() for keyword in title_keywords):
                    exp_entry["title"] = line
            
            # Add a basic description from the context
            exp_entry["description"] = re.sub(r'\s+', ' ', context).strip()
            
            # Add to experience entries
            if "start_date" in exp_entry and "end_date" in exp_entry:
                experience_entries.append(exp_entry)
    
    # Add experience entries to result
    result["experience"] = experience_entries
    
    # Extract certifications
    cert_keywords = ["certif", "license", "accredit", "diploma", "qualified", "certified"]
    cert_section_match = None
    
    # Look for certifications section
    for keyword in cert_keywords:
        cert_section_match = re.search(rf'(?i){keyword}[a-z]*[:|\s]*\n(.*?)(?=\n\n\n|\n[A-Z][a-z]+:\n|\Z)', text, re.DOTALL)
        if cert_section_match:
            break
    
    if cert_section_match:
        cert_section = cert_section_match.group(1)
        # Split by newlines or bullets
        cert_entries = re.split(r'\n|•|·|-|\*', cert_section)
        for entry in cert_entries:
            entry = entry.strip()
            if entry and len(entry) > 5 and len(entry) < 100:
                result["certifications"].append(entry)
    
    # Remove duplicates in lists
    result["skills"]["technical"] = list(set(result["skills"]["technical"]))
    result["skills"]["soft"] = list(set(result["skills"]["soft"]))
    result["certifications"] = list(set(result["certifications"]))
    
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
    
    # Enhanced technical skills to look for - same as in resume parser
    common_tech_skills = [
        # Programming Languages
        "python", "java", "javascript", "typescript", "c\\+\\+", "c#", "ruby", "php", "go", "rust", 
        "swift", "kotlin", "r", "matlab", "perl", "scala", "bash", "powershell", "sql", "pl/sql",
        
        # Web Development
        "html", "css", "sass", "less", "bootstrap", "tailwind", "jquery", "json", "xml", "rest", 
        "soap", "graphql", "api", "webgl", "svg", "webpack", "babel", "gatsby", "next.js", "nuxt.js",
        
        # Frameworks & Libraries
        "react", "angular", "vue", "svelte", "ember", "node", "express", "django", "flask", 
        "spring", "asp.net", "laravel", "rails", "symfony", ".net", "flutter", "xamarin",
        
        # Databases
        "sql", "mysql", "postgresql", "oracle", "mongodb", "sqlite", "redis", "cassandra", "dynamodb", 
        "couchdb", "neo4j", "mariadb", "firebase", "elasticsearch", "nosql", "hbase", "supabase",
        
        # Cloud & DevOps
        "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "terraform", "ansible", 
        "chef", "puppet", "prometheus", "grafana", "git", "github", "gitlab", "bitbucket", 
        "ci/cd", "devops", "cloudflare", "nginx", "apache", "systemd", "serverless",
        
        # AI/ML/Data Science
        "machine learning", "deep learning", "artificial intelligence", "ai", "nlp", "computer vision",
        "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn",
        "data science", "data analysis", "data engineering", "data visualization", "big data", "spark",
        
        # Business Tools & Analytics
        "tableau", "power bi", "excel", "spss", "sas", "looker", "domo", "qlik", "snowflake", "redshift",
        
        # Mobile Development
        "android", "ios", "react native", "ionic", "swift", "objective-c", "kotlin", "flutter",
        
        # Network & Infrastructure
        "networking", "tcp/ip", "dns", "dhcp", "vpn", "firewall", "load balancer", "linux", "unix", 
        "windows", "macos", "active directory", "ldap", "oauth", "saml"
    ]
    
    # Enhanced soft skills to look for - same as in resume parser
    common_soft_skills = [
        # Communication & Interpersonal
        "communication", "teamwork", "collaboration", "interpersonal", "presentation", "public speaking",
        "writing", "technical writing", "reporting", "negotiation", "conflict resolution", "persuasion",
        
        # Leadership & Management
        "leadership", "management", "team management", "people management", "project management", 
        "product management", "strategic planning", "decision making", "mentoring", "coaching", 
        "delegation", "performance management", "team building", "stakeholder management",
        
        # Problem Solving & Thinking
        "problem solving", "critical thinking", "analytical skills", "research", "troubleshooting",
        "debugging", "root cause analysis", "systems thinking", "creative thinking", "innovation",
        
        # Work Habits & Attributes
        "time management", "organization", "prioritization", "multitasking", "attention to detail",
        "adaptability", "flexibility", "reliability", "dependability", "initiative", "self-motivated",
        "proactive", "resourceful", "resilience", "stress management", "work ethic", "accountability",
        
        # Customer & Business Focus
        "customer service", "user experience", "client relations", "relationship building", 
        "business acumen", "market research", "budgeting", "cost analysis", "strategy"
    ]
    
    # Function to extract content from job description sections
    def extract_section(section_names, text):
        section_content = []
        
        # Create a pattern to match any of the section names
        section_pattern = '|'.join(section_names)
        pattern = rf'(?i)({section_pattern})[:|\s]*\n(.*?)(?=\n\n|\n[A-Z][a-z]+:|\Z)'
        
        section_match = re.search(pattern, text, re.DOTALL)
        if section_match:
            section_text = section_match.group(2).strip()
            
            # Look for bullet points
            bullet_points = re.findall(r'(?:•|■|○|◦|▪|▫|⦿|-|★|\*|[0-9]+\.)\s*(.*?)(?=(?:•|■|○|◦|▪|▫|⦿|-|★|\*|[0-9]+\.)\s|\n\n|\Z)', section_text, re.DOTALL)
            
            if bullet_points:
                for point in bullet_points:
                    point = point.strip()
                    if point and len(point) > 5:  # Minimum reasonable content
                        section_content.append(point)
            else:
                # If no bullet points found, split by newlines
                lines = section_text.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and len(line) > 5:
                        section_content.append(line)
        
        return section_content
    
    # Process skills sections
    skill_section_names = ["skill", "qualification", "requirement", "what you'll need", "what we're looking for"]
    skill_items = extract_section(skill_section_names, text)
    
    # Process preferred qualifications
    preferred_section_names = ["preferred", "nice to have", "bonus", "plus", "desirable"]
    preferred_items = extract_section(preferred_section_names, text)
    result["preferred_qualifications"] = preferred_items
    
    # Process responsibilities
    resp_section_names = ["responsibilit", "duties", "what you'll do", "job duties", "key responsibilit", "day to day", "job description"]
    resp_items = extract_section(resp_section_names, text)
    result["responsibilities"] = resp_items
    
    # Categorize skill items as technical or soft
    for item in skill_items:
        item_lower = item.lower()
        
        # Check if the item contains technical skills
        tech_found = False
        for skill in common_tech_skills:
            if re.search(r'\b' + re.escape(skill) + r'\b', item_lower):
                result["skills"]["technical"].append(skill)
                tech_found = True
        
        # Check if the item contains soft skills
        soft_found = False
        for skill in common_soft_skills:
            if re.search(r'\b' + re.escape(skill) + r'\b', item_lower):
                result["skills"]["soft"].append(skill)
                soft_found = True
        
        # For experience and education requirements in the skills section
        if "year" in item_lower and any(x in item_lower for x in ["experience", "work", "industry"]):
            years_match = re.search(r'(\d+)[+]?[\s-]*(year|yr)', item_lower)
            if years_match:
                years = years_match.group(1)
                # Try to identify the domain
                domain = "General"
                domain_keywords = ["software", "web", "data", "cloud", "devops", "network", "security", "mobile", "frontend", "backend"]
                for kw in domain_keywords:
                    if kw in item_lower:
                        domain = kw.capitalize()
                        break
                
                result["experience"].append({
                    "years": f"{years}+ years",
                    "domain": domain
                })
        
        if any(edu in item_lower for edu in ["degree", "bachelor", "master", "phd", "education"]):
            # Try to extract degree and field
            edu_entry = {}
            
            # Check for degree level
            degree_levels = ["bachelor", "master", "phd", "doctorate", "mba", "bs", "ms", "ba", "ma"]
            for level in degree_levels:
                if level in item_lower:
                    edu_entry["degree"] = level.capitalize()
                    break
            
            # Check for field
            edu_fields = ["computer science", "engineering", "business", "information technology", 
                         "data science", "mathematics", "statistics", "economics", "finance"]
            for field in edu_fields:
                if field in item_lower:
                    edu_entry["field"] = field.capitalize()
                    break
            
            if "degree" in edu_entry or "field" in edu_entry:
                result["education"].append(edu_entry)
    
    # If no skills were found through section analysis, do a full-text keyword search
    if not result["skills"]["technical"] and not result["skills"]["soft"]:
        # Check for tech skills
        for skill in common_tech_skills:
            if re.search(r'\b' + re.escape(skill) + r'\b', text.lower()):
                result["skills"]["technical"].append(skill)
        
        # Check for soft skills
        for skill in common_soft_skills:
            if re.search(r'\b' + re.escape(skill) + r'\b', text.lower()):
                result["skills"]["soft"].append(skill)
    
    # Look for experience requirements (years) if none found so far
    if not result["experience"]:
        exp_patterns = [
            r'(\d+)[+]? years? (?:of )?experience',
            r'experience:? (\d+)[+]? years?',
            r'minimum (?:of )?(\d+)[+]? years?',
            r'at least (\d+)[+]? years?',
            r'(\d+)[+]? to (\d+)[+]? years?'
        ]
        
        for pattern in exp_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                years = match.group(1)
                # Get some context to determine domain
                start_pos = max(0, match.start() - 50)
                end_pos = min(len(text), match.end() + 50)
                context = text[start_pos:end_pos].lower()
                
                # Check common domains
                domain = "General"
                domain_keywords = {
                    "software development": ["software", "develop", "coding", "programming"],
                    "Web Development": ["web", "frontend", "backend", "fullstack", "full-stack"],
                    "Data Science": ["data", "analytics", "machine learning", "ai", "statistics"],
                    "DevOps": ["devops", "cloud", "infrastructure", "ci/cd", "deployment"],
                    "Management": ["manage", "lead", "direct", "supervise"],
                    "Design": ["design", "ui", "ux", "user interface", "user experience"]
                }
                
                for domain_name, keywords in domain_keywords.items():
                    if any(kw in context for kw in keywords):
                        domain = domain_name
                        break
                
                result["experience"].append({
                    "years": f"{years}+ years",
                    "domain": domain
                })
    
    # Look for education requirements if none found so far
    if not result["education"]:
        # Look for education section
        edu_section_names = ["education", "qualification", "academic"]
        edu_section = None
        
        for section_name in edu_section_names:
            pattern = rf'(?i){section_name}[:|\s]*\n(.*?)(?=\n\n|\n[A-Z][a-z]+:|\Z)'
            section_match = re.search(pattern, text, re.DOTALL)
            if section_match:
                edu_section = section_match.group(1).strip()
                break
        
        if edu_section:
            # Look for degree information
            degree_patterns = [
                r'(?i)(bachelor|master|phd|doctorate)[\'s]*\s+(of|in)\s+([a-z\s,]+)',
                r'(?i)(bs|ms|ba|ma|mba|b\.tech|m\.tech)\s+(in|of)?\s*([a-z\s,]+)',
                r'(?i)degree\s+in\s+([a-z\s,]+)'
            ]
            
            for pattern in degree_patterns:
                matches = re.finditer(pattern, edu_section)
                for match in matches:
                    edu_entry = {}
                    
                    if match.lastindex >= 1:  # Matches degree type
                        edu_entry["degree"] = match.group(1).capitalize()
                    
                    if match.lastindex >= 3:  # Matches field
                        edu_entry["field"] = match.group(3).capitalize()
                    elif match.lastindex == 1:  # Only matched field
                        edu_entry["field"] = match.group(1).capitalize()
                        edu_entry["degree"] = "Degree"
                    
                    if edu_entry:
                        result["education"].append(edu_entry)
        else:
            # Look for degree requirements in full text
            degree_keywords = ["bachelor", "master", "phd", "doctorate", "bs", "ms", "ba", "ma", "degree"]
            edu_fields = ["computer science", "engineering", "business", "information technology", 
                         "data science", "mathematics", "statistics", "economics", "finance"]
            
            for keyword in degree_keywords:
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
    
    # Remove duplicates in lists
    result["skills"]["technical"] = list(set(result["skills"]["technical"]))
    result["skills"]["soft"] = list(set(result["skills"]["soft"]))
    result["responsibilities"] = list(dict.fromkeys(result["responsibilities"]))  # Preserve order
    result["preferred_qualifications"] = list(dict.fromkeys(result["preferred_qualifications"]))
    
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
    
    # Extract all text from resume for fuzzy matching
    resume_full_text = ""
    
    # Add experience descriptions and all text available
    for exp in resume_info.get("experience", []):
        if isinstance(exp, dict):
            # Add all text from experience entries
            for key, value in exp.items():
                if isinstance(value, str):
                    resume_full_text += value + " "
                elif isinstance(value, list):
                    resume_full_text += " ".join(value) + " "
        elif isinstance(exp, str):
            resume_full_text += exp + " "
    
    # Add education descriptions
    for edu in resume_info.get("education", []):
        if isinstance(edu, dict):
            # Add all text from education entries
            for key, value in edu.items():
                if isinstance(value, str):
                    resume_full_text += value + " "
        elif isinstance(edu, str):
            resume_full_text += edu + " "
    
    # Add certifications and other sections
    for cert in resume_info.get("certifications", []):
        if isinstance(cert, str):
            resume_full_text += cert + " "
    
    # Convert to lowercase for easier matching
    resume_full_text = resume_full_text.lower()
    
    # First perform exact skill matching
    matching_tech_skills = resume_tech_skills.intersection(job_tech_skills)
    matching_soft_skills = resume_soft_skills.intersection(job_soft_skills)
    
    # Then try fuzzy matching for skills not found through exact matches
    potentially_matching_tech = set()
    potentially_matching_soft = set()
    
    # Common skill variations for fuzzy matching
    skill_variations = {
        "javascript": ["js", "java script", "javascript"],
        "typescript": ["ts", "type script", "typescript"],
        "python": ["py", "python3", "python 3"],
        "react": ["reactjs", "react.js", "react js"],
        "angular": ["angularjs", "angular.js", "angular js"],
        "node": ["nodejs", "node.js", "node js"],
        "express": ["expressjs", "express.js", "express js"],
        "mongodb": ["mongo", "mongo db"],
        "postgresql": ["postgres", "pgsql", "psql"],
        "mysql": ["sql", "maria", "mariadb"],
        "machine learning": ["ml", "machine-learning"],
        "artificial intelligence": ["ai", "artificial-intelligence"],
        "amazon web services": ["aws"],
        "microsoft azure": ["azure"],
        "google cloud platform": ["gcp", "google cloud"],
        "continuous integration": ["ci", "ci/cd"],
        "continuous deployment": ["cd", "ci/cd"],
        "devops": ["dev ops", "development operations"],
        "frontend": ["front-end", "front end"],
        "backend": ["back-end", "back end"]
    }
    
    # Check remaining job skills for potential matches in resume text
    for job_skill in job_tech_skills:
        # Skip skills we already found exact matches for
        if job_skill in matching_tech_skills:
            continue
            
        # Check for variations
        variation_found = False
        if job_skill in skill_variations:
            for variation in skill_variations[job_skill]:
                if variation in resume_full_text or re.search(r'\b' + re.escape(variation) + r'\b', resume_full_text):
                    potentially_matching_tech.add(job_skill)
                    variation_found = True
                    break
        
        # If no variation match, try word-by-word matching for multi-word skills
        if not variation_found and ' ' in job_skill:
            words = job_skill.split()
            matches = 0
            for word in words:
                if len(word) > 3 and re.search(r'\b' + re.escape(word) + r'\b', resume_full_text):
                    matches += 1
            
            # If at least 75% of words match, consider it a potential match
            if matches / len(words) >= 0.75:
                potentially_matching_tech.add(job_skill)
    
    # Similar approach for soft skills with synonyms
    soft_skill_synonyms = {
        "communication": ["communicate", "articulate", "present", "convey", "correspond"],
        "teamwork": ["collaborate", "team player", "cooperation", "collaborative"],
        "leadership": ["lead", "manage", "direct", "guide", "mentor", "supervise"],
        "problem solving": ["troubleshoot", "resolve", "analytical", "solution", "debug"],
        "critical thinking": ["analyze", "evaluate", "assess", "decision", "judgment"],
        "time management": ["organize", "prioritize", "schedule", "deadline", "timely"],
        "adaptability": ["flexible", "adjust", "versatile", "agile", "receptive"],
        "creativity": ["innovative", "creative", "design", "original", "inventive"],
        "attention to detail": ["meticulous", "thorough", "precise", "accurate", "detail-oriented"]
    }
    
    for job_skill in job_soft_skills:
        if job_skill in matching_soft_skills:
            continue
            
        if job_skill in soft_skill_synonyms:
            for synonym in soft_skill_synonyms[job_skill]:
                if re.search(r'\b' + re.escape(synonym) + r'\b', resume_full_text):
                    potentially_matching_soft.add(job_skill)
                    break
    
    # Create final matching skills lists including both exact and fuzzy matches
    all_matching_tech = list(matching_tech_skills) + list(potentially_matching_tech)
    all_matching_soft = list(matching_soft_skills) + list(potentially_matching_soft)
    
    # Calculate skill match percentage with weighted scores
    # Exact matches count fully, potential matches count as 0.5
    if job_tech_skills or job_soft_skills:  # Avoid division by zero
        total_job_skills = len(job_tech_skills) + len(job_soft_skills)
        weighted_matches = len(matching_tech_skills) + len(matching_soft_skills) + (0.5 * len(potentially_matching_tech)) + (0.5 * len(potentially_matching_soft))
        skills_match = min(100, round((weighted_matches / total_job_skills) * 100)) if total_job_skills > 0 else 50
    else:
        skills_match = 50  # Default if no skills are specified in job
    
    # Missing skills are those not found in exact or fuzzy matching
    missing_tech_skills = job_tech_skills - set(all_matching_tech)
    missing_soft_skills = job_soft_skills - set(all_matching_soft)
    
    # Prepare skills feedback
    matching_skills = all_matching_tech + all_matching_soft
    missing_skills = list(missing_tech_skills) + list(missing_soft_skills)
    
    # Enhanced experience match calculation
    if job_info.get("experience"):
        # Get the minimum years of experience required
        min_years_required = 0
        required_domain = "General"
        
        for exp in job_info.get("experience", []):
            years_text = exp.get("years", "")
            years_match = re.search(r'(\d+)', years_text)
            if years_match:
                years = int(years_match.group(1))
                if years > min_years_required:
                    min_years_required = years
                    required_domain = exp.get("domain", "General")
        
        # Extract years from resume with multiple methods
        resume_years = 0
        
        # Method 1: Look for explicit start/end dates
        for exp in resume_info.get("experience", []):
            if isinstance(exp, dict):
                if "start_date" in exp and "end_date" in exp:
                    start_year_match = re.search(r'(\d{4})', exp.get("start_date", ""))
                    end_year_match = re.search(r'(\d{4})', exp.get("end_date", ""))
                    
                    if start_year_match:
                        start_year = int(start_year_match.group(1))
                        end_year = 2023  # Default current year
                        
                        if "present" in exp.get("end_date", "").lower():
                            end_year = 2023  # Current year for "present"
                        elif end_year_match:
                            end_year = int(end_year_match.group(1))
                        
                        years = end_year - start_year
                        if years > 0:  # Valid year range
                            resume_years += years
        
        # Method 2: Look for years mentioned in experience descriptions
        if resume_years == 0:
            for exp in resume_info.get("experience", []):
                if isinstance(exp, dict) and "description" in exp:
                    years_match = re.search(r'(\d+)[+]?\s+years?', exp["description"].lower())
                    if years_match:
                        years = int(years_match.group(1))
                        resume_years = max(resume_years, years)
                elif isinstance(exp, str):
                    years_match = re.search(r'(\d+)[+]?\s+years?', exp.lower())
                    if years_match:
                        years = int(years_match.group(1))
                        resume_years = max(resume_years, years)
        
        # Method 3: Estimate from number of positions if still zero
        if resume_years == 0 and resume_info.get("experience"):
            resume_years = min(len(resume_info.get("experience")) * 2, 10)  # Assume 2 years per position, cap at 10
        
        # Calculate match score
        if min_years_required > 0:
            experience_match = min(100, round((resume_years / min_years_required) * 100))
        else:
            experience_match = 70  # Default if years not specified
    else:
        experience_match = 70  # Default if no experience requirements
        min_years_required = 0
        resume_years = 0
        required_domain = "General"
    
    # Enhanced education match with better degree level comparison
    if job_info.get("education"):
        # Education level hierarchy
        education_levels = {
            "high school": 1,
            "associate": 2, 
            "associate's": 2,
            "bachelor": 3, 
            "bachelor's": 3,
            "bs": 3, 
            "ba": 3, 
            "b.s.": 3,
            "b.a.": 3,
            "undergraduate": 3,
            "master": 4, 
            "master's": 4,
            "ms": 4, 
            "ma": 4,
            "m.s.": 4,
            "m.a.": 4,
            "graduate": 4,
            "mba": 4,
            "phd": 5, 
            "doctorate": 5,
            "doctoral": 5
        }
        
        # Find highest education level in resume
        resume_edu_level = 0
        resume_edu_field = ""
        
        for edu in resume_info.get("education", []):
            if isinstance(edu, dict) and "degree" in edu:
                degree_text = edu["degree"].lower()
                # Check for level
                for level_name, level_value in education_levels.items():
                    if level_name in degree_text and level_value > resume_edu_level:
                        resume_edu_level = level_value
                        resume_edu_field = edu.get("field", "")
            elif isinstance(edu, str):
                # Check text for education keywords
                for level_name, level_value in education_levels.items():
                    if level_name in edu.lower() and level_value > resume_edu_level:
                        resume_edu_level = level_value
                        # Try to extract field
                        field_match = re.search(r'(?:in|of)\s+([A-Za-z\s]+)', edu)
                        if field_match:
                            resume_edu_field = field_match.group(1).strip()
        
        # Find required education level in job
        job_edu_level = 0
        job_edu_field = ""
        
        for edu in job_info.get("education", []):
            if isinstance(edu, dict) and "degree" in edu:
                degree_text = edu["degree"].lower()
                # Check for level
                for level_name, level_value in education_levels.items():
                    if level_name in degree_text and level_value > job_edu_level:
                        job_edu_level = level_value
                        job_edu_field = edu.get("field", "")
            elif isinstance(edu, str):
                # Check text for education keywords
                for level_name, level_value in education_levels.items():
                    if level_name in edu.lower() and level_value > job_edu_level:
                        job_edu_level = level_value
                        # Try to extract field
                        field_match = re.search(r'(?:in|of)\s+([A-Za-z\s]+)', edu)
                        if field_match:
                            job_edu_field = field_match.group(1).strip()
        
        # Calculate education match with field relevance
        if job_edu_level > 0:
            # Level match (80% of score)
            level_match = min(100, round((resume_edu_level / job_edu_level) * 100))
            
            # Field match (20% of score)
            field_match = 0
            if job_edu_field and resume_edu_field:
                # Direct field match
                if resume_edu_field.lower() == job_edu_field.lower():
                    field_match = 100
                else:
                    # Related fields
                    related_fields = {
                        "computer science": ["information technology", "software engineering", "computer engineering"],
                        "information technology": ["computer science", "information systems", "cybersecurity"],
                        "business": ["management", "marketing", "finance", "accounting", "economics"],
                        "engineering": ["mechanical engineering", "electrical engineering", "civil engineering"]
                    }
                    
                    for key, related in related_fields.items():
                        if job_edu_field.lower() == key and resume_edu_field.lower() in related:
                            field_match = 75
                            break
                        # Also check reverse relationship
                        if resume_edu_field.lower() == key and job_edu_field.lower() in related:
                            field_match = 75
                            break
            else:
                field_match = 50  # No specific field required or provided
            
            # Combined score
            education_match = (level_match * 0.8) + (field_match * 0.2)
        else:
            education_match = 70  # Default if no specific level required
    else:
        education_match = 70  # Default if no education requirements
        job_edu_level = 0
        resume_edu_level = 0
    
    # Calculate overall match with weighted components
    # Adjust weights based on job level
    skill_weight = 0.5
    exp_weight = 0.3
    edu_weight = 0.2
    
    # For senior positions, experience matters more
    if min_years_required >= 5:
        skill_weight = 0.45
        exp_weight = 0.35
        edu_weight = 0.2
    
    # For academic or specialized positions, education matters more
    if job_edu_level >= 4:  # Masters or PhD
        skill_weight = 0.4
        exp_weight = 0.25
        edu_weight = 0.35
    
    overall_match = round((skills_match * skill_weight) + (experience_match * exp_weight) + (education_match * edu_weight))
    
    # Generate better recommendations
    recommendations = []
    
    # Skills recommendations
    if missing_skills:
        if len(missing_tech_skills) > 3:
            recommendations.append(f"Consider gaining experience with these technical skills: {', '.join(list(missing_tech_skills)[:3])}")
        elif missing_tech_skills:
            recommendations.append(f"Add these technical skills to your resume: {', '.join(list(missing_tech_skills))}")
            
        if missing_soft_skills:
            recommendations.append(f"Highlight these soft skills: {', '.join(list(missing_soft_skills)[:3])}")
    
    # Experience recommendations
    if experience_match < 70:
        if resume_years < min_years_required:
            recommendations.append(f"The job requires {min_years_required}+ years of experience in {required_domain}. Your resume shows approximately {resume_years} years.")
        else:
            recommendations.append(f"Make your {resume_years} years of experience in {required_domain} more prominent on your resume.")
    
    # Education recommendations
    if education_match < 70 and job_edu_level > 0:
        edu_level_names = {1: "High School", 2: "Associate's", 3: "Bachelor's", 4: "Master's", 5: "PhD"}
        
        if job_edu_level > resume_edu_level:
            if job_edu_level in edu_level_names:
                if job_edu_field:
                    recommendations.append(f"This position typically requires a {edu_level_names[job_edu_level]} in {job_edu_field}.")
                else:
                    recommendations.append(f"This position typically requires a {edu_level_names[job_edu_level]} degree.")
        else:
            recommendations.append("Make your educational qualifications more prominent on your resume.")
    
    # Generate better feedback summary
    if overall_match >= 85:
        feedback_summary = "Excellent match! Your resume aligns very well with this job posting."
    elif overall_match >= 70:
        feedback_summary = "Strong match! Your resume aligns well with this position, with a few areas for improvement."
    elif overall_match >= 55:
        feedback_summary = "Good match with some areas for improvement. Consider updating your resume to better highlight relevant skills and experience."
    elif overall_match >= 40:
        feedback_summary = "Moderate match. This job requires additional skills or experience not evident in your resume."
    else:
        feedback_summary = "Limited match. Consider roles that better align with your current qualifications or gaining additional experience in this field."
    
    # Return improved match analysis with standardized field names matching the database schema
    return {
        "skills_match_score": skills_match,
        "experience_match_score": experience_match,
        "education_match_score": education_match,
        "overall_match_score": overall_match,
        "matching_skills": matching_skills,
        "missing_skills": missing_skills,
        "recommendations": recommendations if recommendations else ["Ensure your resume clearly highlights all relevant experience"],
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