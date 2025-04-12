import re
import spacy
from collections import Counter

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    # If model not found, download it
    import subprocess
    subprocess.call(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

def extract_job_requirements(job_description):
    """
    Extract key requirements from a job description
    """
    # Process the job description with spaCy
    doc = nlp(job_description)
    
    # Initialize containers for different requirement types
    requirements = {
        "skills": set(),
        "experience": set(),
        "education": set()
    }
    
    # Common technical skills
    technical_skills = [
        'python', 'java', 'javascript', 'typescript', 'c\\+\\+', 'c#', 'ruby', 'php', 'html', 'css',
        'sql', 'nosql', 'mongodb', 'mysql', 'postgresql', 'oracle', 'aws', 'azure', 'gcp',
        'react', 'angular', 'vue', 'node', 'express', 'django', 'flask', 'spring', 'bootstrap',
        'docker', 'kubernetes', 'jenkins', 'git', 'agile', 'scrum', 'jira', 'devops', 'ci/cd',
        'machine learning', 'deep learning', 'data science', 'artificial intelligence', 'ai', 'nlp',
        'tensorflow', 'pytorch', 'keras', 'pandas', 'numpy', 'scikit-learn', 'matplotlib',
        'power bi', 'tableau', 'excel', 'spss', 'sas', 'r', 'hadoop', 'spark', 'kafka', 'airflow',
        'rest api', 'graphql', 'microservices', 'soa', 'oauth', 'saml', 'ldap',
        'linux', 'unix', 'windows', 'macos', 'bash', 'powershell', 'networking', 'tcp/ip'
    ]
    
    # Common soft skills
    soft_skills = [
        'communication', 'teamwork', 'leadership', 'problem solving', 'critical thinking',
        'time management', 'project management', 'creativity', 'adaptability', 'flexibility',
        'organization', 'prioritization', 'attention to detail', 'analytical skills',
        'interpersonal skills', 'conflict resolution', 'negotiation', 'presentation',
        'decision making', 'stress management', 'mentoring', 'coaching'
    ]
    
    # Extract skills
    for skill in technical_skills + soft_skills:
        pattern = r'\b' + skill + r'\b'
        if re.search(pattern, job_description.lower()):
            requirements["skills"].add(skill)
    
    # Extract skills from requirements sections
    requirements_pattern = r'(?i)(requirements|qualifications|skills required|what you\'ll need).*?(?=\n\n|\Z)'
    requirements_match = re.search(requirements_pattern, job_description, re.DOTALL)
    
    if requirements_match:
        requirements_text = requirements_match.group(0)
        # Look for bullet points or numbered lists
        bullets = re.findall(r'[•\-*\d+\.]\s*(.*?)(?=\n[•\-*\d+\.]|\n\n|\Z)', requirements_text, re.DOTALL)
        
        for bullet in bullets:
            bullet = bullet.strip()
            
            # Check for experience requirements
            experience_match = re.search(r'(\d+)[+]?\s+(year|yr)[s]?\s+(?:of)?\s+experience', bullet, re.IGNORECASE)
            if experience_match:
                years = experience_match.group(1)
                requirements["experience"].add(f"{years}+ years experience")
            
            # Check for education requirements
            education_keywords = ['degree', 'bachelor', 'master', 'phd', 'mba', 'bs', 'ms', 'certification']
            if any(keyword in bullet.lower() for keyword in education_keywords):
                requirements["education"].add(bullet)
            
            # Extract potential skills from the bullet point
            bullet_doc = nlp(bullet)
            for token in bullet_doc:
                if token.pos_ in ('NOUN', 'PROPN') and len(token.text) > 2:
                    potential_skill = token.text.lower()
                    # Check if it's a known skill
                    if potential_skill in technical_skills or potential_skill in soft_skills:
                        requirements["skills"].add(potential_skill)
    
    # Extract experience requirements more broadly
    experience_patterns = [
        r'(\d+)[+]?\s+(year|yr)[s]?\s+(?:of)?\s+experience',
        r'experience:?\s+(\d+)[+]?\s+(year|yr)[s]?',
        r'minimum\s+of\s+(\d+)\s+(year|yr)[s]?'
    ]
    
    for pattern in experience_patterns:
        matches = re.finditer(pattern, job_description, re.IGNORECASE)
        for match in matches:
            years = match.group(1)
            # Get some context to understand what kind of experience
            start_pos = max(0, match.start() - 50)
            end_pos = min(len(job_description), match.end() + 50)
            context = job_description[start_pos:end_pos]
            
            # Try to identify the type of experience
            context_doc = nlp(context)
            for ent in context_doc.ents:
                if ent.label_ in ('ORG', 'PRODUCT') and start_pos <= ent.start_char <= end_pos:
                    requirements["experience"].add(f"{years}+ years experience with {ent.text}")
                    break
            else:
                requirements["experience"].add(f"{years}+ years of professional experience")
    
    # Extract education requirements
    education_patterns = [
        r'(bachelor|master|phd|doctorate|mba|bs|ms|ba|ma)[\''s]*\s+(degree|in|of)',
        r'degree\s+in\s+([a-z\s]+)',
        r'(certification|diploma)\s+in\s+([a-z\s]+)'
    ]
    
    for pattern in education_patterns:
        matches = re.finditer(pattern, job_description, re.IGNORECASE)
        for match in matches:
            start_pos = max(0, match.start() - 20)
            end_pos = min(len(job_description), match.end() + 50)
            context = job_description[start_pos:end_pos]
            education_requirement = re.sub(r'\s+', ' ', context).strip()
            if len(education_requirement) < 100:  # Avoid very long contexts
                requirements["education"].add(education_requirement)
    
    # Convert sets to lists
    for key in requirements:
        requirements[key] = list(requirements[key])
    
    return requirements
