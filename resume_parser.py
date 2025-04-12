import re
import os
import spacy
import io
import subprocess
from PyPDF2 import PdfReader
import docx2txt
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import tempfile
from ai_helper import extract_resume_info
from database import save_resume

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    # If model not found, download it
    import subprocess
    subprocess.call(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

def check_if_pdf_is_scanned(pdf_path):
    """
    Check if a PDF contains mostly images (scanned document) or searchable text
    """
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text
    except Exception as e:
        print(f"Error checking PDF type: {e}")
        return True  # Assume scanned if we can't check properly
    
    # If text length is very short compared to usual resume length, it's likely a scanned document
    if len(text.strip()) < 100:
        return True
    
    return False

def ocr_pdf(pdf_path):
    """
    Perform OCR on PDF that is identified as a scanned document
    """
    print("Performing OCR on scanned PDF document...")
    
    try:
        # Create a temporary directory to store the images
        with tempfile.TemporaryDirectory() as temp_dir:
            # Convert PDF pages to images
            images = convert_from_path(pdf_path)
            
            full_text = ""
            for i, image in enumerate(images):
                # Save each image temporarily
                img_path = os.path.join(temp_dir, f'page_{i}.png')
                image.save(img_path, 'PNG')
                
                # Perform OCR on the image
                text = pytesseract.image_to_string(Image.open(img_path))
                full_text += text + "\n\n"
            
            return full_text
    except Exception as e:
        print(f"OCR failed: {e}")
        return ""

def try_pdfminer_extraction(pdf_path):
    """
    Try extracting text using pdfminer if PyPDF2 doesn't work well
    """
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract
        return pdfminer_extract(pdf_path)
    except Exception as e:
        print(f"pdfminer extraction failed: {e}")
        return ""

def extract_text_from_file(file_path):
    """
    Extract text from various file formats (PDF, DOCX, DOC) with enhanced capabilities
    including OCR for scanned documents
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.pdf':
        # First check if it's a scanned PDF
        if check_if_pdf_is_scanned(file_path):
            # Try OCR if it's a scanned document
            text = ocr_pdf(file_path)
            if text and len(text.strip()) > 100:
                return text
        
        # Try regular PDF extraction with PyPDF2
        text = ""
        try:
            with open(file_path, 'rb') as file:
                reader = PdfReader(file)
                for page in reader.pages:
                    text += (page.extract_text() or "") + "\n"
        except Exception as e:
            print(f"PyPDF2 extraction failed: {e}")
        
        # If PyPDF2 extraction produced little text, try pdfminer
        if not text or len(text.strip()) < 100:
            pdfminer_text = try_pdfminer_extraction(file_path)
            if pdfminer_text and len(pdfminer_text.strip()) > len(text.strip()):
                text = pdfminer_text
        
        # If still not enough text, try OCR even if not initially detected as scanned
        if not text or len(text.strip()) < 100:
            ocr_text = ocr_pdf(file_path)
            if ocr_text and len(ocr_text.strip()) > len(text.strip()):
                text = ocr_text
        
        return text
    
    elif file_ext == '.docx':
        try:
            return docx2txt.process(file_path)
        except Exception as e:
            print(f"DOCX extraction failed: {e}")
            # Try python-docx as fallback
            try:
                from docx import Document
                doc = Document(file_path)
                return "\n".join([paragraph.text for paragraph in doc.paragraphs])
            except Exception as e2:
                print(f"python-docx extraction failed: {e2}")
                return "Failed to extract text from DOCX file"
    
    elif file_ext == '.doc':
        # Try multiple approaches for .doc files
        text = ""
        
        # Try textract
        try:
            import textract
            text = textract.process(file_path).decode('utf-8')
            if text and len(text.strip()) > 100:
                return text
        except Exception as e:
            print(f"textract extraction failed: {e}")
        
        # Try antiword if available
        try:
            result = subprocess.run(['antiword', file_path], capture_output=True, text=True)
            if result.returncode == 0 and result.stdout:
                return result.stdout
        except Exception as e:
            print(f"antiword extraction failed: {e}")
        
        # If all else fails, suggest conversion
        if not text:
            return "Unable to process .doc file. Please convert to .docx or .pdf"
        return text
    
    else:
        return "Unsupported file format"

def extract_contact_info(text):
    """
    Extract email, phone number, and other contact information
    """
    contact_info = {}
    
    # Email extraction
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    if emails:
        contact_info['email'] = emails[0]
    
    # Phone extraction
    phone_pattern = r'(\+\d{1,3}[-.\s]?)?(\d{3}[-.\s]?\d{3}[-.\s]?\d{4}|\(\d{3}\)[-.\s]?\d{3}[-.\s]?\d{4})'
    phones = re.findall(phone_pattern, text)
    if phones:
        # Join the parts of the matched phone and clean it up
        contact_info['phone'] = ''.join(phones[0]).strip()
    
    # Simple name extraction - more advanced methods would be better in production
    # This is a very simplistic approach
    lines = text.split('\n')
    non_empty_lines = [line.strip() for line in lines if line.strip()]
    if non_empty_lines:
        # Assuming the name is in the first few lines
        for line in non_empty_lines[:3]:
            # Exclude lines that seem to be emails, phones, or addresses
            if not re.search(email_pattern, line) and not re.search(phone_pattern, line) and len(line.split()) <= 4:
                contact_info['name'] = line
                break
    
    return contact_info

def extract_education(text):
    """
    Extract education information from resume text
    """
    # Common education keywords
    education_keywords = [
        'education', 'degree', 'university', 'college', 'school', 'bachelor', 
        'master', 'phd', 'doctorate', 'certification', 'diploma'
    ]
    
    education_info = []
    lines = text.split('\n')
    in_education_section = False
    
    # Define a pattern for education entries (degree, institution, year)
    edu_pattern = r'(Bachelor|Master|MBA|PhD|BSc|MSc|B\.Tech|M\.Tech|B\.E|M\.E|B\.A|M\.A)'
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Check if this line marks the beginning of education section
        if any(keyword.lower() in line.lower() for keyword in education_keywords) and len(line) < 50:
            in_education_section = True
            if line.lower() != 'education':  # If it's not just the section title
                education_info.append(line)
            continue
        
        # Check for the end of education section (next section header)
        if in_education_section and line and line[0].isupper() and line.endswith(':'):
            in_education_section = False
            continue
        
        # If we're in education section or line matches education pattern, add it
        if in_education_section and line:
            education_info.append(line)
        elif re.search(edu_pattern, line) and len(line) < 100:
            education_info.append(line)
    
    # If no structured education section is found, try to extract using NLP
    if not education_info:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == 'ORG' and any(keyword.lower() in ent.text.lower() for keyword in ['university', 'college', 'school']):
                # Find the surrounding context
                context_start = max(0, ent.start_char - 50)
                context_end = min(len(text), ent.end_char + 50)
                context = text[context_start:context_end]
                education_info.append(context.strip())
    
    # Remove duplicates and clean up
    clean_education = []
    for item in education_info:
        item = re.sub(r'\s+', ' ', item).strip()
        if item and item not in clean_education and len(item) > 3:
            clean_education.append(item)
    
    return clean_education[:5]  # Limit to 5 entries to avoid over-extraction

def extract_experience(text):
    """
    Extract work experience information from resume text
    """
    # Common experience section keywords
    experience_keywords = [
        'experience', 'employment', 'work history', 'professional experience',
        'career', 'job history', 'positions'
    ]
    
    experience_info = []
    lines = text.split('\n')
    in_experience_section = False
    current_entry = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Check if this line marks the beginning of experience section
        if any(keyword.lower() in line.lower() for keyword in experience_keywords) and len(line) < 50:
            in_experience_section = True
            continue
        
        # Check for the end of experience section (next section header)
        if in_experience_section and line and line[0].isupper() and line.endswith(':'):
            if any(keyword.lower() in line.lower() for keyword in ['education', 'skills', 'projects']):
                in_experience_section = False
                if current_entry:
                    experience_info.append(' '.join(current_entry))
                    current_entry = []
                continue
        
        # Process lines in the experience section
        if in_experience_section and line:
            # If line looks like a new job entry (company name or job title)
            if (re.search(r'\b(19|20)\d{2}\b', line) or 
                len(line) < 50 and any(char.isupper() for char in line)):
                if current_entry:
                    experience_info.append(' '.join(current_entry))
                    current_entry = []
                current_entry.append(line)
            elif current_entry:
                current_entry.append(line)
    
    # Add the last entry if there is one
    if current_entry:
        experience_info.append(' '.join(current_entry))
    
    # If no structured experience section found, try to extract using dates and organizations
    if not experience_info:
        # Look for date ranges which often indicate job periods
        date_pattern = r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)[\s,]+\d{4}\s*[-–—]\s*(Present|Current|Now|(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)[\s,]+\d{4})'
        matches = re.finditer(date_pattern, text, re.IGNORECASE)
        
        for match in matches:
            start_pos = max(0, match.start() - 100)
            end_pos = min(len(text), match.end() + 200)
            context = text[start_pos:end_pos].strip()
            experience_info.append(context)
    
    # Clean up the experience entries
    clean_experience = []
    for item in experience_info:
        item = re.sub(r'\s+', ' ', item).strip()
        if item and len(item) > 10 and item not in clean_experience:
            # Truncate very long entries
            if len(item) > 300:
                item = item[:300] + "..."
            clean_experience.append(item)
    
    return clean_experience[:5]  # Limit to 5 entries to avoid over-extraction

def extract_skills(text):
    """
    Extract skills from resume text
    """
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
    
    all_skills = technical_skills + soft_skills
    skills_found = set()
    
    # 1. Look for skills section
    skills_section_pattern = r'(?i)skills[\s:]*\n(.*?)(?:\n\n|\n[A-Z]|\Z)'
    skills_match = re.search(skills_section_pattern, text, re.DOTALL)
    
    if skills_match:
        skills_text = skills_match.group(1)
        # Split by common delimiters
        for delimiter in [',', '•', '·', '-', '|', '\n']:
            if delimiter in skills_text:
                skills_list = [s.strip() for s in skills_text.split(delimiter) if s.strip()]
                if skills_list:
                    for skill in skills_list:
                        if 3 <= len(skill) <= 30:  # Reasonable skill name length
                            skills_found.add(skill.lower())
    
    # 2. Search for known skills in the entire text
    for skill in all_skills:
        pattern = r'\b' + skill + r'\b'
        if re.search(pattern, text.lower()):
            skills_found.add(skill)
    
    # 3. Use NLP to extract technical entities
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == 'PRODUCT' and 3 <= len(ent.text) <= 20:
            # Check if likely to be a technical product/skill
            lowercase_text = ent.text.lower()
            if any(tech_term in lowercase_text for tech_term in ['software', 'framework', 'library', 'language', 'platform', 'tool']):
                skills_found.add(ent.text.lower())
    
    # Convert to list and sort alphabetically
    skills_list = sorted(list(skills_found))
    
    # Limit to top N skills to avoid noise
    return skills_list[:30]

def parse_resume(file_path, filename):
    """
    Parse a resume file and extract structured information using AI
    """
    # Extract text from the file
    text = extract_text_from_file(file_path)
    
    if not text or text.startswith("Unable to process"):
        return {
            "filename": filename,
            "status": "error",
            "error_message": text if text else "Failed to extract text from file"
        }
    
    try:
        # Use Gemini AI to extract structured information
        print("Using Gemini AI to extract resume information...")
        parsed_data = extract_resume_info(text)
        
        # Save to database
        db_resume = save_resume(filename, text, parsed_data)
        resume_id = db_resume.id if db_resume else None
        
        # Return structured information
        return {
            "filename": filename,
            "status": "success",
            "raw_text": text,
            "parsed_data": parsed_data,
            "resume_id": resume_id  # Include database ID for reference
        }
    except Exception as e:
        print(f"AI extraction failed: {e}, falling back to traditional parsing")
        # Fall back to traditional parsing if AI extraction fails
        
        # Extract contact information
        contact_info = extract_contact_info(text)
        
        # Extract education information
        education = extract_education(text)
        
        # Extract experience information
        experience = extract_experience(text)
        
        # Extract skills
        skills = extract_skills(text)
        
        # Create a structured data format similar to the AI output
        fallback_data = {
            "contact_info": contact_info,
            "education": [{"institution": edu} for edu in education],
            "experience": [{"description": exp} for exp in experience],
            "skills": {"technical": skills, "soft": []}
        }
        
        # Save fallback data to database
        db_resume = save_resume(filename, text, fallback_data)
        resume_id = db_resume.id if db_resume else None
        
        # Return structured information
        return {
            "filename": filename,
            "status": "success", 
            "raw_text": text,
            "parsed_data": fallback_data,
            "resume_id": resume_id
        }
