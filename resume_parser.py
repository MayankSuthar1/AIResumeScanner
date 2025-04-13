import re
import os
import spacy
import io
import subprocess
from PyPDF2 import PdfReader
import docx2txt
import pytesseract
from pdf2image import convert_from_path, convert_from_bytes
from PIL import Image
import tempfile
from ai_helper import extract_resume_info
# from database import save_resume
from platform import system
import fitz  # PyMuPDF

if system() == 'Windows':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path if needed
elif system() == 'Darwin':
    pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'
else:
    pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'  # Update this path for non-Windows systems

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
        # Using PyMuPDF (fitz) for more accurate text detection
        pdf_document = fitz.open(pdf_path)
        
        # Check text content and image content
        text_lengths = []
        image_counts = []
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            page_text = page.get_text()
            text_lengths.append(len(page_text.strip()))
            
            # Count images on the page
            image_list = page.get_images(full=True)
            image_counts.append(len(image_list))
        
        pdf_document.close()
        
        # If there are images and very little text, likely a scanned document
        if sum(text_lengths) < 300 and sum(image_counts) > 0:
            return True
            
        # If text is present but might be incomplete (common with resumes)
        if 100 <= sum(text_lengths) < 1000:
            # This is a gray area - might be partially machine readable
            # Let's return True to apply OCR for better results
            return True
            
        return False
    except Exception as e:
        print(f"Error checking PDF type: {e}")
        try:
            # Fall back to PyPDF2 if fitz fails
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text() or ""
                    text += page_text
                    
            # If text length is very short compared to usual resume length, it's likely a scanned document
            if len(text.strip()) < 200:
                return True
                
            return False
        except:
            return True  # Assume scanned if we can't check properly

def enhance_ocr_quality(image):
    """
    Enhance image quality for better OCR results
    """
    try:
        # Convert to grayscale
        gray_image = image.convert('L')
        
        # Increase contrast
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(gray_image)
        enhanced_image = enhancer.enhance(1.5)  # Increase contrast by 50%
        
        # Optionally apply thresholding for better text detection
        # This converts the image to pure black and white
        threshold = 200
        enhanced_image = enhanced_image.point(lambda p: p > threshold and 255)
        
        return enhanced_image
    except Exception as e:
        print(f"Image enhancement failed: {e}")
        return image  # Return original if enhancement fails

def ocr_pdf(pdf_path, dpi=300):
    """
    Perform OCR on PDF that is identified as a scanned document
    with improved image processing and quality
    """
    print("Performing OCR on scanned PDF document...")
    
    try:
        # Create a temporary directory to store the images
        with tempfile.TemporaryDirectory() as temp_dir:
            # Convert PDF pages to images with higher DPI for better quality
            images = convert_from_path(pdf_path, dpi=dpi)
            
            full_text = ""
            for i, image in enumerate(images):
                # Enhance image quality
                enhanced_image = enhance_ocr_quality(image)
                
                # Save each image temporarily
                img_path = os.path.join(temp_dir, f'page_{i}.png')
                enhanced_image.save(img_path, 'PNG')
                
                # Perform OCR on the image with improved configuration
                text = pytesseract.image_to_string(
                    Image.open(img_path), 
                    lang='eng',  # English language
                    config='--psm 6'  # Assume a single uniform block of text
                )
                
                full_text += text + "\n\n"
            
            return full_text
    except Exception as e:
        print(f"OCR failed: {e}")
        return ""

def extract_text_from_pdf_with_pymupdf(pdf_path):
    """
    Extract text from PDF using PyMuPDF (fitz) which often has better extraction capabilities
    """
    try:
        text = ""
        pdf_document = fitz.open(pdf_path)
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            # Extract text with the raw option which preserves more formatting
            page_text = page.get_text("text")
            text += page_text + "\n"
            
        pdf_document.close()
        return text
    except Exception as e:
        print(f"PyMuPDF extraction failed: {e}")
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
        print(f"Processing PDF file: {file_path}")
        
        # Step 1: Try PyMuPDF extraction first (often best quality)
        pymupdf_text = extract_text_from_pdf_with_pymupdf(file_path)
        
        # Step 2: Check if it's a scanned document that needs OCR
        is_scanned = check_if_pdf_is_scanned(file_path)
        
        # If PyMuPDF got good text and it's not detected as scanned, use that
        if pymupdf_text and len(pymupdf_text.strip()) > 500 and not is_scanned:
            print("Successfully extracted text using PyMuPDF")
            return pymupdf_text
        
        # Step 3: If scanned or insufficient text, try OCR with enhanced quality
        if is_scanned or len(pymupdf_text.strip()) < 500:
            print("PDF appears to be scanned or has limited machine-readable text. Using OCR...")
            ocr_text = ocr_pdf(file_path, dpi=400)  # Higher DPI for better quality
            
            # If OCR produced good text
            if ocr_text and len(ocr_text.strip()) > 300:
                print("Successfully extracted text using enhanced OCR")
                
                # If we have both PyMuPDF text and OCR text, combine them intelligently
                if pymupdf_text and len(pymupdf_text.strip()) > 100:
                    # Use the longer text, but prioritize OCR for scanned documents
                    if is_scanned or len(ocr_text) > len(pymupdf_text):
                        print("Using OCR text (higher quality for this document)")
                        return ocr_text
                    else:
                        print("Using PyMuPDF text (higher quality for this document)")
                        return pymupdf_text
                return ocr_text
        
        # Step 4: Try other PDF extraction methods as fallbacks
        print("Trying alternative extraction methods...")
        
        # Try regular PDF extraction with PyPDF2
        pypdf2_text = ""
        try:
            with open(file_path, 'rb') as file:
                reader = PdfReader(file)
                for page in reader.pages:
                    pypdf2_text += (page.extract_text() or "") + "\n"
                    
            if pypdf2_text and len(pypdf2_text.strip()) > len(pymupdf_text.strip()):
                print("PyPDF2 extraction provided better results")
                return pypdf2_text
        except Exception as e:
            print(f"PyPDF2 extraction failed: {e}")
        
        # Try pdfminer
        pdfminer_text = try_pdfminer_extraction(file_path)
        if pdfminer_text and len(pdfminer_text.strip()) > max(len(pymupdf_text.strip()), len(pypdf2_text.strip())):
            print("PDFMiner extraction provided better results")
            return pdfminer_text
        
        # Return the best text we have so far
        for text in [ocr_pdf(file_path, dpi=600), pymupdf_text, pypdf2_text, pdfminer_text]:
            if text and len(text.strip()) > 100:
                return text
        
        # If all else failed
        return "Failed to extract meaningful text from PDF file"
    
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
        # db_resume = save_resume(filename, text, parsed_data)
        # resume_id = db_resume.id if db_resume else None
        
        # Return structured information
        return {
            "filename": filename,
            "status": "success",
            "raw_text": text,
            "parsed_data": parsed_data,
            # "resume_id": resume_id  # Include database ID for reference
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
        # db_resume = save_resume(filename, text, fallback_data)
        # resume_id = db_resume.id if db_resume else None
        
        # Return structured information
        return {
            "filename": filename,
            "status": "success", 
            "raw_text": text,
            "parsed_data": fallback_data,
            # "resume_id": resume_id
        }
