import os
from sqlalchemy import create_engine, Column, Integer, String, Text, JSON, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json

# Get database URL from environment variable
DATABASE_URL = os.environ.get("DATABASE_URL")

# Ensure we have a database URL
if not DATABASE_URL:
    print("Warning: DATABASE_URL environment variable not set. Database functionality will be disabled.")
    DATABASE_URL = "sqlite:///:memory:"  # Fallback to in-memory SQLite for development

try:
    # Create SQLAlchemy engine with error handling
    engine = create_engine(DATABASE_URL)
    
    # Create a base class for declarative models
    Base = declarative_base()
    
    print("Database connection established successfully")
except Exception as e:
    print(f"Error connecting to database: {e}")
    # Set up a dummy in-memory database as fallback
    DATABASE_URL = "sqlite:///:memory:"
    engine = create_engine(DATABASE_URL)
    Base = declarative_base()

# Define Resume model
class Resume(Base):
    __tablename__ = 'resumes'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    contact_name = Column(String(255))
    contact_email = Column(String(255))
    contact_phone = Column(String(50))
    contact_location = Column(String(255))
    full_text = Column(Text)
    parsed_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.filename,
            'contact_name': self.contact_name,
            'contact_email': self.contact_email,
            'contact_phone': self.contact_phone,
            'contact_location': self.contact_location,
            'parsed_data': self.parsed_data,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

# Define JobMatch model to store match results
class JobMatch(Base):
    __tablename__ = 'job_matches'
    
    id = Column(Integer, primary_key=True)
    resume_id = Column(Integer, nullable=False)
    job_title = Column(String(255))
    job_description = Column(Text)
    skills_match_score = Column(Float)
    experience_match_score = Column(Float)
    education_match_score = Column(Float)
    overall_match_score = Column(Float)
    matching_skills = Column(JSON)
    missing_skills = Column(JSON)
    recommendations = Column(JSON)
    feedback_summary = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'resume_id': self.resume_id,
            'job_title': self.job_title,
            'skills_match_score': self.skills_match_score,
            'experience_match_score': self.experience_match_score,
            'education_match_score': self.education_match_score,
            'overall_match_score': self.overall_match_score,
            'matching_skills': self.matching_skills,
            'missing_skills': self.missing_skills,
            'recommendations': self.recommendations,
            'feedback_summary': self.feedback_summary,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

# Create all tables in the database
Base.metadata.create_all(engine)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db_session():
    """Get a database session"""
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()

def save_resume(filename, full_text, parsed_data):
    """
    Save resume information to the database
    
    Args:
        filename: Original filename
        full_text: Extracted text from resume
        parsed_data: Structured data extracted from resume
        
    Returns:
        Resume object
    """
    db = SessionLocal()
    try:
        # Extract contact info for indexed fields
        contact_name = parsed_data.get('contact_info', {}).get('name', '')
        contact_email = parsed_data.get('contact_info', {}).get('email', '')
        contact_phone = parsed_data.get('contact_info', {}).get('phone', '')
        contact_location = parsed_data.get('contact_info', {}).get('location', '')
        
        # Create resume object
        resume = Resume(
            filename=filename,
            contact_name=contact_name,
            contact_email=contact_email,
            contact_phone=contact_phone,
            contact_location=contact_location,
            full_text=full_text,
            parsed_data=parsed_data
        )
        
        db.add(resume)
        db.commit()
        db.refresh(resume)
        return resume
    except Exception as e:
        db.rollback()
        print(f"Error saving resume to database: {e}")
        return None
    finally:
        db.close()

def save_job_match(resume_id, job_title, job_description, match_analysis):
    """
    Save job match results to the database
    
    Args:
        resume_id: ID of the resume
        job_title: Title of the job
        job_description: Full job description
        match_analysis: Match analysis results
        
    Returns:
        JobMatch object
    """
    db = SessionLocal()
    try:
        # Extract match data
        scores = match_analysis.get('scores', {})
        
        # Create job match object
        job_match = JobMatch(
            resume_id=resume_id,
            job_title=job_title,
            job_description=job_description,
            skills_match_score=scores.get('skills_match', 0),
            experience_match_score=scores.get('experience_match', 0),
            education_match_score=scores.get('education_match', 0),
            overall_match_score=scores.get('overall_match', 0),
            matching_skills=match_analysis.get('matching_skills', []),
            missing_skills=match_analysis.get('missing_skills', []),
            recommendations=match_analysis.get('recommendations', []),
            feedback_summary=match_analysis.get('feedback_summary', '')
        )
        
        db.add(job_match)
        db.commit()
        db.refresh(job_match)
        return job_match
    except Exception as e:
        db.rollback()
        print(f"Error saving job match to database: {e}")
        return None
    finally:
        db.close()

def get_all_resumes():
    """
    Get all resumes from the database
    
    Returns:
        List of Resume objects or empty list if error
    """
    try:
        db = SessionLocal()
        try:
            resumes = db.query(Resume).all()
            return resumes
        except Exception as e:
            print(f"Error fetching resumes from database: {e}")
            return []
        finally:
            db.close()
    except Exception as e:
        print(f"Error creating database session: {e}")
        return []

def get_resume_by_id(resume_id):
    """
    Get resume by ID
    
    Args:
        resume_id: ID of the resume
        
    Returns:
        Resume object or None if error
    """
    try:
        db = SessionLocal()
        try:
            resume = db.query(Resume).filter(Resume.id == resume_id).first()
            return resume
        except Exception as e:
            print(f"Error fetching resume by ID from database: {e}")
            return None
        finally:
            db.close()
    except Exception as e:
        print(f"Error creating database session: {e}")
        return None

def get_job_matches_by_resume_id(resume_id):
    """
    Get all job matches for a resume
    
    Args:
        resume_id: ID of the resume
        
    Returns:
        List of JobMatch objects or empty list if error
    """
    try:
        db = SessionLocal()
        try:
            matches = db.query(JobMatch).filter(JobMatch.resume_id == resume_id).all()
            return matches
        except Exception as e:
            print(f"Error fetching job matches from database: {e}")
            return []
        finally:
            db.close()
    except Exception as e:
        print(f"Error creating database session: {e}")
        return []