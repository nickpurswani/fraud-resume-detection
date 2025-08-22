"""
Utility functions for the Fraudulent Candidate Detection Tool
"""

import re
import os
import json
import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import dateparser
import pandas as pd
import numpy as np
from pathlib import Path
import phonenumbers
from email_validator import validate_email, EmailNotValidError
import PyPDF2
import pdfplumber
from docx import Document
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz, process
import spacy
from sentence_transformers import SentenceTransformer
import textstat

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextExtractor:
    """Extract text from various file formats"""

    @staticmethod
    def extract_from_pdf(file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                # Try pdfplumber first (better for complex layouts)
                try:
                    with pdfplumber.open(file) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                except Exception as e:
                    logger.warning(f"pdfplumber failed: {e}, trying PyPDF2")
                    # Fallback to PyPDF2
                    file.seek(0)
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
            return ""

    @staticmethod
    def extract_from_docx(file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(file_path)
            text = []
            for paragraph in doc.paragraphs:
                text.append(paragraph.text)
            return '\n'.join(text)
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {e}")
            return ""

    @staticmethod
    def extract_from_txt(file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error extracting text from TXT {file_path}: {e}")
            return ""

    @classmethod
    def extract_text(cls, file_path: str) -> str:
        """Extract text based on file extension"""
        file_extension = Path(file_path).suffix.lower()

        if file_extension == '.pdf':
            return cls.extract_from_pdf(file_path)
        elif file_extension in ['.docx', '.doc']:
            return cls.extract_from_docx(file_path)
        elif file_extension == '.txt':
            return cls.extract_from_txt(file_path)
        else:
            logger.warning(f"Unsupported file format: {file_extension}")
            return ""

class TextProcessor:
    """Process and clean text data"""

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""

        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', '', text)

        # Remove multiple punctuation
        text = re.sub(r'[\.]{2,}', '.', text)
        text = re.sub(r'[\,]{2,}', ',', text)

        return text.strip()

    @staticmethod
    def extract_emails(text: str) -> List[str]:
        """Extract email addresses from text"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)

        # Validate emails
        valid_emails = []
        for email in emails:
            try:
                validate_email(email)
                valid_emails.append(email.lower())
            except EmailNotValidError:
                continue

        return list(set(valid_emails))

    @staticmethod
    def extract_phone_numbers(text: str) -> List[str]:
        """Extract phone numbers from text"""
        phone_numbers = []

        # Common phone patterns
        patterns = [
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            r'\(\d{3}\)\s*\d{3}[-.]?\d{4}',
            r'\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    # Parse and format phone number
                    parsed = phonenumbers.parse(match, None)
                    if phonenumbers.is_valid_number(parsed):
                        formatted = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
                        phone_numbers.append(formatted)
                except:
                    continue

        return list(set(phone_numbers))

    @staticmethod
    def extract_dates(text: str) -> List[datetime]:
        """Extract dates from text"""
        # Common date patterns
        date_patterns = [
            r'\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b',
            r'\b\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}\b',
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b'
        ]

        dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    parsed_date = dateparser.parse(match)
                    if parsed_date:
                        dates.append(parsed_date)
                except:
                    continue

        return sorted(list(set(dates)))

class SimilarityCalculator:
    """Calculate various similarity metrics"""

    @staticmethod
    def cosine_similarity(text1: str, text2: str, model: Optional[SentenceTransformer] = None) -> float:
        """Calculate cosine similarity using sentence transformers"""
        if not model:
            model = SentenceTransformer('all-MiniLM-L6-v2')

        embeddings = model.encode([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float(similarity)

    @staticmethod
    def fuzzy_similarity(text1: str, text2: str) -> Dict[str, float]:
        """Calculate fuzzy similarity metrics"""
        return {
            'ratio': fuzz.ratio(text1, text2) / 100.0,
            'partial_ratio': fuzz.partial_ratio(text1, text2) / 100.0,
            'token_sort_ratio': fuzz.token_sort_ratio(text1, text2) / 100.0,
            'token_set_ratio': fuzz.token_set_ratio(text1, text2) / 100.0
        }

    @staticmethod
    def sequence_similarity(text1: str, text2: str) -> float:
        """Calculate sequence similarity using difflib"""
        return SequenceMatcher(None, text1, text2).ratio()

class DateValidator:
    """Validate and analyze dates"""

    @staticmethod
    def is_valid_date_range(start_date: datetime, end_date: datetime) -> bool:
        """Check if date range is valid"""
        return start_date <= end_date

    @staticmethod
    def calculate_duration(start_date: datetime, end_date: datetime) -> Dict[str, float]:
        """Calculate duration between dates"""
        if not DateValidator.is_valid_date_range(start_date, end_date):
            return {'days': 0, 'months': 0, 'years': 0}

        duration = end_date - start_date
        days = duration.days
        months = days / 30.44  # Average days per month
        years = days / 365.25  # Average days per year

        return {
            'days': days,
            'months': round(months, 2),
            'years': round(years, 2)
        }

    @staticmethod
    def detect_overlapping_periods(periods: List[Tuple[datetime, datetime]]) -> List[Tuple[int, int]]:
        """Detect overlapping time periods"""
        overlaps = []
        for i, (start1, end1) in enumerate(periods):
            for j, (start2, end2) in enumerate(periods[i+1:], i+1):
                if start1 <= end2 and start2 <= end1:
                    overlaps.append((i, j))
        return overlaps

class LocationValidator:
    """Validate locations and detect inconsistencies"""

    @staticmethod
    def normalize_location(location: str) -> str:
        """Normalize location string"""
        if not location:
            return ""

        # Remove extra spaces and convert to title case
        location = re.sub(r'\s+', ' ', location.strip()).title()

        # Common abbreviations
        abbreviations = {
            'Ny': 'NY', 'Ca': 'CA', 'Tx': 'TX', 'Fl': 'FL',
            'Us': 'US', 'Usa': 'USA', 'Uk': 'UK'
        }

        for abbrev, full in abbreviations.items():
            location = re.sub(rf'\b{abbrev}\b', full, location)

        return location

    @staticmethod
    def extract_locations(text: str) -> List[str]:
        """Extract location mentions from text"""
        # Simple location patterns (can be enhanced with NER)
        location_patterns = [
            r'\b[A-Z][a-z]+,\s*[A-Z]{2}\b',  # City, State
            r'\b[A-Z][a-z]+,\s*[A-Z][a-z]+\b',  # City, Country
        ]

        locations = []
        for pattern in location_patterns:
            matches = re.findall(pattern, text)
            locations.extend([LocationValidator.normalize_location(match) for match in matches])

        return list(set(locations))

class SkillExtractor:
    """Extract and validate skills from text"""

    def __init__(self, skills_database_path: Optional[str] = None):
        self.skills_database = self._load_skills_database(skills_database_path)

    def _load_skills_database(self, path: Optional[str]) -> List[str]:
        """Load skills database"""
        if path and os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    return data.get('skills', [])
            except Exception as e:
                logger.warning(f"Could not load skills database: {e}")

        # Default skills list
        return [
            'python', 'java', 'javascript', 'c++', 'sql', 'html', 'css',
            'react', 'angular', 'vue', 'django', 'flask', 'spring',
            'machine learning', 'deep learning', 'data science', 'ai',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes',
            'git', 'agile', 'scrum', 'project management'
        ]

    def extract_skills(self, text: str, threshold: float = 0.8) -> List[str]:
        """Extract skills from text"""
        text_lower = text.lower()
        found_skills = []

        for skill in self.skills_database:
            # Direct match
            if skill.lower() in text_lower:
                found_skills.append(skill)
            else:
                # Fuzzy match
                best_match = process.extractOne(skill, text_lower.split())
                if best_match and best_match[1] >= threshold * 100:
                    found_skills.append(skill)

        return list(set(found_skills))

class ReadabilityAnalyzer:
    """Analyze text readability and complexity"""

    @staticmethod
    def analyze_readability(text: str) -> Dict[str, float]:
        """Analyze text readability using various metrics"""
        return {
            'flesch_reading_ease': textstat.flesch_reading_ease(text),
            'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
            'automated_readability_index': textstat.automated_readability_index(text),
            'coleman_liau_index': textstat.coleman_liau_index(text),
            'gunning_fog': textstat.gunning_fog(text),
            'avg_sentence_length': textstat.avg_sentence_length(text),
            'avg_syllables_per_word': textstat.avg_syllables_per_word(text)
        }

class CacheManager:
    """Simple file-based cache manager"""

    def __init__(self, cache_dir: str = '.cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_key(self, key: str) -> str:
        """Generate cache key hash"""
        return hashlib.md5(key.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        cache_key = self._get_cache_key(key)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    # Check if cache is expired
                    if datetime.fromisoformat(data['timestamp']) + timedelta(hours=1) > datetime.now():
                        return data['value']
            except Exception as e:
                logger.warning(f"Cache read error: {e}")

        return None

    def set(self, key: str, value: Any) -> None:
        """Set cached value"""
        cache_key = self._get_cache_key(key)
        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    'value': value,
                    'timestamp': datetime.now().isoformat()
                }, f)
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

def create_sample_data():
    """Create sample resumes and job descriptions for testing"""
    sample_resume = """
John Smith
Software Engineer
Email: john.smith@email.com
Phone: (555) 123-4567
Location: San Francisco, CA

EXPERIENCE:
Senior Software Engineer | Google | 2020 - Present
- Led a team of 10 engineers in developing scalable web applications
- Implemented machine learning algorithms to improve user experience
- Increased system performance by 300% through optimization

Software Engineer | Facebook | 2018 - 2020
- Developed full-stack applications using React and Node.js
- Collaborated with cross-functional teams on product development

EDUCATION:
Master of Science in Computer Science | Stanford University | 2018
Bachelor of Science in Computer Science | UC Berkeley | 2016

SKILLS:
Python, Java, JavaScript, React, Node.js, Machine Learning, AWS, Docker
"""

    sample_job_description = """
Senior Software Engineer - AI/ML Focus

We are seeking an experienced Senior Software Engineer with expertise in artificial intelligence and machine learning to join our growing team.

Requirements:
- 5+ years of software development experience
- Strong proficiency in Python and JavaScript
- Experience with machine learning frameworks
- Knowledge of cloud platforms (AWS, GCP, Azure)
- Bachelor's degree in Computer Science or related field

Responsibilities:
- Design and implement ML-powered features
- Collaborate with data science team
- Lead technical projects and mentor junior developers
- Optimize system performance and scalability
"""

    return sample_resume, sample_job_description

def setup_logging(log_file: str = 'fraud_detection.log', level: str = 'INFO'):
    """Setup logging configuration"""
    log_level = getattr(logging, level.upper())

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def validate_file_upload(file_path: str, max_size_mb: int = 10) -> Dict[str, Any]:
    """Validate uploaded file"""
    if not os.path.exists(file_path):
        return {'valid': False, 'error': 'File does not exist'}

    # Check file size
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        return {'valid': False, 'error': f'File size ({file_size_mb:.2f}MB) exceeds limit ({max_size_mb}MB)'}

    # Check file extension
    supported_extensions = ['.pdf', '.docx', '.doc', '.txt']
    file_extension = Path(file_path).suffix.lower()
    if file_extension not in supported_extensions:
        return {'valid': False, 'error': f'Unsupported file format: {file_extension}'}

    return {'valid': True, 'size_mb': file_size_mb, 'extension': file_extension}

def calculate_confidence_score(scores: Dict[str, float]) -> float:
    """Calculate overall confidence score"""
    if not scores:
        return 0.0

    weights = {
        'experience_consistency': 0.25,
        'education_validity': 0.20,
        'skills_alignment': 0.20,
        'timeline_consistency': 0.15,
        'content_originality': 0.20
    }

    weighted_score = 0.0
    total_weight = 0.0

    for metric, score in scores.items():
        weight = weights.get(metric, 0.1)
        weighted_score += score * weight
        total_weight += weight

    return weighted_score / total_weight if total_weight > 0 else 0.0

if __name__ == "__main__":
    # Test utilities
    sample_resume, sample_job_desc = create_sample_data()

    # Test text processing
    processor = TextProcessor()
    emails = processor.extract_emails(sample_resume)
    phones = processor.extract_phone_numbers(sample_resume)

    print(f"Extracted emails: {emails}")
    print(f"Extracted phones: {phones}")

    # Test similarity calculation
    similarity_calc = SimilarityCalculator()
    similarity = similarity_calc.fuzzy_similarity(sample_resume[:100], sample_job_desc[:100])
    print(f"Similarity scores: {similarity}")
