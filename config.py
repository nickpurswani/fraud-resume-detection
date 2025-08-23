"""
Configuration file for Fraudulent Candidate Detection Tool
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for the fraud detection tool"""

    # API Keys (use environment variables)
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
    LINKEDIN_API_KEY = os.getenv('LINKEDIN_API_KEY', '')
    BRIGHTDATA_API_KEY = os.getenv('BRIGHTDATA_API_KEY', '')

    # Gemini AI Configuration
    GEMINI_MODEL = "gemini-1.5-flash"
    GEMINI_MAX_TOKENS = 2000
    GEMINI_TEMPERATURE = 0.3
    GEMINI_SAFETY_SETTINGS = {
        'HARASSMENT': 'BLOCK_MEDIUM_AND_ABOVE',
        'HATE_SPEECH': 'BLOCK_MEDIUM_AND_ABOVE',
        'SEXUALLY_EXPLICIT': 'BLOCK_MEDIUM_AND_ABOVE',
        'DANGEROUS_CONTENT': 'BLOCK_MEDIUM_AND_ABOVE',
    }

    # NLP Model Configurations
    SPACY_MODEL = "en_core_web_sm"
    SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"

    # Fraud Detection Thresholds
    FRAUD_THRESHOLDS = {
        'experience_inconsistency': 0.7,
        'education_mismatch': 0.6,
        'skill_experience_gap': 0.8,
        'plagiarism_similarity': 0.85,
        'timeline_inconsistency': 0.7,
        'location_discrepancy': 0.6,
        'salary_inflation': 0.8,
        'duplicate_content': 0.9
    }

    # Resume Analysis Settings
    MIN_EXPERIENCE_YEARS = 0
    MAX_EXPERIENCE_YEARS = 50
    SUSPICIOUS_KEYWORDS = [
        'rockstar', 'ninja', 'guru', 'wizard',
        'expert in everything', 'all technologies',
        'best in class', 'world-class'
    ]

    # Job Title Hierarchies (for career progression analysis)
    JOB_HIERARCHIES = {
        'software': [
            'intern', 'junior developer', 'developer', 'software engineer',
            'senior developer', 'senior software engineer', 'lead developer',
            'principal engineer', 'architect', 'engineering manager',
            'director of engineering', 'vp of engineering', 'cto'
        ],
        'data': [
            'data intern', 'junior data analyst', 'data analyst',
            'senior data analyst', 'data scientist', 'senior data scientist',
            'lead data scientist', 'principal data scientist',
            'data science manager', 'director of data science'
        ],
        'marketing': [
            'marketing intern', 'marketing assistant', 'marketing coordinator',
            'marketing specialist', 'marketing manager', 'senior marketing manager',
            'marketing director', 'vp of marketing', 'cmo'
        ]
    }

    # File Handling
    SUPPORTED_RESUME_FORMATS = ['.pdf', '.docx', '.txt', '.doc']
    MAX_FILE_SIZE_MB = 10
    UPLOAD_FOLDER = 'uploads'

    # Report Configuration
    REPORT_TEMPLATES_DIR = 'templates'
    OUTPUT_DIR = 'reports'

    # LinkedIn Verification Settings
    LINKEDIN_BASE_URL = "https://api.linkedin.com/v2"
    LINKEDIN_TIMEOUT = 30

    # Bright Data LinkedIn Configuration
    BRIGHTDATA_BASE_URL = "https://api.brightdata.com/datasets/v3"
    BRIGHTDATA_DATASET_ID = "gd_l1viktl72bvl7bjuj0"
    BRIGHTDATA_TIMEOUT = 60
    BRIGHTDATA_INCLUDE_ERRORS = True

    # Database Configuration (if needed)
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///fraud_detection.db')

    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = 'fraud_detection.log'

    # Cache Configuration
    CACHE_ENABLED = True
    CACHE_TIMEOUT = 3600  # 1 hour

    # Rate Limiting
    API_RATE_LIMIT = {
        'gemini': 60,  # requests per minute
        'linkedin': 100,  # requests per hour
        'brightdata': 50  # requests per hour
    }

    # Common Job Descriptions (for plagiarism detection)
    COMMON_JOB_DESCRIPTIONS_PATH = 'data/common_job_descriptions.json'

    # Skills Database
    SKILLS_DATABASE_PATH = 'data/skills_database.json'

    # University Database
    UNIVERSITIES_DATABASE_PATH = 'data/universities.json'

    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """Validate configuration and return status"""
        issues = []
        warnings = []

        # Check API keys
        if not cls.GEMINI_API_KEY:
            warnings.append("Gemini API key not provided - some features will be limited")

        if not cls.LINKEDIN_API_KEY:
            warnings.append("LinkedIn API key not provided - profile verification disabled")

        if not cls.BRIGHTDATA_API_KEY:
            warnings.append("Bright Data API key not provided - advanced LinkedIn data collection disabled")

        # Check thresholds
        for key, value in cls.FRAUD_THRESHOLDS.items():
            if not 0 <= value <= 1:
                issues.append(f"Invalid threshold for {key}: {value} (should be 0-1)")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }

    @classmethod
    def get_gemini_config(cls) -> Dict[str, Any]:
        """Get Gemini AI configuration"""
        return {
            'api_key': cls.GEMINI_API_KEY,
            'model': cls.GEMINI_MODEL,
            'max_tokens': cls.GEMINI_MAX_TOKENS,
            'temperature': cls.GEMINI_TEMPERATURE,
            'safety_settings': cls.GEMINI_SAFETY_SETTINGS
        }

    @classmethod
    def get_fraud_threshold(cls, fraud_type: str) -> float:
        """Get fraud threshold for specific type"""
        return cls.FRAUD_THRESHOLDS.get(fraud_type, 0.7)

    @classmethod
    def get_brightdata_config(cls) -> Dict[str, Any]:
        """Get Bright Data configuration"""
        return {
            'api_key': cls.BRIGHTDATA_API_KEY,
            'base_url': cls.BRIGHTDATA_BASE_URL,
            'dataset_id': cls.BRIGHTDATA_DATASET_ID,
            'timeout': cls.BRIGHTDATA_TIMEOUT,
            'include_errors': cls.BRIGHTDATA_INCLUDE_ERRORS
        }

# Create a global config instance
config = Config()
