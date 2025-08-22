"""
Fraudulent Candidate Detection Tool

A comprehensive tool for detecting fraudulent patterns in resumes and candidate profiles
using AI, NLP, and data verification techniques.
"""

__version__ = "1.0.0"
__author__ = "Fraud Detection Team"
__email__ = "contact@frauddetection.com"

# Import main modules for easier access
from .fraud_detector import FraudDetector
from .nlp_analyzer import NLPAnalyzer
from .linkedin_verifier import LinkedInVerifier
from .fit_scorer import FitScorer
from .report_generator import ReportGenerator
from .utils import (
    TextExtractor,
    TextProcessor,
    SimilarityCalculator,
    DateValidator,
    LocationValidator,
    SkillExtractor,
    ReadabilityAnalyzer
)

__all__ = [
    'FraudDetector',
    'NLPAnalyzer',
    'LinkedInVerifier',
    'FitScorer',
    'ReportGenerator',
    'TextExtractor',
    'TextProcessor',
    'SimilarityCalculator',
    'DateValidator',
    'LocationValidator',
    'SkillExtractor',
    'ReadabilityAnalyzer'
]
