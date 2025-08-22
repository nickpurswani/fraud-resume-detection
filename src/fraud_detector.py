"""
Main Fraud Detection Engine for Fraudulent Candidate Detection Tool

This module contains the core fraud detection logic that orchestrates various
analysis components to identify fraudulent patterns in resumes and candidate profiles.
"""

import logging
import json
import os
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import dateparser
from dataclasses import dataclass, asdict
from enum import Enum
import statistics

from .nlp_analyzer import NLPAnalyzer
from .gemini_analyzer import GeminiAnalyzer, GeminiAnalysisResult, AnalysisType
from .utils import (
    TextProcessor, SimilarityCalculator, DateValidator,
    LocationValidator, SkillExtractor, calculate_confidence_score
)

# Configure logging
logger = logging.getLogger(__name__)

class FraudType(Enum):
    """Enumeration of different fraud types"""
    EXPERIENCE_INCONSISTENCY = "experience_inconsistency"
    EDUCATION_MISMATCH = "education_mismatch"
    SKILL_EXPERIENCE_GAP = "skill_experience_gap"
    PLAGIARISM = "plagiarism"
    TIMELINE_INCONSISTENCY = "timeline_inconsistency"
    LOCATION_DISCREPANCY = "location_discrepancy"
    SALARY_INFLATION = "salary_inflation"
    DUPLICATE_CONTENT = "duplicate_content"
    CAREER_PROGRESSION_ANOMALY = "career_progression_anomaly"
    SUSPICIOUS_KEYWORDS = "suspicious_keywords"

class RiskLevel(Enum):
    """Risk level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class FraudFlag:
    """Represents a detected fraud flag"""
    fraud_type: FraudType
    risk_level: RiskLevel
    confidence: float
    description: str
    evidence: Dict[str, Any]
    severity_score: float
    recommendation: str

@dataclass
class ExperienceEntry:
    """Represents a work experience entry"""
    title: str
    company: str
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    duration_months: Optional[float]
    description: str
    location: Optional[str]
    salary: Optional[str]

@dataclass
class EducationEntry:
    """Represents an education entry"""
    degree: str
    institution: str
    graduation_year: Optional[int]
    gpa: Optional[float]
    location: Optional[str]
    field_of_study: Optional[str]

class FraudDetector:
    """Main fraud detection engine"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize fraud detector

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.nlp_analyzer = NLPAnalyzer()
        self.text_processor = TextProcessor()
        self.similarity_calculator = SimilarityCalculator()
        self.date_validator = DateValidator()
        self.location_validator = LocationValidator()
        self.skill_extractor = SkillExtractor()

        # Initialize Gemini AI analyzer if API key available
        gemini_api_key = (self.config.get('gemini_api_key', '') or
                         os.getenv('GEMINI_API_KEY', ''))
        if gemini_api_key:
            try:
                self.gemini_analyzer = GeminiAnalyzer(config=self.config)
                logger.info("Gemini AI analyzer initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize Gemini analyzer: {e}")
                self.gemini_analyzer = None
        else:
            logger.warning("No Gemini API key provided - advanced AI analysis will be limited. Set GEMINI_API_KEY environment variable.")
            self.gemini_analyzer = None

        # Load fraud detection thresholds
        self.thresholds = self.config.get('fraud_thresholds', {
            'experience_inconsistency': 0.7,
            'education_mismatch': 0.6,
            'skill_experience_gap': 0.8,
            'plagiarism_similarity': 0.85,
            'timeline_inconsistency': 0.7,
            'location_discrepancy': 0.6,
            'salary_inflation': 0.8,
            'duplicate_content': 0.9
        })

        # Load job hierarchies for career progression analysis
        self.job_hierarchies = self.config.get('job_hierarchies', self._get_default_hierarchies())

        # Suspicious keywords that might indicate fraud
        self.suspicious_keywords = self.config.get('suspicious_keywords', [
            'rockstar', 'ninja', 'guru', 'wizard', 'expert in everything',
            'all technologies', 'best in class', 'world-class', 'top performer',
            'award-winning', 'industry leader', 'thought leader'
        ])

    def _get_default_hierarchies(self) -> Dict[str, List[str]]:
        """Get default job hierarchies for different fields"""
        return {
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
            ],
            'sales': [
                'sales intern', 'sales representative', 'account executive',
                'senior account executive', 'sales manager', 'regional sales manager',
                'sales director', 'vp of sales', 'chief revenue officer'
            ],
            'finance': [
                'financial analyst', 'senior financial analyst', 'finance manager',
                'senior finance manager', 'finance director', 'vp of finance', 'cfo'
            ]
        }

    def analyze_resume(self, resume_text: str, job_description: Optional[str] = None,
                      reference_resumes: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Main method to analyze a resume for fraudulent patterns

        Args:
            resume_text: The resume text to analyze
            job_description: Optional job description for fit analysis
            reference_resumes: Optional list of reference resumes for plagiarism detection

        Returns:
            Comprehensive fraud analysis results
        """
        logger.info("Starting comprehensive fraud analysis...")

        analysis_results = {
            'timestamp': datetime.now().isoformat(),
            'resume_length': len(resume_text),
            'fraud_flags': [],
            'risk_assessment': {},
            'detailed_analysis': {},
            'recommendations': [],
            'confidence_scores': {}
        }

        try:
            # Step 1: NLP Analysis
            logger.info("Performing NLP analysis...")
            nlp_results = self.nlp_analyzer.generate_nlp_report(resume_text, reference_resumes)
            analysis_results['detailed_analysis']['nlp'] = nlp_results

            # Step 2: Extract structured information
            logger.info("Extracting structured information...")
            structured_info = self._extract_structured_info(resume_text)
            analysis_results['detailed_analysis']['structured_info'] = structured_info

            # Step 3: Fraud Detection Analysis
            logger.info("Running fraud detection checks...")

            # Check for experience inconsistencies
            exp_flags = self._check_experience_inconsistencies(structured_info['experiences'])
            analysis_results['fraud_flags'].extend(exp_flags)

            # Check for education mismatches
            edu_flags = self._check_education_inconsistencies(structured_info['education'])
            analysis_results['fraud_flags'].extend(edu_flags)

            # Check for skill-experience mismatches
            skill_flags = self._check_skill_experience_mismatch(
                structured_info['skills'], structured_info['experiences']
            )
            analysis_results['fraud_flags'].extend(skill_flags)

            # Check timeline inconsistencies
            timeline_flags = self._check_timeline_inconsistencies(
                structured_info['experiences'], structured_info['education']
            )
            analysis_results['fraud_flags'].extend(timeline_flags)

            # Check for plagiarism if reference resumes provided
            if reference_resumes:
                plagiarism_flags = self._check_plagiarism(resume_text, reference_resumes)
                analysis_results['fraud_flags'].extend(plagiarism_flags)

            # Check for suspicious keywords
            keyword_flags = self._check_suspicious_keywords(resume_text)
            analysis_results['fraud_flags'].extend(keyword_flags)

            # Check career progression anomalies
            progression_flags = self._check_career_progression(structured_info['experiences'])
            analysis_results['fraud_flags'].extend(progression_flags)

            # Check location discrepancies
            location_flags = self._check_location_discrepancies(structured_info)
            analysis_results['fraud_flags'].extend(location_flags)

            # Step 4: Advanced AI Analysis with Gemini (if available)
            if self.gemini_analyzer:
                logger.info("Running advanced Gemini AI analysis...")
                try:
                    gemini_result = self.gemini_analyzer.comprehensive_fraud_analysis(
                        resume_text, job_description
                    )
                    analysis_results['gemini_analysis'] = {
                        'risk_level': gemini_result.risk_level.value,
                        'confidence': gemini_result.confidence,
                        'findings': gemini_result.findings,
                        'evidence': gemini_result.evidence,
                        'recommendations': gemini_result.recommendations,
                        'processing_time': gemini_result.processing_time
                    }

                    # Enhance fraud flags with Gemini insights
                    self._integrate_gemini_findings(analysis_results['fraud_flags'], gemini_result)

                except Exception as e:
                    logger.warning(f"Gemini AI analysis failed: {e}")
                    analysis_results['gemini_analysis'] = {'error': str(e)}

            # Step 5: Risk Assessment
            logger.info("Calculating risk assessment...")
            analysis_results['risk_assessment'] = self._calculate_risk_assessment(
                analysis_results['fraud_flags']
            )

            # Add top-level fraud score and risk level for easier access
            analysis_results['fraud_score'] = analysis_results['risk_assessment']['risk_score']
            analysis_results['risk_level'] = analysis_results['risk_assessment']['overall_risk']

            # Step 6: Generate confidence scores
            analysis_results['confidence_scores'] = self._calculate_confidence_scores(
                analysis_results['fraud_flags'], structured_info
            )

            # Step 7: Generate recommendations
            analysis_results['recommendations'] = self._generate_recommendations(
                analysis_results['fraud_flags']
            )

            analysis_results['status'] = 'success'
            logger.info("Fraud analysis completed successfully")

        except Exception as e:
            logger.error(f"Error during fraud analysis: {e}")
            analysis_results['status'] = 'error'
            analysis_results['error_message'] = str(e)

        return analysis_results

    def _extract_structured_info(self, resume_text: str) -> Dict[str, Any]:
        """Extract structured information from resume"""
        logger.info("Extracting structured information...")

        # Use NLP analyzer to get basic info
        personal_info = self.nlp_analyzer.extract_personal_info(resume_text)
        work_exp = self.nlp_analyzer.extract_work_experience(resume_text)
        education = self.nlp_analyzer.extract_education(resume_text)
        skills = self.nlp_analyzer.extract_skills(resume_text)

        # Convert to structured format
        experiences = []
        for exp in work_exp:
            experiences.append(self._parse_experience_entry(exp, resume_text))

        education_entries = []
        for edu in education:
            education_entries.append(self._parse_education_entry(edu, resume_text))

        return {
            'personal_info': personal_info,
            'experiences': experiences,
            'education': education_entries,
            'skills': skills,
            'raw_nlp_data': {
                'work_experience': work_exp,
                'education': education,
                'skills': skills
            }
        }

    def _parse_experience_entry(self, exp_data: Dict[str, str], full_text: str) -> ExperienceEntry:
        """Parse experience entry into structured format"""
        title = exp_data.get('title', '').strip()
        company = exp_data.get('company', '').strip()
        duration = exp_data.get('duration', '').strip()

        # Parse dates
        start_date, end_date = self._parse_date_range(duration)
        duration_months = None

        if start_date and end_date:
            delta = end_date - start_date
            duration_months = delta.days / 30.44

        # Extract description (simplified - would need more sophisticated parsing)
        description = ""

        return ExperienceEntry(
            title=title,
            company=company,
            start_date=start_date,
            end_date=end_date,
            duration_months=duration_months,
            description=description,
            location=None,
            salary=None
        )

    def _parse_education_entry(self, edu_data: Dict[str, str], full_text: str) -> EducationEntry:
        """Parse education entry into structured format"""
        degree = edu_data.get('degree', '').strip()
        year = edu_data.get('year', '').strip()

        graduation_year = None
        if year and year.isdigit():
            graduation_year = int(year)

        return EducationEntry(
            degree=degree,
            institution="",  # Would need more sophisticated extraction
            graduation_year=graduation_year,
            gpa=None,
            location=None,
            field_of_study=None
        )

    def _parse_date_range(self, duration_str: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Parse date range string into start and end dates"""
        if not duration_str:
            return None, None

        # Common patterns for date ranges
        patterns = [
            r'(\d{4})\s*[-–]\s*(\d{4})',
            r'(\d{4})\s*[-–]\s*(present|current)',
            r'([A-Za-z]+\s+\d{4})\s*[-–]\s*([A-Za-z]+\s+\d{4})',
            r'([A-Za-z]+\s+\d{4})\s*[-–]\s*(present|current)'
        ]

        for pattern in patterns:
            match = re.search(pattern, duration_str, re.IGNORECASE)
            if match:
                start_str, end_str = match.groups()

                start_date = dateparser.parse(start_str)
                if end_str.lower() in ['present', 'current']:
                    end_date = datetime.now()
                else:
                    end_date = dateparser.parse(end_str)

                return start_date, end_date

        return None, None

    def _check_experience_inconsistencies(self, experiences: List[ExperienceEntry]) -> List[FraudFlag]:
        """Check for inconsistencies in work experience"""
        flags = []

        for i, exp in enumerate(experiences):
            # Check for unrealistic duration
            if exp.duration_months and exp.duration_months > 600:  # > 50 years
                flags.append(FraudFlag(
                    fraud_type=FraudType.EXPERIENCE_INCONSISTENCY,
                    risk_level=RiskLevel.HIGH,
                    confidence=0.9,
                    description=f"Unrealistic work duration: {exp.duration_months:.1f} months at {exp.company}",
                    evidence={
                        'company': exp.company,
                        'duration_months': exp.duration_months,
                        'title': exp.title
                    },
                    severity_score=0.8,
                    recommendation="Verify employment dates and duration with the company"
                ))

            # Check for unrealistic job titles for experience level
            if self._is_unrealistic_title_for_duration(exp.title, exp.duration_months):
                flags.append(FraudFlag(
                    fraud_type=FraudType.EXPERIENCE_INCONSISTENCY,
                    risk_level=RiskLevel.MEDIUM,
                    confidence=0.7,
                    description=f"Job title '{exp.title}' may be unrealistic for duration {exp.duration_months:.1f} months",
                    evidence={
                        'title': exp.title,
                        'duration_months': exp.duration_months,
                        'company': exp.company
                    },
                    severity_score=0.6,
                    recommendation="Verify job responsibilities and actual title with employer"
                ))

        return flags

    def _is_unrealistic_title_for_duration(self, title: str, duration_months: Optional[float]) -> bool:
        """Check if job title is unrealistic for the given duration"""
        if not title or not duration_months:
            return False

        title_lower = title.lower()

        # Senior roles with very short duration (< 12 months)
        senior_keywords = ['senior', 'lead', 'principal', 'director', 'vp', 'chief', 'head of']
        if any(keyword in title_lower for keyword in senior_keywords) and duration_months < 12:
            return True

        # C-level positions with very short duration (< 24 months)
        c_level_keywords = ['ceo', 'cto', 'cfo', 'cmo', 'coo']
        if any(keyword in title_lower for keyword in c_level_keywords) and duration_months < 24:
            return True

        return False

    def _check_education_inconsistencies(self, education: List[EducationEntry]) -> List[FraudFlag]:
        """Check for inconsistencies in education"""
        flags = []

        for edu in education:
            # Check for unrealistic graduation years
            if edu.graduation_year:
                current_year = datetime.now().year
                if edu.graduation_year > current_year + 5:  # Future dates
                    flags.append(FraudFlag(
                        fraud_type=FraudType.EDUCATION_MISMATCH,
                        risk_level=RiskLevel.HIGH,
                        confidence=0.95,
                        description=f"Future graduation year: {edu.graduation_year}",
                        evidence={
                            'graduation_year': edu.graduation_year,
                            'degree': edu.degree,
                            'current_year': current_year
                        },
                        severity_score=0.9,
                        recommendation="Verify graduation date with educational institution"
                    ))
                elif edu.graduation_year < 1950:  # Too old
                    flags.append(FraudFlag(
                        fraud_type=FraudType.EDUCATION_MISMATCH,
                        risk_level=RiskLevel.MEDIUM,
                        confidence=0.8,
                        description=f"Unusually old graduation year: {edu.graduation_year}",
                        evidence={
                            'graduation_year': edu.graduation_year,
                            'degree': edu.degree
                        },
                        severity_score=0.6,
                        recommendation="Verify graduation date with educational institution"
                    ))

        # Check for degree progression inconsistencies
        graduation_years = [edu.graduation_year for edu in education if edu.graduation_year]
        if len(graduation_years) > 1:
            graduation_years.sort()

            # Check if higher degrees come before lower degrees
            degree_levels = {'bachelor': 1, 'master': 2, 'phd': 3, 'doctorate': 3}

            for i, edu in enumerate(education):
                if edu.graduation_year and edu.degree:
                    current_level = self._get_degree_level(edu.degree)
                    for other_edu in education:
                        if (other_edu.graduation_year and other_edu.degree and
                            other_edu.graduation_year < edu.graduation_year):
                            other_level = self._get_degree_level(other_edu.degree)
                            if other_level > current_level:
                                flags.append(FraudFlag(
                                    fraud_type=FraudType.EDUCATION_MISMATCH,
                                    risk_level=RiskLevel.MEDIUM,
                                    confidence=0.7,
                                    description="Degree progression appears inconsistent",
                                    evidence={
                                        'earlier_degree': other_edu.degree,
                                        'earlier_year': other_edu.graduation_year,
                                        'later_degree': edu.degree,
                                        'later_year': edu.graduation_year
                                    },
                                    severity_score=0.5,
                                    recommendation="Verify educational timeline and degree requirements"
                                ))

        return flags

    def _get_degree_level(self, degree: str) -> int:
        """Get numerical level of degree (1=bachelor, 2=master, 3=phd)"""
        degree_lower = degree.lower()

        if any(word in degree_lower for word in ['phd', 'ph.d', 'doctorate']):
            return 3
        elif any(word in degree_lower for word in ['master', 'mba', 'm.s', 'm.a']):
            return 2
        elif any(word in degree_lower for word in ['bachelor', 'b.s', 'b.a']):
            return 1
        else:
            return 0  # Unknown or other

    def _check_skill_experience_mismatch(self, skills: Dict[str, List[str]],
                                       experiences: List[ExperienceEntry]) -> List[FraudFlag]:
        """Check for mismatches between claimed skills and experience"""
        flags = []

        # Calculate total experience duration
        total_months = sum(exp.duration_months for exp in experiences if exp.duration_months)
        total_years = total_months / 12 if total_months else 0

        # Check for advanced skills with minimal experience
        advanced_skills = skills.get('technical', []) + skills.get('programming', [])

        if total_years < 2 and len(advanced_skills) > 10:
            flags.append(FraudFlag(
                fraud_type=FraudType.SKILL_EXPERIENCE_GAP,
                risk_level=RiskLevel.MEDIUM,
                confidence=0.6,
                description=f"Many advanced skills ({len(advanced_skills)}) claimed with limited experience ({total_years:.1f} years)",
                evidence={
                    'total_experience_years': total_years,
                    'advanced_skills_count': len(advanced_skills),
                    'skills': advanced_skills
                },
                severity_score=0.5,
                recommendation="Verify technical skills through practical assessment or portfolio review"
            ))

        # Check for technology skills that don't match job history
        tech_skills = skills.get('programming', []) + skills.get('tools', [])
        job_titles = [exp.title.lower() for exp in experiences]

        # If claiming programming skills but no technical job titles
        if tech_skills and not any(
            keyword in ' '.join(job_titles)
            for keyword in ['engineer', 'developer', 'programmer', 'analyst', 'architect']
        ):
            flags.append(FraudFlag(
                fraud_type=FraudType.SKILL_EXPERIENCE_GAP,
                risk_level=RiskLevel.MEDIUM,
                confidence=0.7,
                description="Technical skills claimed but no technical job titles in experience",
                evidence={
                    'technical_skills': tech_skills,
                    'job_titles': [exp.title for exp in experiences]
                },
                severity_score=0.6,
                recommendation="Verify technical experience and skills through code review or technical interview"
            ))

        return flags

    def _check_timeline_inconsistencies(self, experiences: List[ExperienceEntry],
                                      education: List[EducationEntry]) -> List[FraudFlag]:
        """Check for timeline inconsistencies"""
        flags = []

        # Check for overlapping work experiences
        work_periods = []
        for exp in experiences:
            if exp.start_date and exp.end_date:
                work_periods.append((exp.start_date, exp.end_date, exp))

        overlaps = self.date_validator.detect_overlapping_periods(
            [(start, end) for start, end, _ in work_periods]
        )

        for i, j in overlaps:
            exp1 = work_periods[i][2]
            exp2 = work_periods[j][2]
            flags.append(FraudFlag(
                fraud_type=FraudType.TIMELINE_INCONSISTENCY,
                risk_level=RiskLevel.HIGH,
                confidence=0.9,
                description=f"Overlapping work periods: {exp1.company} and {exp2.company}",
                evidence={
                    'job1': {'company': exp1.company, 'title': exp1.title, 'start': exp1.start_date, 'end': exp1.end_date},
                    'job2': {'company': exp2.company, 'title': exp2.title, 'start': exp2.start_date, 'end': exp2.end_date}
                },
                severity_score=0.8,
                recommendation="Verify employment dates with both employers"
            ))

        # Check if work experience starts before education ends
        for edu in education:
            if edu.graduation_year:
                grad_date = datetime(edu.graduation_year, 6, 1)  # Assume June graduation

                for exp in experiences:
                    if exp.start_date and exp.start_date < grad_date:
                        # Check if it's reasonable (internships, part-time work)
                        if exp.duration_months and exp.duration_months > 24:  # Full-time equivalent
                            flags.append(FraudFlag(
                                fraud_type=FraudType.TIMELINE_INCONSISTENCY,
                                risk_level=RiskLevel.MEDIUM,
                                confidence=0.6,
                                description=f"Extended work experience before graduation: {exp.company} started before {edu.graduation_year}",
                                evidence={
                                    'work_start': exp.start_date,
                                    'graduation_year': edu.graduation_year,
                                    'company': exp.company,
                                    'degree': edu.degree
                                },
                                severity_score=0.4,
                                recommendation="Verify if this was part-time work or internship during studies"
                            ))

        return flags

    def _check_plagiarism(self, resume_text: str, reference_resumes: List[str]) -> List[FraudFlag]:
        """Check for plagiarism against reference resumes"""
        flags = []

        plagiarism_results = self.nlp_analyzer.detect_plagiarism(
            resume_text, reference_resumes,
            threshold=self.thresholds.get('plagiarism_similarity', 0.85)
        )

        if plagiarism_results['is_plagiarized']:
            risk_level = RiskLevel.CRITICAL if plagiarism_results['max_similarity'] > 0.95 else RiskLevel.HIGH

            flags.append(FraudFlag(
                fraud_type=FraudType.PLAGIARISM,
                risk_level=risk_level,
                confidence=plagiarism_results['max_similarity'],
                description=f"High similarity to existing resume (similarity: {plagiarism_results['max_similarity']:.2f})",
                evidence={
                    'similarity_score': plagiarism_results['max_similarity'],
                    'suspicious_matches': plagiarism_results['suspicious_matches'],
                    'similar_chunks_count': len(plagiarism_results['similar_chunks'])
                },
                severity_score=plagiarism_results['max_similarity'],
                recommendation="Conduct in-person interview to verify authenticity of experience and skills"
            ))

        return flags

    def _check_suspicious_keywords(self, resume_text: str) -> List[FraudFlag]:
        """Check for suspicious keywords that might indicate fraud"""
        flags = []

        text_lower = resume_text.lower()
        found_keywords = []

        for keyword in self.suspicious_keywords:
            if keyword.lower() in text_lower:
                found_keywords.append(keyword)

        if found_keywords:
            flags.append(FraudFlag(
                fraud_type=FraudType.SUSPICIOUS_KEYWORDS,
                risk_level=RiskLevel.MEDIUM,
                confidence=0.6,
                description=f"Suspicious keywords found: {', '.join(found_keywords)}",
                evidence={
                    'keywords': found_keywords,
                    'keyword_count': len(found_keywords)
                },
                severity_score=0.3,
                recommendation="Look for concrete examples and achievements rather than superlative claims"
            ))

        return flags

    def _check_career_progression(self, experiences: List[ExperienceEntry]) -> List[FraudFlag]:
        """Check for unusual career progression patterns"""
        flags = []

        if len(experiences) < 2:
            return flags

        # Sort experiences by start date
        sorted_experiences = sorted(
            [exp for exp in experiences if exp.start_date],
            key=lambda x: x.start_date
        )

        # Analyze progression
        for i in range(1, len(sorted_experiences)):
            prev_exp = sorted_experiences[i-1]
            curr_exp = sorted_experiences[i]

            # Check for unrealistic jumps (e.g., intern to VP)
            prev_level = self._get_job_level(prev_exp.title)
            curr_level = self._get_job_level(curr_exp.title)

            if curr_level - prev_level > 3:  # More than 3 levels jump
                flags.append(FraudFlag(
                    fraud_type=FraudType.CAREER_PROGRESSION_ANOMALY,
                    risk_level=RiskLevel.MEDIUM,
                    confidence=0.7,
                    description=f"Unusually rapid career progression: {prev_exp.title} to {curr_exp.title}",
                    evidence={
                        'previous_title': prev_exp.title,
                        'current_title': curr_exp.title,
                        'level_jump': curr_level - prev_level,
                        'time_gap_months': (curr_exp.start_date - prev_exp.start_date).days / 30.44 if curr_exp.start_date and prev_exp.start_date else None
                    },
                    severity_score=0.5,
                    recommendation="Verify career progression and responsibilities with previous employers"
                ))

        return flags

    def _get_job_level(self, title: str) -> int:
        """Get numerical job level (0=intern, 10=C-level)"""
        if not title:
            return 0

        title_lower = title.lower()

        # C-level
        if any(word in title_lower for word in ['ceo', 'cto', 'cfo', 'cmo', 'coo', 'chief']):
            return 10

        # VP level
        elif 'vp' in title_lower or 'vice president' in title_lower:
            return 9

        # Director level
        elif 'director' in title_lower:
            return 8

        # Manager level
        elif 'manager' in title_lower:
            return 6

        # Lead/Principal level
        elif any(word in title_lower for word in ['lead', 'principal', 'architect']):
            return 5

        # Senior level
        elif 'senior' in title_lower:
            return 4

        # Regular level
        elif any(word in title_lower for word in ['engineer', 'developer', 'analyst', 'specialist']):
            return 3

        # Junior level
        elif 'junior' in title_lower:
            return 2

        # Intern level
        elif 'intern' in title_lower:
            return 1

        else:
            return 3  # Default to regular level

    def _check_location_discrepancies(self, structured_info: Dict[str, Any]) -> List[FraudFlag]:
        """Check for location discrepancies"""
        flags = []

        # Extract locations from different sources
        personal_locations = structured_info['personal_info'].get('locations', [])

        # Simple check for now - can be enhanced with geographic validation
        if len(set(personal_locations)) > 3:  # Multiple different locations
            flags.append(FraudFlag(
                fraud_type=FraudType.LOCATION_DISCREPANCY,
                risk_level=RiskLevel.MEDIUM,
                confidence=0.5,
                description=f"Multiple locations mentioned: {personal_locations}",
                evidence={
                    'locations': personal_locations,
                    'location_count': len(set(personal_locations))
                },
                severity_score=0.3,
                recommendation="Verify current location and work authorization status"
            ))

        return flags

    def _calculate_risk_assessment(self, fraud_flags: List[FraudFlag]) -> Dict[str, Any]:
        """Calculate overall risk assessment"""
        if not fraud_flags:
            return {
                'overall_risk': RiskLevel.LOW.value,
                'risk_score': 0.0,
                'critical_flags': 0,
                'high_flags': 0,
                'medium_flags': 0,
                'low_flags': 0,
                'total_flags': 0
            }

        # Count flags by risk level
        risk_counts = Counter([flag.risk_level.value for flag in fraud_flags])

        # Calculate weighted risk score
        risk_weights = {
            RiskLevel.LOW.value: 0.1,
            RiskLevel.MEDIUM.value: 0.3,
            RiskLevel.HIGH.value: 0.7,
            RiskLevel.CRITICAL.value: 1.0
        }

        total_weighted_score = sum(
            risk_counts.get(level, 0) * weight
            for level, weight in risk_weights.items()
        )

        total_flags = len(fraud_flags)
        risk_score = min(total_weighted_score / max(total_flags, 1), 1.0)

        # Determine overall risk level
        if risk_counts.get(RiskLevel.CRITICAL.value, 0) > 0 or risk_score > 0.8:
            overall_risk = RiskLevel.CRITICAL.value
        elif risk_counts.get(RiskLevel.HIGH.value, 0) > 0 or risk_score > 0.6:
            overall_risk = RiskLevel.HIGH.value
        elif risk_counts.get(RiskLevel.MEDIUM.value, 0) > 0 or risk_score > 0.3:
            overall_risk = RiskLevel.MEDIUM.value
        else:
            overall_risk = RiskLevel.LOW.value

        return {
            'overall_risk': overall_risk,
            'risk_score': risk_score,
            'critical_flags': risk_counts.get(RiskLevel.CRITICAL.value, 0),
            'high_flags': risk_counts.get(RiskLevel.HIGH.value, 0),
            'medium_flags': risk_counts.get(RiskLevel.MEDIUM.value, 0),
            'low_flags': risk_counts.get(RiskLevel.LOW.value, 0),
            'total_flags': total_flags,
            'fraud_types': list(set(flag.fraud_type.value for flag in fraud_flags))
        }

    def _calculate_confidence_scores(self, fraud_flags: List[FraudFlag],
                                   structured_info: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence scores for different aspects"""
        scores = {
            'experience_authenticity': 1.0,
            'education_validity': 1.0,
            'skills_alignment': 1.0,
            'timeline_consistency': 1.0,
            'content_originality': 1.0,
            'overall_authenticity': 1.0
        }

        # Reduce confidence based on fraud flags
        for flag in fraud_flags:
            impact = flag.severity_score * (1 - flag.confidence)

            if flag.fraud_type in [FraudType.EXPERIENCE_INCONSISTENCY, FraudType.CAREER_PROGRESSION_ANOMALY]:
                scores['experience_authenticity'] = max(0.0, scores['experience_authenticity'] - impact)

            elif flag.fraud_type == FraudType.EDUCATION_MISMATCH:
                scores['education_validity'] = max(0.0, scores['education_validity'] - impact)

            elif flag.fraud_type == FraudType.SKILL_EXPERIENCE_GAP:
                scores['skills_alignment'] = max(0.0, scores['skills_alignment'] - impact)

            elif flag.fraud_type == FraudType.TIMELINE_INCONSISTENCY:
                scores['timeline_consistency'] = max(0.0, scores['timeline_consistency'] - impact)

            elif flag.fraud_type in [FraudType.PLAGIARISM, FraudType.DUPLICATE_CONTENT]:
                scores['content_originality'] = max(0.0, scores['content_originality'] - impact)

        # Calculate overall authenticity score
        weights = {
            'experience_authenticity': 0.25,
            'education_validity': 0.20,
            'skills_alignment': 0.20,
            'timeline_consistency': 0.15,
            'content_originality': 0.20
        }

        scores['overall_authenticity'] = sum(
            scores[metric] * weight for metric, weight in weights.items()
        )

        return scores

    def _generate_recommendations(self, fraud_flags: List[FraudFlag]) -> List[Dict[str, Any]]:
        """Generate recommendations based on detected fraud flags"""
        if not fraud_flags:
            return [{
                'type': 'positive',
                'title': 'Clean Resume',
                'description': 'No significant fraud indicators detected.',
                'priority': 'info',
                'actions': ['Proceed with normal interview process']
            }]

        recommendations = []

        # Group recommendations by fraud type
        fraud_types = set(flag.fraud_type for flag in fraud_flags)

        for fraud_type in fraud_types:
            type_flags = [flag for flag in fraud_flags if flag.fraud_type == fraud_type]
            highest_severity_flag = max(type_flags, key=lambda f: f.severity_score)

            if fraud_type == FraudType.EXPERIENCE_INCONSISTENCY:
                recommendations.append({
                    'type': 'verification',
                    'title': 'Verify Work Experience',
                    'description': 'Employment history contains inconsistencies that require verification.',
                    'priority': 'high' if highest_severity_flag.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL] else 'medium',
                    'actions': [
                        'Contact previous employers directly',
                        'Request employment verification letters',
                        'Verify job titles and responsibilities',
                        'Check employment dates and duration'
                    ]
                })

            elif fraud_type == FraudType.EDUCATION_MISMATCH:
                recommendations.append({
                    'type': 'verification',
                    'title': 'Verify Educational Background',
                    'description': 'Educational credentials require verification.',
                    'priority': 'high' if highest_severity_flag.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL] else 'medium',
                    'actions': [
                        'Request official transcripts',
                        'Verify degrees with institutions',
                        'Check graduation dates',
                        'Validate GPAs and honors'
                    ]
                })

            elif fraud_type == FraudType.SKILL_EXPERIENCE_GAP:
                recommendations.append({
                    'type': 'assessment',
                    'title': 'Technical Skills Assessment',
                    'description': 'Claimed skills may not align with experience level.',
                    'priority': 'medium',
                    'actions': [
                        'Conduct technical skills assessment',
                        'Request portfolio or code samples',
                        'Perform hands-on technical interview',
                        'Verify certifications and training'
                    ]
                })

            elif fraud_type == FraudType.PLAGIARISM:
                recommendations.append({
                    'type': 'investigation',
                    'title': 'Content Originality Review',
                    'description': 'Resume content shows high similarity to other sources.',
                    'priority': 'critical',
                    'actions': [
                        'Conduct in-depth interview about experience',
                        'Ask for specific examples and details',
                        'Verify unique accomplishments',
                        'Consider rejecting if plagiarism is confirmed'
                    ]
                })

            elif fraud_type == FraudType.TIMELINE_INCONSISTENCY:
                recommendations.append({
                    'type': 'clarification',
                    'title': 'Clarify Timeline Issues',
                    'description': 'Timeline inconsistencies need clarification.',
                    'priority': 'medium',
                    'actions': [
                        'Request detailed employment timeline',
                        'Clarify overlapping employment periods',
                        'Verify gaps in employment',
                        'Confirm education and work timeline alignment'
                    ]
                })

        # Add general recommendations based on overall risk
        high_risk_flags = [flag for flag in fraud_flags if flag.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]]

        if high_risk_flags:
            recommendations.append({
                'type': 'caution',
                'title': 'High Risk Candidate',
                'description': 'Multiple serious fraud indicators detected. Exercise extreme caution.',
                'priority': 'critical',
                'actions': [
                    'Conduct extensive background verification',
                    'Require multiple rounds of interviews',
                    'Verify all claims independently',
                    'Consider hiring probation period if proceeding',
                    'Document all verification attempts'
                ]
            })

        return recommendations

    def _integrate_gemini_findings(self, existing_flags: List[FraudFlag],
                                 gemini_result: GeminiAnalysisResult) -> None:
        """Integrate Gemini AI findings with existing fraud flags"""
        try:
            # Add Gemini findings as additional fraud flags
            for finding in gemini_result.findings:
                # Determine risk level based on Gemini's assessment
                if gemini_result.risk_level.value == 'critical':
                    risk_level = RiskLevel.CRITICAL
                elif gemini_result.risk_level.value == 'high':
                    risk_level = RiskLevel.HIGH
                elif gemini_result.risk_level.value == 'medium':
                    risk_level = RiskLevel.MEDIUM
                else:
                    risk_level = RiskLevel.LOW

                # Create fraud flag from Gemini finding
                gemini_flag = FraudFlag(
                    fraud_type=FraudType.SUSPICIOUS_KEYWORDS,  # Generic type for AI findings
                    risk_level=risk_level,
                    confidence=gemini_result.confidence,
                    description=f"AI Analysis: {finding}",
                    evidence={
                        'source': 'gemini_ai',
                        'analysis_type': gemini_result.analysis_type.value,
                        'raw_evidence': gemini_result.evidence
                    },
                    severity_score=gemini_result.confidence,
                    recommendation=f"Investigate AI finding: {finding}"
                )
                existing_flags.append(gemini_flag)

            logger.info(f"Integrated {len(gemini_result.findings)} Gemini AI findings")

        except Exception as e:
            logger.error(f"Failed to integrate Gemini findings: {e}")

    def generate_fraud_report(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive fraud detection report"""
        report = {
            'report_id': f"FR_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'detailed_findings': {},
            'verification_requirements': {},
            'decision_support': {}
        }

        # Extract key information
        fraud_flags = [FraudFlag(**flag) if isinstance(flag, dict) else flag
                      for flag in analysis_results.get('fraud_flags', [])]
        risk_assessment = analysis_results.get('risk_assessment', {})
        confidence_scores = analysis_results.get('confidence_scores', {})

        # Summary section
        report['summary'] = {
            'overall_risk_level': risk_assessment.get('overall_risk', 'low'),
            'risk_score': risk_assessment.get('risk_score', 0.0),
            'total_fraud_flags': len(fraud_flags),
            'authenticity_confidence': confidence_scores.get('overall_authenticity', 1.0),
            'recommendation': 'REJECT' if risk_assessment.get('overall_risk') == 'critical' else
                           'INVESTIGATE' if risk_assessment.get('overall_risk') in ['high', 'medium'] else
                           'PROCEED'
        }

        # Detailed findings
        report['detailed_findings'] = {
            'fraud_flags_by_type': {},
            'risk_breakdown': risk_assessment,
            'confidence_breakdown': confidence_scores
        }

        # Group fraud flags by type
        for flag in fraud_flags:
            flag_type = flag.fraud_type.value
            if flag_type not in report['detailed_findings']['fraud_flags_by_type']:
                report['detailed_findings']['fraud_flags_by_type'][flag_type] = []

            report['detailed_findings']['fraud_flags_by_type'][flag_type].append({
                'description': flag.description,
                'risk_level': flag.risk_level.value,
                'confidence': flag.confidence,
                'severity_score': flag.severity_score,
                'evidence': flag.evidence
            })

        # Verification requirements
        verification_needed = []
        for flag in fraud_flags:
            if flag.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                verification_needed.append({
                    'type': flag.fraud_type.value,
                    'urgency': 'high',
                    'method': flag.recommendation
                })

        report['verification_requirements'] = {
            'immediate_verification_needed': len(verification_needed) > 0,
            'verification_items': verification_needed
        }

        # Decision support
        report['decision_support'] = {
            'proceed_probability': confidence_scores.get('overall_authenticity', 1.0),
            'investigation_required': risk_assessment.get('overall_risk') in ['medium', 'high'],
            'rejection_recommended': risk_assessment.get('overall_risk') == 'critical',
            'next_steps': analysis_results.get('recommendations', [])
        }

        return report


# Utility function to convert dataclass to dict for JSON serialization
def fraud_flag_to_dict(flag: FraudFlag) -> Dict[str, Any]:
    """Convert FraudFlag to dictionary for serialization"""
    return {
        'fraud_type': flag.fraud_type.value,
        'risk_level': flag.risk_level.value,
        'confidence': flag.confidence,
        'description': flag.description,
        'evidence': flag.evidence,
        'severity_score': flag.severity_score,
        'recommendation': flag.recommendation
    }


if __name__ == "__main__":
    # Test the fraud detector
    from config import config

    detector = FraudDetector(config.get_fraud_threshold)

    sample_resume = """
    John Smith
    Senior Software Engineer / Team Lead
    Email: john.smith@email.com
    Phone: (555) 123-4567

    EXPERIENCE:
    Chief Technology Officer | Google | 2020 - Present
    - Led engineering organization of 500+ engineers
    - Increased revenue by 1000% through innovative solutions

    Senior Software Engineer | Startup XYZ | 2019 - 2020
    - Single-handedly built entire platform
    - Expert in all programming languages and technologies

    EDUCATION:
    PhD in Computer Science | Harvard University | 2025
    Master of Science | MIT | 2018
    Bachelor of Science | Stanford | 2016

    SKILLS:
    Python, Java, JavaScript, C++, Ruby, Go, Rust, Scala, Haskell,
    Machine Learning, AI, Blockchain, Quantum Computing, All Cloud Platforms
    """

    results = detector.analyze_resume(sample_resume)
    print("Fraud Detection Results:")
    print(f"Risk Level: {results['risk_assessment']['overall_risk']}")
    print(f"Total Flags: {len(results['fraud_flags'])}")

    for flag in results['fraud_flags']:
        print(f"- {flag.description} (Risk: {flag.risk_level.value})")

    # Generate report
    report = detector.generate_fraud_report(results)
    print(f"\nRecommendation: {report['summary']['recommendation']}")
