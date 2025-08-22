"""
LinkedIn Profile Verifier for Fraudulent Candidate Detection Tool

This module provides LinkedIn profile verification capabilities including:
- Profile data extraction and parsing
- Cross-validation with resume information
- Discrepancy detection between LinkedIn and resume
- Employment history verification
- Skills and endorsements analysis
- Network and connection analysis
"""

import logging
import json
import re
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import requests
from dataclasses import dataclass, asdict
from urllib.parse import urljoin, urlparse
import hashlib
from enum import Enum
import random

from .utils import DateValidator, SimilarityCalculator, TextProcessor

# Configure logging
logger = logging.getLogger(__name__)

class VerificationStatus(Enum):
    """Verification status enumeration"""
    VERIFIED = "verified"
    DISCREPANCY = "discrepancy"
    NOT_FOUND = "not_found"
    ERROR = "error"
    PARTIAL = "partial"

class DiscrepancyType(Enum):
    """Types of discrepancies between resume and LinkedIn"""
    JOB_TITLE_MISMATCH = "job_title_mismatch"
    COMPANY_MISMATCH = "company_mismatch"
    DATE_MISMATCH = "date_mismatch"
    DURATION_MISMATCH = "duration_mismatch"
    SKILLS_MISMATCH = "skills_mismatch"
    EDUCATION_MISMATCH = "education_mismatch"
    LOCATION_MISMATCH = "location_mismatch"
    MISSING_EXPERIENCE = "missing_experience"
    EXTRA_EXPERIENCE = "extra_experience"

@dataclass
class LinkedInProfile:
    """LinkedIn profile data structure"""
    profile_id: str
    full_name: str
    headline: str
    location: str
    summary: str
    experience: List[Dict[str, Any]]
    education: List[Dict[str, Any]]
    skills: List[Dict[str, Any]]
    connections_count: Optional[int]
    profile_url: str
    last_updated: datetime
    verification_date: datetime

@dataclass
class ProfileDiscrepancy:
    """Represents a discrepancy between resume and LinkedIn profile"""
    discrepancy_type: DiscrepancyType
    description: str
    resume_value: Any
    linkedin_value: Any
    confidence: float
    severity: str  # low, medium, high, critical
    field: str
    suggestion: str

class LinkedInVerifier:
    """LinkedIn profile verification engine"""

    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize LinkedIn verifier

        Args:
            api_key: LinkedIn API key
            config: Configuration dictionary
        """
        self.api_key = api_key
        self.config = config or {}
        self.base_url = self.config.get('linkedin_base_url', 'https://api.linkedin.com/v2')
        self.timeout = self.config.get('linkedin_timeout', 30)

        # Rate limiting
        self.rate_limit = self.config.get('api_rate_limit', {}).get('linkedin', 100)  # per hour
        self.last_request_time = 0
        self.request_count = 0
        self.request_window_start = time.time()

        # Initialize utilities
        self.date_validator = DateValidator()
        self.similarity_calc = SimilarityCalculator()
        self.text_processor = TextProcessor()

        # Mock data for testing when no API key is provided
        self.use_mock_data = not bool(api_key)
        if self.use_mock_data:
            logger.warning("No LinkedIn API key provided. Using mock data for demonstration.")

    def _rate_limit_check(self) -> bool:
        """Check if we're within rate limits"""
        current_time = time.time()

        # Reset counter if window has passed
        if current_time - self.request_window_start > 3600:  # 1 hour
            self.request_count = 0
            self.request_window_start = current_time

        if self.request_count >= self.rate_limit:
            logger.warning("LinkedIn API rate limit reached")
            return False

        return True

    def _make_api_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make API request to LinkedIn"""
        if not self._rate_limit_check():
            raise Exception("Rate limit exceeded")

        if self.use_mock_data:
            return self._get_mock_profile_data(endpoint, params)

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'X-Restli-Protocol-Version': '2.0.0'
        }

        url = urljoin(self.base_url, endpoint)

        try:
            response = requests.get(url, headers=headers, params=params, timeout=self.timeout)
            response.raise_for_status()

            self.request_count += 1
            self.last_request_time = time.time()

            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"LinkedIn API request failed: {e}")
            raise

    def _get_mock_profile_data(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate mock LinkedIn profile data for testing"""
        profile_variations = [
            {
                'id': 'mock_profile_1',
                'firstName': 'John',
                'lastName': 'Smith',
                'headline': 'Senior Software Engineer at Google',
                'location': {'name': 'San Francisco Bay Area'},
                'summary': 'Experienced software engineer with expertise in Python, Java, and machine learning.',
                'positions': [
                    {
                        'title': 'Senior Software Engineer',
                        'companyName': 'Google',
                        'location': {'name': 'Mountain View, CA'},
                        'startDate': {'year': 2020, 'month': 3},
                        'endDate': None,  # Current position
                        'description': 'Leading development of scalable web applications and machine learning systems.'
                    },
                    {
                        'title': 'Software Engineer',
                        'companyName': 'Microsoft',
                        'location': {'name': 'Seattle, WA'},
                        'startDate': {'year': 2018, 'month': 6},
                        'endDate': {'year': 2020, 'month': 2},
                        'description': 'Developed cloud-based solutions and collaborated with cross-functional teams.'
                    }
                ],
                'educations': [
                    {
                        'schoolName': 'Stanford University',
                        'degree': 'Master of Science',
                        'fieldOfStudy': 'Computer Science',
                        'startDate': {'year': 2016},
                        'endDate': {'year': 2018}
                    },
                    {
                        'schoolName': 'UC Berkeley',
                        'degree': 'Bachelor of Science',
                        'fieldOfStudy': 'Computer Science',
                        'startDate': {'year': 2012},
                        'endDate': {'year': 2016}
                    }
                ],
                'skills': [
                    {'name': 'Python', 'endorsementCount': 45},
                    {'name': 'Java', 'endorsementCount': 38},
                    {'name': 'JavaScript', 'endorsementCount': 32},
                    {'name': 'Machine Learning', 'endorsementCount': 28},
                    {'name': 'AWS', 'endorsementCount': 22}
                ]
            },
            {
                'id': 'mock_profile_2',
                'firstName': 'Jane',
                'lastName': 'Doe',
                'headline': 'Data Scientist | Machine Learning Expert',
                'location': {'name': 'New York, NY'},
                'summary': 'Data scientist passionate about using ML to solve complex business problems.',
                'positions': [
                    {
                        'title': 'Senior Data Scientist',
                        'companyName': 'Netflix',
                        'location': {'name': 'Los Gatos, CA'},
                        'startDate': {'year': 2021, 'month': 1},
                        'endDate': None,
                        'description': 'Building recommendation systems and predictive models.'
                    }
                ],
                'educations': [
                    {
                        'schoolName': 'MIT',
                        'degree': 'PhD',
                        'fieldOfStudy': 'Computer Science',
                        'startDate': {'year': 2017},
                        'endDate': {'year': 2021}
                    }
                ],
                'skills': [
                    {'name': 'Python', 'endorsementCount': 67},
                    {'name': 'Machine Learning', 'endorsementCount': 89},
                    {'name': 'Deep Learning', 'endorsementCount': 54}
                ]
            }
        ]

        # Return a random profile or the first one
        return random.choice(profile_variations) if len(profile_variations) > 1 else profile_variations[0]

    def search_profile_by_name(self, full_name: str, company: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for LinkedIn profiles by name and optionally company

        Args:
            full_name: Full name to search for
            company: Optional company name to narrow search

        Returns:
            List of potential matching profiles
        """
        logger.info(f"Searching LinkedIn for profile: {full_name}")

        if self.use_mock_data:
            # Return mock search results
            mock_results = [self._get_mock_profile_data('search', {'name': full_name})]
            # Modify name to match search
            name_parts = full_name.split()
            if len(name_parts) >= 2:
                mock_results[0]['firstName'] = name_parts[0]
                mock_results[0]['lastName'] = name_parts[-1]
            return mock_results

        search_params = {
            'keywords': full_name,
            'facet.current-company': [company] if company else None
        }

        try:
            response = self._make_api_request('/people-search', search_params)
            return response.get('elements', [])
        except Exception as e:
            logger.error(f"Profile search failed: {e}")
            return []

    def get_profile_details(self, profile_id: str) -> Optional[LinkedInProfile]:
        """
        Get detailed profile information

        Args:
            profile_id: LinkedIn profile ID

        Returns:
            LinkedInProfile object or None if not found
        """
        logger.info(f"Fetching profile details for: {profile_id}")

        try:
            if self.use_mock_data:
                profile_data = self._get_mock_profile_data(f'/people/{profile_id}')
            else:
                profile_data = self._make_api_request(f'/people/{profile_id}')

            return self._parse_profile_data(profile_data)

        except Exception as e:
            logger.error(f"Failed to fetch profile details: {e}")
            return None

    def _parse_profile_data(self, raw_data: Dict[str, Any]) -> LinkedInProfile:
        """Parse raw LinkedIn API response into LinkedInProfile object"""

        # Parse experience
        experience = []
        for position in raw_data.get('positions', []):
            exp_entry = {
                'title': position.get('title', ''),
                'company': position.get('companyName', ''),
                'location': position.get('location', {}).get('name', ''),
                'start_date': self._parse_linkedin_date(position.get('startDate')),
                'end_date': self._parse_linkedin_date(position.get('endDate')),
                'description': position.get('description', ''),
                'is_current': position.get('endDate') is None
            }
            experience.append(exp_entry)

        # Parse education
        education = []
        for edu in raw_data.get('educations', []):
            edu_entry = {
                'school': edu.get('schoolName', ''),
                'degree': edu.get('degree', ''),
                'field_of_study': edu.get('fieldOfStudy', ''),
                'start_year': edu.get('startDate', {}).get('year'),
                'end_year': edu.get('endDate', {}).get('year'),
                'gpa': edu.get('grade')
            }
            education.append(edu_entry)

        # Parse skills
        skills = []
        for skill in raw_data.get('skills', []):
            skill_entry = {
                'name': skill.get('name', ''),
                'endorsements': skill.get('endorsementCount', 0)
            }
            skills.append(skill_entry)

        return LinkedInProfile(
            profile_id=raw_data.get('id', ''),
            full_name=f"{raw_data.get('firstName', '')} {raw_data.get('lastName', '')}".strip(),
            headline=raw_data.get('headline', ''),
            location=raw_data.get('location', {}).get('name', ''),
            summary=raw_data.get('summary', ''),
            experience=experience,
            education=education,
            skills=skills,
            connections_count=raw_data.get('numConnections'),
            profile_url=raw_data.get('publicProfileUrl', ''),
            last_updated=datetime.now(),
            verification_date=datetime.now()
        )

    def _parse_linkedin_date(self, date_obj: Optional[Dict[str, int]]) -> Optional[datetime]:
        """Parse LinkedIn date object to datetime"""
        if not date_obj:
            return None

        year = date_obj.get('year')
        month = date_obj.get('month', 1)
        day = date_obj.get('day', 1)

        if year:
            try:
                return datetime(year, month, day)
            except ValueError:
                return datetime(year, 1, 1)  # Fallback to January 1st

        return None

    def verify_against_resume(self, resume_data: Dict[str, Any],
                            linkedin_profile: LinkedInProfile) -> Dict[str, Any]:
        """
        Verify resume data against LinkedIn profile

        Args:
            resume_data: Structured resume data
            linkedin_profile: LinkedIn profile data

        Returns:
            Verification results with discrepancies and scores
        """
        logger.info("Verifying resume against LinkedIn profile")

        verification_results = {
            'overall_match_score': 0.0,
            'verification_status': VerificationStatus.ERROR,
            'discrepancies': [],
            'matches': [],
            'missing_from_resume': [],
            'missing_from_linkedin': [],
            'confidence_scores': {},
            'summary': {}
        }

        try:
            # Verify personal information
            personal_discrepancies = self._verify_personal_info(
                resume_data.get('personal_info', {}), linkedin_profile
            )
            verification_results['discrepancies'].extend(personal_discrepancies)

            # Verify work experience
            experience_results = self._verify_work_experience(
                resume_data.get('experiences', []), linkedin_profile.experience
            )
            verification_results['discrepancies'].extend(experience_results['discrepancies'])
            verification_results['missing_from_resume'].extend(experience_results['missing_from_resume'])
            verification_results['missing_from_linkedin'].extend(experience_results['missing_from_linkedin'])

            # Verify education
            education_results = self._verify_education(
                resume_data.get('education', []), linkedin_profile.education
            )
            verification_results['discrepancies'].extend(education_results['discrepancies'])

            # Verify skills
            skills_results = self._verify_skills(
                resume_data.get('skills', {}), linkedin_profile.skills
            )
            verification_results['discrepancies'].extend(skills_results['discrepancies'])

            # Calculate overall match score
            verification_results['overall_match_score'] = self._calculate_overall_match_score(
                verification_results['discrepancies'],
                len(linkedin_profile.experience),
                len(linkedin_profile.education),
                len(linkedin_profile.skills)
            )

            # Determine verification status
            verification_results['verification_status'] = self._determine_verification_status(
                verification_results['overall_match_score'], verification_results['discrepancies']
            )

            # Calculate confidence scores
            verification_results['confidence_scores'] = self._calculate_confidence_scores(
                verification_results['discrepancies']
            )

            # Generate summary
            verification_results['summary'] = self._generate_verification_summary(
                verification_results
            )

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            verification_results['verification_status'] = VerificationStatus.ERROR
            verification_results['error'] = str(e)

        return verification_results

    def _verify_personal_info(self, resume_personal: Dict[str, Any],
                            linkedin_profile: LinkedInProfile) -> List[ProfileDiscrepancy]:
        """Verify personal information between resume and LinkedIn"""
        discrepancies = []

        # Name verification
        resume_names = resume_personal.get('names', [])
        if resume_names and linkedin_profile.full_name:
            name_match = any(
                self.similarity_calc.fuzzy_similarity(name, linkedin_profile.full_name)['ratio'] > 0.8
                for name in resume_names
            )

            if not name_match:
                discrepancies.append(ProfileDiscrepancy(
                    discrepancy_type=DiscrepancyType.COMPANY_MISMATCH,  # Using as general mismatch
                    description="Name mismatch between resume and LinkedIn profile",
                    resume_value=resume_names,
                    linkedin_value=linkedin_profile.full_name,
                    confidence=0.9,
                    severity="high",
                    field="name",
                    suggestion="Verify identity documents and confirm legal name"
                ))

        # Location verification
        resume_locations = resume_personal.get('locations', [])
        if resume_locations and linkedin_profile.location:
            location_match = any(
                self.similarity_calc.fuzzy_similarity(loc, linkedin_profile.location)['ratio'] > 0.7
                for loc in resume_locations
            )

            if not location_match:
                discrepancies.append(ProfileDiscrepancy(
                    discrepancy_type=DiscrepancyType.LOCATION_MISMATCH,
                    description="Location mismatch between resume and LinkedIn",
                    resume_value=resume_locations,
                    linkedin_value=linkedin_profile.location,
                    confidence=0.6,
                    severity="medium",
                    field="location",
                    suggestion="Clarify current location and work authorization"
                ))

        return discrepancies

    def _verify_work_experience(self, resume_experiences: List[Dict[str, Any]],
                              linkedin_experiences: List[Dict[str, Any]]) -> Dict[str, List]:
        """Verify work experience between resume and LinkedIn"""
        results = {
            'discrepancies': [],
            'missing_from_resume': [],
            'missing_from_linkedin': []
        }

        # Create mappings for comparison
        resume_jobs = []
        for exp in resume_experiences:
            resume_jobs.append({
                'title': exp.title if hasattr(exp, 'title') else exp.get('title', ''),
                'company': exp.company if hasattr(exp, 'company') else exp.get('company', ''),
                'start_date': exp.start_date if hasattr(exp, 'start_date') else exp.get('start_date'),
                'end_date': exp.end_date if hasattr(exp, 'end_date') else exp.get('end_date'),
                'original': exp
            })

        # Match resume experiences with LinkedIn experiences
        matched_linkedin = set()

        for resume_job in resume_jobs:
            best_match = None
            best_score = 0.0
            best_linkedin_idx = -1

            for i, linkedin_job in enumerate(linkedin_experiences):
                if i in matched_linkedin:
                    continue

                # Calculate similarity scores
                company_sim = self.similarity_calc.fuzzy_similarity(
                    resume_job['company'], linkedin_job['company']
                )['ratio']

                title_sim = self.similarity_calc.fuzzy_similarity(
                    resume_job['title'], linkedin_job['title']
                )['ratio']

                # Combined score (weighted)
                combined_score = (company_sim * 0.6) + (title_sim * 0.4)

                if combined_score > best_score and combined_score > 0.7:  # Threshold for match
                    best_score = combined_score
                    best_match = linkedin_job
                    best_linkedin_idx = i

            if best_match:
                matched_linkedin.add(best_linkedin_idx)

                # Check for discrepancies in matched jobs
                discrepancies = self._compare_job_details(resume_job, best_match)
                results['discrepancies'].extend(discrepancies)
            else:
                # Job in resume but not found in LinkedIn
                results['missing_from_linkedin'].append(resume_job['original'])

        # Find LinkedIn experiences not matched with resume
        for i, linkedin_job in enumerate(linkedin_experiences):
            if i not in matched_linkedin:
                results['missing_from_resume'].append(linkedin_job)

        return results

    def _compare_job_details(self, resume_job: Dict[str, Any],
                           linkedin_job: Dict[str, Any]) -> List[ProfileDiscrepancy]:
        """Compare details of matched jobs"""
        discrepancies = []

        # Title comparison
        title_sim = self.similarity_calc.fuzzy_similarity(
            resume_job['title'], linkedin_job['title']
        )['ratio']

        if title_sim < 0.8:
            discrepancies.append(ProfileDiscrepancy(
                discrepancy_type=DiscrepancyType.JOB_TITLE_MISMATCH,
                description=f"Job title mismatch: Resume '{resume_job['title']}' vs LinkedIn '{linkedin_job['title']}'",
                resume_value=resume_job['title'],
                linkedin_value=linkedin_job['title'],
                confidence=1.0 - title_sim,
                severity="medium" if title_sim > 0.6 else "high",
                field="job_title",
                suggestion="Verify actual job title with employer"
            ))

        # Date comparison
        if resume_job['start_date'] and linkedin_job['start_date']:
            date_diff = abs((resume_job['start_date'] - linkedin_job['start_date']).days)
            if date_diff > 90:  # More than 3 months difference
                discrepancies.append(ProfileDiscrepancy(
                    discrepancy_type=DiscrepancyType.DATE_MISMATCH,
                    description=f"Start date mismatch: {date_diff} days difference",
                    resume_value=resume_job['start_date'],
                    linkedin_value=linkedin_job['start_date'],
                    confidence=min(date_diff / 365.0, 1.0),  # Confidence increases with larger diff
                    severity="medium" if date_diff < 180 else "high",
                    field="start_date",
                    suggestion="Verify employment start date with HR records"
                ))

        return discrepancies

    def _verify_education(self, resume_education: List[Dict[str, Any]],
                        linkedin_education: List[Dict[str, Any]]) -> Dict[str, List]:
        """Verify education between resume and LinkedIn"""
        results = {
            'discrepancies': [],
            'missing_from_resume': [],
            'missing_from_linkedin': []
        }

        # Simple matching based on school name and degree
        for resume_edu in resume_education:
            school_name = resume_edu.institution if hasattr(resume_edu, 'institution') else resume_edu.get('institution', '')
            degree = resume_edu.degree if hasattr(resume_edu, 'degree') else resume_edu.get('degree', '')

            # Find matching education in LinkedIn
            found_match = False
            for linkedin_edu in linkedin_education:
                school_sim = self.similarity_calc.fuzzy_similarity(
                    school_name, linkedin_edu['school']
                )['ratio']

                degree_sim = self.similarity_calc.fuzzy_similarity(
                    degree, linkedin_edu['degree']
                )['ratio']

                if school_sim > 0.8 and degree_sim > 0.7:
                    found_match = True

                    # Check graduation year
                    resume_year = resume_edu.graduation_year if hasattr(resume_edu, 'graduation_year') else resume_edu.get('graduation_year')
                    linkedin_year = linkedin_edu['end_year']

                    if resume_year and linkedin_year and abs(resume_year - linkedin_year) > 1:
                        results['discrepancies'].append(ProfileDiscrepancy(
                            discrepancy_type=DiscrepancyType.EDUCATION_MISMATCH,
                            description=f"Graduation year mismatch: Resume {resume_year} vs LinkedIn {linkedin_year}",
                            resume_value=resume_year,
                            linkedin_value=linkedin_year,
                            confidence=0.8,
                            severity="medium",
                            field="graduation_year",
                            suggestion="Verify graduation date with institution"
                        ))
                    break

            if not found_match:
                results['missing_from_linkedin'].append(resume_edu)

        return results

    def _verify_skills(self, resume_skills: Dict[str, List[str]],
                      linkedin_skills: List[Dict[str, Any]]) -> Dict[str, List]:
        """Verify skills between resume and LinkedIn"""
        results = {
            'discrepancies': []
        }

        # Flatten resume skills
        all_resume_skills = []
        for category, skills_list in resume_skills.items():
            all_resume_skills.extend(skills_list)

        linkedin_skill_names = [skill['name'].lower() for skill in linkedin_skills]

        # Check for skills in resume but not in LinkedIn
        missing_skills = []
        for skill in all_resume_skills:
            skill_lower = skill.lower()
            if skill_lower not in linkedin_skill_names:
                # Check for partial matches
                partial_match = any(
                    self.similarity_calc.fuzzy_similarity(skill_lower, linkedin_skill)['ratio'] > 0.8
                    for linkedin_skill in linkedin_skill_names
                )

                if not partial_match:
                    missing_skills.append(skill)

        if len(missing_skills) > len(all_resume_skills) * 0.5:  # More than 50% missing
            results['discrepancies'].append(ProfileDiscrepancy(
                discrepancy_type=DiscrepancyType.SKILLS_MISMATCH,
                description=f"Many skills in resume not found in LinkedIn: {missing_skills[:5]}...",
                resume_value=missing_skills,
                linkedin_value=linkedin_skill_names,
                confidence=len(missing_skills) / len(all_resume_skills),
                severity="medium",
                field="skills",
                suggestion="Update LinkedIn profile with current skills or verify claimed skills"
            ))

        return results

    def _calculate_overall_match_score(self, discrepancies: List[ProfileDiscrepancy],
                                     linkedin_exp_count: int, linkedin_edu_count: int,
                                     linkedin_skills_count: int) -> float:
        """Calculate overall match score between resume and LinkedIn"""
        if not discrepancies:
            return 1.0

        # Base score
        base_score = 1.0

        # Deduct points for each discrepancy based on severity
        severity_weights = {
            'low': 0.05,
            'medium': 0.15,
            'high': 0.25,
            'critical': 0.4
        }

        total_deduction = sum(
            severity_weights.get(disc.severity, 0.1) * disc.confidence
            for disc in discrepancies
        )

        # Normalize by profile completeness (profiles with more data are more reliable)
        completeness_factor = min(
            (linkedin_exp_count + linkedin_edu_count + linkedin_skills_count) / 10.0, 1.0
        )

        final_score = max(0.0, base_score - (total_deduction * completeness_factor))

        return round(final_score, 3)

    def _determine_verification_status(self, match_score: float,
                                     discrepancies: List[ProfileDiscrepancy]) -> VerificationStatus:
        """Determine overall verification status"""
        critical_discrepancies = [d for d in discrepancies if d.severity == 'critical']
        high_discrepancies = [d for d in discrepancies if d.severity == 'high']

        if critical_discrepancies:
            return VerificationStatus.DISCREPANCY
        elif match_score < 0.3:
            return VerificationStatus.DISCREPANCY
        elif match_score < 0.7 or high_discrepancies:
            return VerificationStatus.PARTIAL
        elif match_score >= 0.8:
            return VerificationStatus.VERIFIED
        else:
            return VerificationStatus.PARTIAL

    def _calculate_confidence_scores(self, discrepancies: List[ProfileDiscrepancy]) -> Dict[str, float]:
        """Calculate confidence scores for different aspects"""
        scores = {
            'name_verification': 1.0,
            'experience_verification': 1.0,
            'education_verification': 1.0,
            'skills_verification': 1.0,
            'timeline_verification': 1.0,
            'overall_confidence': 1.0
        }

        # Reduce confidence based on discrepancies
        field_mapping = {
            'name': 'name_verification',
            'job_title': 'experience_verification',
            'company': 'experience_verification',
            'start_date': 'timeline_verification',
            'end_date': 'timeline_verification',
            'graduation_year': 'education_verification',
            'skills': 'skills_verification'
        }

        for disc in discrepancies:
            confidence_field = field_mapping.get(disc.field, 'overall_confidence')
            impact = disc.confidence * 0.3  # Scale impact
            scores[confidence_field] = max(0.0, scores[confidence_field] - impact)

        # Calculate overall confidence
        weights = {
            'name_verification': 0.2,
            'experience_verification': 0.3,
            'education_verification': 0.2,
            'skills_verification': 0.15,
            'timeline_verification': 0.15
        }

        scores['overall_confidence'] = sum(
            scores[field] * weight for field, weight in weights.items()
        )

        return scores

    def _generate_verification_summary(self, verification_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate verification summary"""
        summary = {
            'total_discrepancies': len(verification_results['discrepancies']),
            'critical_issues': len([d for d in verification_results['discrepancies'] if d.severity == 'critical']),
            'high_priority_issues': len([d for d in verification_results['discrepancies'] if d.severity == 'high']),
            'match_quality': 'excellent' if verification_results['overall_match_score'] > 0.8 else
                           'good' if verification_results['overall_match_score'] > 0.6 else
                           'poor' if verification_results['overall_match_score'] > 0.3 else 'very_poor',
            'recommendation': self._get_verification_recommendation(verification_results),
            'key_concerns': [d.description for d in verification_results['discrepancies']
                           if d.severity in ['high', 'critical']][:3]
        }

        return summary

    def _get_verification_recommendation(self, verification_results: Dict[str, Any]) -> str:
        """Get verification recommendation based on results"""
        status = verification_results['verification_status']
        match_score = verification_results['overall_match_score']
        critical_issues = len([d for d in verification_results['discrepancies'] if d.severity == 'critical'])

        if status == VerificationStatus.VERIFIED and match_score > 0.9:
            return "PROCEED - Strong LinkedIn verification"
        elif status == VerificationStatus.VERIFIED and match_score > 0.7:
            return "PROCEED WITH CONFIDENCE - Good LinkedIn verification"
        elif status == VerificationStatus.PARTIAL:
            return "INVESTIGATE - Some discrepancies found, additional verification recommended"
        elif status == VerificationStatus.DISCREPANCY and critical_issues > 0:
            return "CAUTION - Critical discrepancies found, thorough investigation required"
        elif status == VerificationStatus.NOT_FOUND:
            return "NO VERIFICATION - LinkedIn profile not found or inaccessible"
        else:
            return "MANUAL REVIEW - Complex verification case requiring human judgment"

    def generate_verification_report(self, verification_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive LinkedIn verification report"""
        report = {
            'report_id': f"LV_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'verification_summary': verification_results.get('summary', {}),
            'match_analysis': {
                'overall_score': verification_results.get('overall_match_score', 0.0),
                'status': verification_results.get('verification_status', VerificationStatus.ERROR).value,
                'confidence_breakdown': verification_results.get('confidence_scores', {})
            },
            'discrepancy_analysis': {
                'total_discrepancies': len(verification_results.get('discrepancies', [])),
                'by_severity': self._group_discrepancies_by_severity(verification_results.get('discrepancies', [])),
                'by_type': self._group_discrepancies_by_type(verification_results.get('discrepancies', [])),
                'detailed_discrepancies': [self._format_discrepancy(d) for d in verification_results.get('discrepancies', [])]
            },
            'missing_data_analysis': {
                'missing_from_resume': len(verification_results.get('missing_from_resume', [])),
                'missing_from_linkedin': len(verification_results.get('missing_from_linkedin', [])),
                'details': {
                    'resume_gaps': verification_results.get('missing_from_resume', []),
                    'linkedin_gaps': verification_results.get('missing_from_linkedin', [])
                }
            },
            'recommendations': self._generate_linkedin_recommendations(verification_results)
        }

        return report

    def _group_discrepancies_by_severity(self, discrepancies: List[ProfileDiscrepancy]) -> Dict[str, int]:
        """Group discrepancies by severity level"""
        severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}

        for disc in discrepancies:
            severity = disc.severity if hasattr(disc, 'severity') else getattr(disc, 'severity', 'medium')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        return severity_counts

    def _group_discrepancies_by_type(self, discrepancies: List[ProfileDiscrepancy]) -> Dict[str, int]:
        """Group discrepancies by type"""
        type_counts = {}

        for disc in discrepancies:
            disc_type = disc.discrepancy_type.value if hasattr(disc.discrepancy_type, 'value') else str(disc.discrepancy_type)
            type_counts[disc_type] = type_counts.get(disc_type, 0) + 1

        return type_counts

    def _format_discrepancy(self, discrepancy: ProfileDiscrepancy) -> Dict[str, Any]:
        """Format discrepancy for report"""
        return {
            'type': discrepancy.discrepancy_type.value if hasattr(discrepancy.discrepancy_type, 'value') else str(discrepancy.discrepancy_type),
            'description': discrepancy.description,
            'severity': discrepancy.severity,
            'confidence': discrepancy.confidence,
            'field': discrepancy.field,
            'resume_value': str(discrepancy.resume_value),
            'linkedin_value': str(discrepancy.linkedin_value),
            'suggestion': discrepancy.suggestion
        }

    def _generate_linkedin_recommendations(self, verification_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on LinkedIn verification results"""
        recommendations = []

        status = verification_results.get('verification_status')
        discrepancies = verification_results.get('discrepancies', [])
        match_score = verification_results.get('overall_match_score', 0.0)

        if status == VerificationStatus.NOT_FOUND:
            recommendations.append({
                'priority': 'high',
                'category': 'profile_missing',
                'title': 'LinkedIn Profile Not Found',
                'description': 'Unable to locate LinkedIn profile for verification',
                'actions': [
                    'Request LinkedIn profile URL from candidate',
                    'Verify professional network presence through other platforms',
                    'Consider alternative reference verification methods'
                ]
            })

        elif match_score < 0.5:
            recommendations.append({
                'priority': 'critical',
                'category': 'major_discrepancies',
                'title': 'Significant Profile Discrepancies',
                'description': 'Major inconsistencies between resume and LinkedIn profile',
                'actions': [
                    'Conduct detailed interview about work history',
                    'Request employment verification letters',
                    'Verify identity and professional credentials',
                    'Consider rejecting if discrepancies cannot be explained'
                ]
            })

        critical_discrepancies = [d for d in discrepancies if d.severity == 'critical']
        if critical_discrepancies:
            recommendations.append({
                'priority': 'critical',
                'category': 'critical_issues',
                'title': 'Critical Verification Issues',
                'description': f'Found {len(critical_discrepancies)} critical discrepancies requiring immediate attention',
                'actions': [
                    'Investigate each critical discrepancy individually',
                    'Contact previous employers directly',
                    'Verify educational credentials with institutions',
                    'Document all verification attempts'
                ]
            })

        missing_from_linkedin = verification_results.get('missing_from_linkedin', [])
        if len(missing_from_linkedin) > 2:
            recommendations.append({
                'priority': 'medium',
                'category': 'incomplete_profile',
                'title': 'Incomplete LinkedIn Profile',
                'description': f'LinkedIn profile missing {len(missing_from_linkedin)} experiences mentioned in resume',
                'actions': [
                    'Ask candidate to explain missing experiences',
                    'Verify recent or significant positions independently',
                    'Consider if profile is outdated or incomplete'
                ]
            })

        return recommendations

    def batch_verify_candidates(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Verify multiple candidates in batch"""
        batch_results = {
            'batch_id': f"BV_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'total_candidates': len(candidates),
            'results': [],
            'summary_stats': {
                'verified': 0,
                'partial': 0,
                'discrepancy': 0,
                'not_found': 0,
                'errors': 0
            }
        }

        for i, candidate in enumerate(candidates):
            try:
                logger.info(f"Processing candidate {i+1}/{len(candidates)}")

                # Search for LinkedIn profile
                profiles = self.search_profile_by_name(
                    candidate.get('name', ''),
                    candidate.get('current_company')
                )

                if not profiles:
                    result = {
                        'candidate_id': candidate.get('id', f'candidate_{i}'),
                        'status': VerificationStatus.NOT_FOUND.value,
                        'verification_results': None,
                        'error': 'LinkedIn profile not found'
                    }
                    batch_results['summary_stats']['not_found'] += 1
                else:
                    # Get detailed profile for best match
                    profile_data = self.get_profile_details(profiles[0]['id'])

                    if profile_data:
                        # Perform verification
                        verification_results = self.verify_against_resume(
                            candidate.get('resume_data', {}),
                            profile_data
                        )

                        result = {
                            'candidate_id': candidate.get('id', f'candidate_{i}'),
                            'status': verification_results['verification_status'].value,
                            'verification_results': verification_results,
                            'linkedin_profile': asdict(profile_data)
                        }

                        # Update summary stats
                        status = verification_results['verification_status']
                        if status == VerificationStatus.VERIFIED:
                            batch_results['summary_stats']['verified'] += 1
                        elif status == VerificationStatus.PARTIAL:
                            batch_results['summary_stats']['partial'] += 1
                        elif status == VerificationStatus.DISCREPANCY:
                            batch_results['summary_stats']['discrepancy'] += 1
                    else:
                        result = {
                            'candidate_id': candidate.get('id', f'candidate_{i}'),
                            'status': VerificationStatus.ERROR.value,
                            'verification_results': None,
                            'error': 'Failed to retrieve profile details'
                        }
                        batch_results['summary_stats']['errors'] += 1

            except Exception as e:
                logger.error(f"Error processing candidate {i}: {e}")
                result = {
                    'candidate_id': candidate.get('id', f'candidate_{i}'),
                    'status': VerificationStatus.ERROR.value,
                    'verification_results': None,
                    'error': str(e)
                }
                batch_results['summary_stats']['errors'] += 1

            batch_results['results'].append(result)

            # Rate limiting - sleep between requests
            if not self.use_mock_data:
                time.sleep(1)  # 1 second between requests

        return batch_results


if __name__ == "__main__":
    # Test the LinkedIn verifier
    verifier = LinkedInVerifier()  # Will use mock data

    # Test profile search
    profiles = verifier.search_profile_by_name("John Smith", "Google")
    print(f"Found {len(profiles)} profiles")

    if profiles:
        # Get detailed profile
        profile = verifier.get_profile_details(profiles[0]['id'])
        print(f"Profile: {profile.full_name} - {profile.headline}")

        # Test verification against mock resume data
        mock_resume_data = {
            'personal_info': {
                'names': ['John Smith'],
                'locations': ['San Francisco, CA']
            },
            'experiences': [
                {
                    'title': 'Senior Software Engineer',
                    'company': 'Google',
                    'start_date': datetime(2020, 1, 1),
                    'end_date': None
                }
            ],
            'education': [
                {
                    'institution': 'Stanford University',
                    'degree': 'Master of Science',
                    'graduation_year': 2018
                }
            ],
            'skills': {
                'programming': ['Python', 'Java'],
                'technical': ['Machine Learning']
            }
        }

        verification_results = verifier.verify_against_resume(mock_resume_data, profile)
        print(f"Verification Status: {verification_results['verification_status'].value}")
        print(f"Match Score: {verification_results['overall_match_score']}")
        print(f"Discrepancies: {len(verification_results['discrepancies'])}")

        # Generate report
        report = verifier.generate_verification_report(verification_results)
        print(f"Report ID: {report['report_id']}")
        print(f"Recommendation: {report['verification_summary'].get('recommendation', 'N/A')}")
