"""
Fit Scorer for Fraudulent Candidate Detection Tool

This module provides comprehensive resume-job description matching capabilities including:
- Semantic similarity analysis
- Skills matching and gap analysis
- Experience level alignment
- Education requirements verification
- Qualification scoring
- Overqualification detection
- Cultural fit indicators
- Red flag identification for unrealistic matches
"""

import logging
import re
import json
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from sentence_transformers import SentenceTransformer

from .nlp_analyzer import NLPAnalyzer
from .utils import SimilarityCalculator, SkillExtractor

# Configure logging
logger = logging.getLogger(__name__)

class FitLevel(Enum):
    """Fit level enumeration"""
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    MISMATCH = "mismatch"

class QualificationStatus(Enum):
    """Qualification status"""
    OVERQUALIFIED = "overqualified"
    QUALIFIED = "qualified"
    UNDERQUALIFIED = "underqualified"
    UNQUALIFIED = "unqualified"

class RequirementType(Enum):
    """Requirement types"""
    MUST_HAVE = "must_have"
    PREFERRED = "preferred"
    NICE_TO_HAVE = "nice_to_have"

@dataclass
class SkillMatch:
    """Represents a skill match between resume and job requirements"""
    skill: str
    resume_has: bool
    job_requires: bool
    requirement_type: RequirementType
    match_confidence: float
    years_experience: Optional[int]
    proficiency_level: Optional[str]

@dataclass
class ExperienceMatch:
    """Represents experience matching analysis"""
    required_years: Optional[int]
    candidate_years: float
    experience_gap: float
    relevant_experience: List[Dict[str, Any]]
    industry_match: bool
    role_progression: str

@dataclass
class EducationMatch:
    """Represents education matching analysis"""
    required_degree: Optional[str]
    candidate_degrees: List[str]
    meets_requirement: bool
    overqualified: bool
    relevant_field: bool
    accreditation_verified: bool

@dataclass
class FitAnalysis:
    """Comprehensive fit analysis results"""
    overall_score: float
    fit_level: FitLevel
    qualification_status: QualificationStatus
    skill_matches: List[SkillMatch]
    experience_match: ExperienceMatch
    education_match: EducationMatch
    strengths: List[str]
    gaps: List[str]
    red_flags: List[str]
    recommendations: List[str]

class FitScorer:
    """Resume-job description fit scoring engine"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize fit scorer

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.nlp_analyzer = NLPAnalyzer()
        self.similarity_calc = SimilarityCalculator()
        self.skill_extractor = SkillExtractor()

        # Load scoring weights
        self.scoring_weights = self.config.get('scoring_weights', {
            'skills': 0.35,
            'experience': 0.25,
            'education': 0.20,
            'semantic_similarity': 0.20
        })

        # Experience level mappings
        self.experience_levels = {
            'intern': (0, 1),
            'entry': (0, 2),
            'junior': (1, 3),
            'mid': (3, 6),
            'senior': (5, 10),
            'lead': (7, 15),
            'principal': (10, 20),
            'director': (8, 25),
            'vp': (12, 30),
            'c-level': (15, 40)
        }

        # Common skill categories
        self.skill_categories = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go'],
            'web': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'express'],
            'database': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform'],
            'data': ['pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'spark'],
            'tools': ['git', 'jenkins', 'jira', 'confluence', 'slack', 'figma']
        }

        # Suspicious fit patterns that might indicate fraud
        self.fraud_indicators = {
            'overqualification_threshold': 0.9,
            'perfect_match_threshold': 0.98,
            'impossible_skill_combinations': [
                ['beginner', 'expert'],
                ['junior', '10+ years'],
                ['entry level', 'senior']
            ]
        }

    def calculate_fit_score(self, resume_data: Dict[str, Any],
                          job_description: str) -> FitAnalysis:
        """
        Calculate comprehensive fit score between resume and job description

        Args:
            resume_data: Structured resume data
            job_description: Job description text

        Returns:
            Comprehensive fit analysis
        """
        logger.info("Calculating resume-job fit score")

        try:
            # Parse job description
            job_requirements = self._parse_job_description(job_description)

            # Analyze different aspects
            skill_analysis = self._analyze_skills_fit(
                resume_data.get('skills', {}),
                job_requirements['skills']
            )

            experience_analysis = self._analyze_experience_fit(
                resume_data.get('experiences', []),
                job_requirements['experience']
            )

            education_analysis = self._analyze_education_fit(
                resume_data.get('education', []),
                job_requirements['education']
            )

            semantic_score = self._calculate_semantic_similarity(
                resume_data, job_description
            )

            # Calculate overall score
            overall_score = self._calculate_overall_score(
                skill_analysis['score'],
                experience_analysis['score'],
                education_analysis['score'],
                semantic_score
            )

            # Determine fit level and qualification status
            fit_level = self._determine_fit_level(overall_score)
            qualification_status = self._determine_qualification_status(
                skill_analysis, experience_analysis, education_analysis
            )

            # Identify strengths, gaps, and red flags
            strengths = self._identify_strengths(
                skill_analysis, experience_analysis, education_analysis
            )
            gaps = self._identify_gaps(
                skill_analysis, experience_analysis, education_analysis
            )
            red_flags = self._identify_red_flags(
                resume_data, job_requirements, overall_score
            )

            # Generate recommendations
            recommendations = self._generate_recommendations(
                skill_analysis, experience_analysis, education_analysis, gaps, red_flags
            )

            return FitAnalysis(
                overall_score=overall_score,
                fit_level=fit_level,
                qualification_status=qualification_status,
                skill_matches=skill_analysis['matches'],
                experience_match=experience_analysis['match_details'],
                education_match=education_analysis['match_details'],
                strengths=strengths,
                gaps=gaps,
                red_flags=red_flags,
                recommendations=recommendations
            )

        except Exception as e:
            logger.error(f"Error calculating fit score: {e}")
            return FitAnalysis(
                overall_score=0.0,
                fit_level=FitLevel.MISMATCH,
                qualification_status=QualificationStatus.UNQUALIFIED,
                skill_matches=[],
                experience_match=ExperienceMatch(None, 0.0, 0.0, [], False, "unknown"),
                education_match=EducationMatch(None, [], False, False, False, False),
                strengths=[],
                gaps=["Analysis failed"],
                red_flags=["Fit calculation error"],
                recommendations=["Manual review required"]
            )

    def _parse_job_description(self, job_description: str) -> Dict[str, Any]:
        """Parse job description to extract requirements"""
        logger.info("Parsing job description")

        requirements = {
            'skills': {'must_have': [], 'preferred': [], 'nice_to_have': []},
            'experience': {'years': None, 'type': [], 'industry': []},
            'education': {'required': [], 'preferred': []},
            'responsibilities': [],
            'qualifications': []
        }

        # Use NLP to extract information
        nlp_results = self.nlp_analyzer.generate_nlp_report(job_description)

        # Extract skills
        job_skills = nlp_results['analysis_results']['skills']

        # Categorize skills based on context
        text_lower = job_description.lower()

        # Must-have skills (required, must have, essential)
        must_have_patterns = [
            r'(?:required|must have|essential|mandatory)[\s\S]*?(?=(?:preferred|nice|optional|plus|bonus|\n\n|\Z))',
            r'requirements?:[\s\S]*?(?=(?:preferred|nice|optional|qualifications|\n\n|\Z))'
        ]

        for pattern in must_have_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                extracted_skills = self.skill_extractor.extract_skills(match)
                requirements['skills']['must_have'].extend(extracted_skills)

        # Preferred skills
        preferred_patterns = [
            r'(?:preferred|nice to have|plus|bonus|advantage)[\s\S]*?(?=(?:required|must|qualifications|\n\n|\Z))',
            r'preferred qualifications?:[\s\S]*?(?=(?:required|must|\n\n|\Z))'
        ]

        for pattern in preferred_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                extracted_skills = self.skill_extractor.extract_skills(match)
                requirements['skills']['preferred'].extend(extracted_skills)

        # Extract experience requirements
        exp_patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\+?\s*years?\s*in',
            r'minimum\s*(?:of\s*)?(\d+)\s*years?'
        ]

        for pattern in exp_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                requirements['experience']['years'] = int(matches[0])
                break

        # Extract education requirements
        education_patterns = [
            r'bachelor[\'s]*\s*(?:degree)?',
            r'master[\'s]*\s*(?:degree)?',
            r'phd|doctorate',
            r'high school|ged'
        ]

        for pattern in education_patterns:
            if re.search(pattern, text_lower):
                degree_type = pattern.replace(r'[\'s]*\s*(?:degree)?', '').replace('|', ' or ')
                requirements['education']['required'].append(degree_type)

        # Remove duplicates and clean up
        for skill_type in requirements['skills']:
            requirements['skills'][skill_type] = list(set(requirements['skills'][skill_type]))

        return requirements

    def _analyze_skills_fit(self, resume_skills: Dict[str, List[str]],
                           job_skills: Dict[str, List[str]]) -> Dict[str, Any]:
        """Analyze skills fit between resume and job requirements"""

        # Flatten resume skills
        all_resume_skills = []
        for category, skills in resume_skills.items():
            all_resume_skills.extend([skill.lower() for skill in skills])

        skill_matches = []
        total_score = 0.0
        max_score = 0.0

        # Check must-have skills
        for skill in job_skills.get('must_have', []):
            skill_lower = skill.lower()
            max_score += 1.0

            # Direct match
            if skill_lower in all_resume_skills:
                match = SkillMatch(
                    skill=skill,
                    resume_has=True,
                    job_requires=True,
                    requirement_type=RequirementType.MUST_HAVE,
                    match_confidence=1.0,
                    years_experience=None,
                    proficiency_level=None
                )
                total_score += 1.0
            else:
                # Fuzzy match
                best_match_score = 0.0
                for resume_skill in all_resume_skills:
                    similarity = self.similarity_calc.fuzzy_similarity(skill_lower, resume_skill)
                    if similarity['ratio'] > best_match_score:
                        best_match_score = similarity['ratio']

                if best_match_score > 0.8:
                    match = SkillMatch(
                        skill=skill,
                        resume_has=True,
                        job_requires=True,
                        requirement_type=RequirementType.MUST_HAVE,
                        match_confidence=best_match_score,
                        years_experience=None,
                        proficiency_level=None
                    )
                    total_score += best_match_score
                else:
                    match = SkillMatch(
                        skill=skill,
                        resume_has=False,
                        job_requires=True,
                        requirement_type=RequirementType.MUST_HAVE,
                        match_confidence=0.0,
                        years_experience=None,
                        proficiency_level=None
                    )

            skill_matches.append(match)

        # Check preferred skills (weighted less)
        for skill in job_skills.get('preferred', []):
            skill_lower = skill.lower()
            max_score += 0.5

            if skill_lower in all_resume_skills:
                match = SkillMatch(
                    skill=skill,
                    resume_has=True,
                    job_requires=True,
                    requirement_type=RequirementType.PREFERRED,
                    match_confidence=1.0,
                    years_experience=None,
                    proficiency_level=None
                )
                total_score += 0.5
                skill_matches.append(match)

        # Calculate final skills score
        skills_score = (total_score / max_score) if max_score > 0 else 0.0

        return {
            'score': min(skills_score, 1.0),
            'matches': skill_matches,
            'missing_must_have': [m.skill for m in skill_matches
                                if m.requirement_type == RequirementType.MUST_HAVE and not m.resume_has],
            'missing_preferred': [m.skill for m in skill_matches
                                if m.requirement_type == RequirementType.PREFERRED and not m.resume_has]
        }

    def _analyze_experience_fit(self, resume_experiences: List[Dict[str, Any]],
                              job_experience: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze experience fit"""

        # Calculate total experience years
        total_years = 0.0
        relevant_experiences = []

        for exp in resume_experiences:
            duration = exp.duration_months if hasattr(exp, 'duration_months') else exp.get('duration_months', 0)
            if duration:
                total_years += duration / 12.0
                relevant_experiences.append({
                    'title': exp.title if hasattr(exp, 'title') else exp.get('title', ''),
                    'company': exp.company if hasattr(exp, 'company') else exp.get('company', ''),
                    'years': duration / 12.0
                })

        required_years = job_experience.get('years', 0)
        experience_gap = max(0, required_years - total_years)

        # Calculate experience score
        if required_years == 0:
            exp_score = 1.0
        elif total_years >= required_years:
            # Bonus for exceeding requirements, but cap to avoid overqualification penalty
            exp_score = min(1.2, total_years / required_years)
        else:
            # Penalty for insufficient experience
            exp_score = total_years / required_years

        # Industry/role relevance analysis (simplified)
        industry_match = len(relevant_experiences) > 0

        # Role progression analysis
        if len(relevant_experiences) >= 2:
            role_progression = "progressive"
        elif len(relevant_experiences) == 1:
            role_progression = "stable"
        else:
            role_progression = "unknown"

        match_details = ExperienceMatch(
            required_years=required_years,
            candidate_years=total_years,
            experience_gap=experience_gap,
            relevant_experience=relevant_experiences,
            industry_match=industry_match,
            role_progression=role_progression
        )

        return {
            'score': min(exp_score, 1.0),
            'match_details': match_details
        }

    def _analyze_education_fit(self, resume_education: List[Dict[str, Any]],
                             job_education: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze education fit"""

        required_degrees = job_education.get('required', [])
        candidate_degrees = []

        for edu in resume_education:
            degree = edu.degree if hasattr(edu, 'degree') else edu.get('degree', '')
            if degree:
                candidate_degrees.append(degree.lower())

        meets_requirement = True
        overqualified = False
        relevant_field = True  # Simplified - would need field analysis

        if required_degrees:
            # Check if candidate meets minimum requirement
            meets_requirement = False
            degree_hierarchy = ['high school', 'associate', 'bachelor', 'master', 'phd', 'doctorate']

            for req_degree in required_degrees:
                req_degree_lower = req_degree.lower()
                for candidate_degree in candidate_degrees:
                    # Simple matching - could be enhanced
                    if (req_degree_lower in candidate_degree or
                        any(req_word in candidate_degree for req_word in req_degree_lower.split())):
                        meets_requirement = True
                        break

                # Check for overqualification
                if 'bachelor' in req_degree_lower and any('master' in cd or 'phd' in cd for cd in candidate_degrees):
                    overqualified = True
                elif 'master' in req_degree_lower and any('phd' in cd for cd in candidate_degrees):
                    overqualified = True

        # Calculate education score
        if not required_degrees:
            edu_score = 1.0  # No requirements
        elif meets_requirement:
            edu_score = 1.1 if overqualified else 1.0
        else:
            edu_score = 0.3  # Doesn't meet minimum requirement

        match_details = EducationMatch(
            required_degree=required_degrees[0] if required_degrees else None,
            candidate_degrees=[degree for degree in candidate_degrees],
            meets_requirement=meets_requirement,
            overqualified=overqualified,
            relevant_field=relevant_field,
            accreditation_verified=False  # Would need verification
        )

        return {
            'score': min(edu_score, 1.0),
            'match_details': match_details
        }

    def _calculate_semantic_similarity(self, resume_data: Dict[str, Any],
                                     job_description: str) -> float:
        """Calculate semantic similarity between resume and job description"""

        # Reconstruct resume text (simplified)
        resume_text_parts = []

        # Add experiences
        experiences = resume_data.get('experiences', [])
        for exp in experiences:
            title = exp.title if hasattr(exp, 'title') else exp.get('title', '')
            company = exp.company if hasattr(exp, 'company') else exp.get('company', '')
            description = exp.description if hasattr(exp, 'description') else exp.get('description', '')
            resume_text_parts.append(f"{title} {company} {description}")

        # Add skills
        skills = resume_data.get('skills', {})
        for category, skill_list in skills.items():
            resume_text_parts.extend(skill_list)

        resume_text = ' '.join(resume_text_parts)

        # Calculate similarity using multiple methods
        similarities = self.similarity_calc.calculate_text_similarity(resume_text, job_description)

        # Use average of different similarity metrics
        semantic_score = np.mean([
            similarities.get('semantic_similarity', 0.0),
            similarities.get('tfidf_similarity', 0.0),
            similarities.get('fuzzy_token_sort', 0.0)
        ])

        return min(semantic_score, 1.0)

    def _calculate_overall_score(self, skills_score: float, experience_score: float,
                               education_score: float, semantic_score: float) -> float:
        """Calculate weighted overall fit score"""

        weighted_score = (
            skills_score * self.scoring_weights['skills'] +
            experience_score * self.scoring_weights['experience'] +
            education_score * self.scoring_weights['education'] +
            semantic_score * self.scoring_weights['semantic_similarity']
        )

        return round(min(weighted_score, 1.0), 3)

    def _determine_fit_level(self, overall_score: float) -> FitLevel:
        """Determine fit level based on overall score"""
        if overall_score >= 0.85:
            return FitLevel.EXCELLENT
        elif overall_score >= 0.70:
            return FitLevel.GOOD
        elif overall_score >= 0.50:
            return FitLevel.AVERAGE
        elif overall_score >= 0.30:
            return FitLevel.POOR
        else:
            return FitLevel.MISMATCH

    def _determine_qualification_status(self, skill_analysis: Dict[str, Any],
                                      experience_analysis: Dict[str, Any],
                                      education_analysis: Dict[str, Any]) -> QualificationStatus:
        """Determine overall qualification status"""

        skills_score = skill_analysis['score']
        exp_score = experience_analysis['score']
        edu_score = education_analysis['score']

        # Check for overqualification
        exp_match = experience_analysis['match_details']
        edu_match = education_analysis['match_details']

        if (exp_match.candidate_years > (exp_match.required_years or 0) * 2 and
            edu_match.overqualified):
            return QualificationStatus.OVERQUALIFIED

        # Check minimum qualifications
        missing_must_have = len(skill_analysis.get('missing_must_have', []))

        if missing_must_have > 2:
            return QualificationStatus.UNQUALIFIED
        elif missing_must_have > 0 or exp_score < 0.7:
            return QualificationStatus.UNDERQUALIFIED
        else:
            return QualificationStatus.QUALIFIED

    def _identify_strengths(self, skill_analysis: Dict[str, Any],
                          experience_analysis: Dict[str, Any],
                          education_analysis: Dict[str, Any]) -> List[str]:
        """Identify candidate strengths"""
        strengths = []

        # Skills strengths
        matched_must_have = [m for m in skill_analysis['matches']
                           if m.requirement_type == RequirementType.MUST_HAVE and m.resume_has]
        if len(matched_must_have) == len([m for m in skill_analysis['matches']
                                        if m.requirement_type == RequirementType.MUST_HAVE]):
            strengths.append("Meets all required technical skills")

        # Experience strengths
        exp_match = experience_analysis['match_details']
        if exp_match.candidate_years > (exp_match.required_years or 0) * 1.5:
            strengths.append(f"Exceeds experience requirement with {exp_match.candidate_years:.1f} years")

        if exp_match.role_progression == "progressive":
            strengths.append("Demonstrates career progression")

        # Education strengths
        edu_match = education_analysis['match_details']
        if edu_match.overqualified:
            strengths.append("Exceeds educational requirements")

        return strengths

    def _identify_gaps(self, skill_analysis: Dict[str, Any],
                      experience_analysis: Dict[str, Any],
                      education_analysis: Dict[str, Any]) -> List[str]:
        """Identify gaps in candidate qualifications"""
        gaps = []

        # Missing required skills
        missing_must_have = skill_analysis.get('missing_must_have', [])
        if missing_must_have:
            gaps.append(f"Missing required skills: {', '.join(missing_must_have[:3])}")

        # Experience gaps
        exp_match = experience_analysis['match_details']
        if exp_match.experience_gap > 0:
            gaps.append(f"Needs {exp_match.experience_gap:.1f} more years of experience")

        # Education gaps
        edu_match = education_analysis['match_details']
        if not edu_match.meets_requirement:
            gaps.append("Does not meet educational requirements")

        return gaps

    def _identify_red_flags(self, resume_data: Dict[str, Any],
                          job_requirements: Dict[str, Any],
                          overall_score: float) -> List[str]:
        """Identify potential fraud red flags in fit analysis"""
        red_flags = []

        # Suspiciously perfect match
        if overall_score >= self.fraud_indicators['perfect_match_threshold']:
            red_flags.append("Suspiciously perfect match - may indicate resume tailoring or fraud")

        # Overqualification red flag
        experiences = resume_data.get('experiences', [])
        total_years = sum(exp.duration_months/12 if hasattr(exp, 'duration_months') and exp.duration_months
                         else exp.get('duration_months', 0)/12 for exp in experiences)
        required_years = job_requirements.get('experience', {}).get('years', 0)

        if required_years > 0 and total_years > required_years * 3:
            red_flags.append("Candidate may be significantly overqualified for this position")

        # Skills mismatch patterns
        skills = resume_data.get('skills', {})
        all_skills = []
        for category, skill_list in skills.items():
            all_skills.extend(skill_list)

        # Too many skills for experience level
        if len(all_skills) > 20 and total_years < 3:
            red_flags.append("Claims extensive skills with limited experience")

        # Impossible combinations
        skill_text = ' '.join(all_skills).lower()
        for impossible_combo in self.fraud_indicators['impossible_skill_combinations']:
            if all(term in skill_text for term in impossible_combo):
                red_flags.append(f"Contradictory skill claims: {impossible_combo}")

        return red_flags

    def _generate_recommendations(self, skill_analysis: Dict[str, Any],
                                experience_analysis: Dict[str, Any],
                                education_analysis: Dict[str, Any],
                                gaps: List[str], red_flags: List[str]) -> List[str]:
        """Generate recommendations based on fit analysis"""
        recommendations = []

        if red_flags:
            recommendations.append("PRIORITY: Investigate potential fraud indicators through detailed interview")

        missing_must_have = skill_analysis.get('missing_must_have', [])
        if missing_must_have:
            recommendations.append(f"Assess proficiency in: {', '.join(missing_must_have[:3])}")

        exp_match = experience_analysis['match_details']
        if exp_match.experience_gap > 2:
            recommendations.append("Consider if equivalent experience or training can substitute")
        elif exp_match.experience_gap > 0:
            recommendations.append("Verify depth of experience in relevant areas")

        if exp_match.candidate_years > (exp_match.required_years or 0) * 2:
            recommendations.append("Assess salary expectations and retention risk due to overqualification")

        edu_match = education_analysis['match_details']
        if not edu_match.meets_requirement:
            recommendations.append("Verify if work experience compensates for educational gap")

        if not recommendations:
            recommendations.append("Strong fit - proceed with standard interview process")

        return recommendations

    def generate_fit_report(self, fit_analysis: FitAnalysis) -> Dict[str, Any]:
        """Generate comprehensive fit analysis report"""

        report = {
            'report_id': f"FS_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'executive_summary': {
                'overall_score': fit_analysis.overall_score,
                'fit_level': fit_analysis.fit_level.value,
                'qualification_status': fit_analysis.qualification_status.value,
                'recommendation': self._get_hiring_recommendation(fit_analysis)
            },
            'detailed_analysis': {
                'skills_analysis': {
                    'total_matches': len([m for m in fit_analysis.skill_matches if m.resume_has]),
                    'missing_required': len([m for m in fit_analysis.skill_matches
                                           if m.requirement_type == RequirementType.MUST_HAVE and not m.resume_has]),
                    'skill_breakdown': [asdict(match) for match in fit_analysis.skill_matches]
                },
                'experience_analysis': asdict(fit_analysis.experience_match),
                'education_analysis': asdict(fit_analysis.education_match)
            },
            'risk_assessment': {
                'red_flags_count': len(fit_analysis.red_flags),
                'red_flags': fit_analysis.red_flags,
                'fraud_risk': 'high' if len(fit_analysis.red_flags) > 2 else
                             'medium' if fit_analysis.red_flags else 'low'
            },
            'gaps_and_strengths': {
                'strengths': fit_analysis.strengths,
                'gaps': fit_analysis.gaps,
                'development_areas': fit_analysis.gaps
            },
            'next_steps': {
                'recommendations': fit_analysis.recommendations,
                'interview_focus_areas': self._get_interview_focus_areas(fit_analysis),
                'verification_needed': len(fit_analysis.red_flags) > 0
            }
        }

        return report

    def _get_hiring_recommendation(self, fit_analysis: FitAnalysis) -> str:
        """Get hiring recommendation based on fit analysis"""
        if fit_analysis.red_flags and len(fit_analysis.red_flags) > 2:
            return "DO NOT HIRE - Multiple fraud indicators detected"
        elif fit_analysis.fit_level == FitLevel.EXCELLENT and fit_analysis.qualification_status == QualificationStatus.QUALIFIED:
            return "STRONG HIRE - Excellent fit for the role"
        elif fit_analysis.fit_level == FitLevel.GOOD and fit_analysis.qualification_status in [QualificationStatus.QUALIFIED, QualificationStatus.OVERQUALIFIED]:
            return "HIRE - Good fit with minor gaps addressable through training"
        elif fit_analysis.fit_level == FitLevel.AVERAGE and fit_analysis.qualification_status == QualificationStatus.QUALIFIED:
            return "CONDITIONAL HIRE - Average fit, consider if other candidates unavailable"
        elif fit_analysis.qualification_status == QualificationStatus.UNDERQUALIFIED:
            return "DO NOT HIRE - Does not meet minimum requirements"
        elif fit_analysis.qualification_status == QualificationStatus.OVERQUALIFIED:
            return "CAUTION - Overqualified candidate, assess retention risk"
        else:
            return "NO HIRE - Poor fit for the role"

    def _get_interview_focus_areas(self, fit_analysis: FitAnalysis) -> List[str]:
        """Get areas to focus on during interview based on fit analysis"""
        focus_areas = []

        # Focus on missing skills
        missing_skills = [m.skill for m in fit_analysis.skill_matches
                         if m.requirement_type == RequirementType.MUST_HAVE and not m.resume_has]
        if missing_skills:
            focus_areas.append(f"Assess knowledge in: {', '.join(missing_skills[:3])}")

        # Focus on experience gaps
        if fit_analysis.experience_match.experience_gap > 0:
            focus_areas.append("Probe depth of relevant experience and problem-solving approach")

        # Focus on red flags
        if fit_analysis.red_flags:
            focus_areas.append("Verify claims that seem inconsistent or overstated")

        # Focus on overqualification
        if fit_analysis.qualification_status == QualificationStatus.OVERQUALIFIED:
            focus_areas.append("Assess motivation and long-term interest in the role")

        if not focus_areas:
            focus_areas.append("Standard behavioral and technical assessment")

        return focus_areas

    def batch_score_candidates(self, candidates_data: List[Dict[str, Any]],
                              job_description: str) -> Dict[str, Any]:
        """Score multiple candidates against the same job description"""
        batch_results = {
            'batch_id': f"BS_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'job_description_summary': job_description[:200] + "..." if len(job_description) > 200 else job_description,
            'total_candidates': len(candidates_data),
            'results': [],
            'ranking': [],
            'summary_stats': {
                'excellent_fit': 0,
                'good_fit': 0,
                'average_fit': 0,
                'poor_fit': 0,
                'mismatch': 0,
                'qualified': 0,
                'overqualified': 0,
                'underqualified': 0,
                'unqualified': 0,
                'red_flags_detected': 0
            }
        }

        scored_candidates = []

        for i, candidate_data in enumerate(candidates_data):
            try:
                logger.info(f"Scoring candidate {i+1}/{len(candidates_data)}")

                fit_analysis = self.calculate_fit_score(
                    candidate_data.get('resume_data', {}),
                    job_description
                )

                result = {
                    'candidate_id': candidate_data.get('id', f'candidate_{i}'),
                    'candidate_name': candidate_data.get('name', f'Candidate {i+1}'),
                    'fit_analysis': asdict(fit_analysis),
                    'overall_score': fit_analysis.overall_score,
                    'fit_level': fit_analysis.fit_level.value,
                    'qualification_status': fit_analysis.qualification_status.value,
                    'red_flags_count': len(fit_analysis.red_flags),
                    'recommendation': self._get_hiring_recommendation(fit_analysis)
                }

                batch_results['results'].append(result)
                scored_candidates.append((result, fit_analysis))

                # Update summary stats
                batch_results['summary_stats'][fit_analysis.fit_level.value.replace('-', '_') + '_fit'] += 1
                batch_results['summary_stats'][fit_analysis.qualification_status.value] += 1
                if fit_analysis.red_flags:
                    batch_results['summary_stats']['red_flags_detected'] += 1

            except Exception as e:
                logger.error(f"Error scoring candidate {i}: {e}")
                error_result = {
                    'candidate_id': candidate_data.get('id', f'candidate_{i}'),
                    'candidate_name': candidate_data.get('name', f'Candidate {i+1}'),
                    'error': str(e),
                    'overall_score': 0.0,
                    'fit_level': 'error',
                    'recommendation': 'MANUAL REVIEW - Scoring failed'
                }
                batch_results['results'].append(error_result)

        # Create ranking
        valid_candidates = [(result, analysis) for result, analysis in scored_candidates if 'error' not in result]
        valid_candidates.sort(key=lambda x: x[1].overall_score, reverse=True)

        batch_results['ranking'] = [
            {
                'rank': i + 1,
                'candidate_id': result['candidate_id'],
                'candidate_name': result['candidate_name'],
                'overall_score': result['overall_score'],
                'fit_level': result['fit_level'],
                'recommendation': result['recommendation']
            }
            for i, (result, _) in enumerate(valid_candidates)
        ]

        return batch_results


if __name__ == "__main__":
    # Test the fit scorer
    scorer = FitScorer()

    # Sample resume data
    sample_resume = {
        'skills': {
            'programming': ['Python', 'Java', 'JavaScript'],
            'tools': ['Git', 'Docker', 'AWS'],
            'technical': ['Machine Learning', 'API Development']
        },
        'experiences': [
            {
                'title': 'Software Engineer',
                'company': 'Tech Corp',
                'duration_months': 24,
                'description': 'Developed web applications using Python and JavaScript'
            },
            {
                'title': 'Junior Developer',
                'company': 'Startup Inc',
                'duration_months': 18,
                'description': 'Built REST APIs and worked with databases'
            }
        ],
        'education': [
            {
                'degree': 'Bachelor of Science in Computer Science',
                'institution': 'State University',
                'graduation_year': 2020
            }
        ]
    }

    # Sample job description
    sample_job_desc = """
    We are looking for a Senior Software Engineer with 3+ years of experience.

    Required Skills:
    - Python programming
    - JavaScript and modern frameworks
    - API development
    - Cloud platforms (AWS preferred)
    - Bachelor's degree in Computer Science or related field

    Preferred Skills:
    - Machine Learning experience
    - Docker containerization
    - Git version control

    Responsibilities:
    - Design and implement scalable web applications
    - Collaborate with cross-functional teams
    - Mentor junior developers
    """

    # Calculate fit score
    fit_analysis = scorer.calculate_fit_score(sample_resume, sample_job_desc)

    print("Fit Analysis Results:")
    print(f"Overall Score: {fit_analysis.overall_score}")
    print(f"Fit Level: {fit_analysis.fit_level.value}")
    print(f"Qualification Status: {fit_analysis.qualification_status.value}")
    print(f"Strengths: {fit_analysis.strengths}")
    print(f"Gaps: {fit_analysis.gaps}")
    print(f"Red Flags: {fit_analysis.red_flags}")

    # Generate report
    report = scorer.generate_fit_report(fit_analysis)
    print(f"\nReport ID: {report['report_id']}")
    print(f"Hiring Recommendation: {report['executive_summary']['recommendation']}")
