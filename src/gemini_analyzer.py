"""
Gemini AI Analyzer for Fraudulent Candidate Detection Tool

This module provides advanced AI-powered fraud detection capabilities using Google's Gemini AI.
It leverages Gemini's natural language understanding to perform sophisticated analysis of
resume content, identifying subtle patterns and inconsistencies that traditional methods might miss.
"""

import logging
import json
import re
import time
import os
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("Google Gemini AI not available. Install with: pip install google-genai")

# Configure logging
logger = logging.getLogger(__name__)

class AnalysisType(Enum):
    """Types of analysis that can be performed"""
    EXPERIENCE_VALIDATION = "experience_validation"
    EDUCATION_VERIFICATION = "education_verification"
    SKILLS_ASSESSMENT = "skills_assessment"
    TIMELINE_ANALYSIS = "timeline_analysis"
    CONTENT_ORIGINALITY = "content_originality"
    WRITING_STYLE = "writing_style"
    COMPREHENSIVE_FRAUD = "comprehensive_fraud"

class RiskLevel(Enum):
    """Risk level assessment"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class GeminiAnalysisResult:
    """Result from Gemini AI analysis"""
    analysis_type: AnalysisType
    risk_level: RiskLevel
    confidence: float
    findings: List[str]
    evidence: Dict[str, Any]
    recommendations: List[str]
    raw_response: str
    processing_time: float

class GeminiAnalyzer:
    """Advanced fraud detection using Google Gemini AI"""

    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Gemini analyzer

        Args:
            api_key: Gemini API key
            config: Configuration dictionary
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("Google Gemini AI not available. Install with: pip install google-genai")

        self.config = config or {}
        self.api_key = api_key or self.config.get('gemini_api_key', '') or os.getenv('GEMINI_API_KEY', '')

        if not self.api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY environment variable or pass api_key parameter.")

        # Initialize Gemini client
        try:
            self.client = genai.Client()
            self.model = self.config.get('gemini_model', 'gemini-2.5-flash')
            self.temperature = self.config.get('gemini_temperature', 0.3)
            self.max_tokens = self.config.get('gemini_max_tokens', 2000)

            # Safety settings
            self.safety_settings = self.config.get('gemini_safety_settings', {
                'HARASSMENT': 'BLOCK_MEDIUM_AND_ABOVE',
                'HATE_SPEECH': 'BLOCK_MEDIUM_AND_ABOVE',
                'SEXUALLY_EXPLICIT': 'BLOCK_MEDIUM_AND_ABOVE',
                'DANGEROUS_CONTENT': 'BLOCK_MEDIUM_AND_ABOVE',
            })

            # Rate limiting
            self.rate_limit = self.config.get('gemini_rate_limit', 60)  # requests per minute
            self.request_times = []

            # Cache for repeated analyses
            self.cache = {}
            self.cache_enabled = self.config.get('cache_enabled', True)

        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        current_time = time.time()

        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if current_time - t < 60]

        if len(self.request_times) >= self.rate_limit:
            logger.warning("Gemini API rate limit reached")
            return False

        return True

    def _get_cache_key(self, content: str, analysis_type: str) -> str:
        """Generate cache key for content and analysis type"""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        return f"{analysis_type}_{content_hash}"

    def _make_api_call(self, prompt: str, analysis_type: str) -> str:
        """Make API call to Gemini with rate limiting and caching"""
        # Check cache first
        cache_key = self._get_cache_key(prompt, analysis_type)
        if self.cache_enabled and cache_key in self.cache:
            logger.info(f"Using cached result for {analysis_type}")
            return self.cache[cache_key]

        # Check rate limit
        if not self._check_rate_limit():
            raise Exception("Rate limit exceeded. Please try again later.")

        try:
            start_time = time.time()

            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                )
            )

            self.request_times.append(time.time())

            # Cache the response
            if self.cache_enabled:
                self.cache[cache_key] = response.text

            logger.info(f"Gemini API call completed in {time.time() - start_time:.2f}s")
            return response.text

        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            raise

    def analyze_experience_fraud(self, resume_text: str, job_description: Optional[str] = None) -> GeminiAnalysisResult:
        """Analyze experience-related fraud patterns"""

        prompt = f"""
        You are an expert HR fraud detection specialist. Analyze the following resume for experience-related fraud indicators.

        Look for these specific patterns:
        1. Unrealistic job titles for experience level
        2. Inflated responsibilities or achievements
        3. Impossible career progression speed
        4. Inconsistent company names or locations
        5. Vague or generic job descriptions
        6. Timeline gaps or overlaps
        7. Salary claims that seem inflated
        8. Claims of managing unrealistic numbers of people/projects

        Resume Text:
        {resume_text}

        {"Job Description for Context: " + job_description if job_description else ""}

        Provide your analysis in JSON format:
        {{
            "risk_level": "low|medium|high|critical",
            "confidence": 0.85,
            "findings": ["specific finding 1", "specific finding 2"],
            "evidence": {{
                "suspicious_titles": ["title1", "title2"],
                "timeline_issues": ["issue1", "issue2"],
                "inflated_claims": ["claim1", "claim2"]
            }},
            "recommendations": ["recommendation 1", "recommendation 2"],
            "explanation": "Brief explanation of the analysis"
        }}
        """

        start_time = time.time()

        try:
            response = self._make_api_call(prompt, "experience_fraud")
            result = self._parse_json_response(response, AnalysisType.EXPERIENCE_VALIDATION)
            result.processing_time = time.time() - start_time
            return result

        except Exception as e:
            logger.error(f"Experience fraud analysis failed: {e}")
            return self._create_error_result(AnalysisType.EXPERIENCE_VALIDATION, str(e))

    def analyze_education_fraud(self, resume_text: str) -> GeminiAnalysisResult:
        """Analyze education-related fraud patterns"""

        prompt = f"""
        You are an expert education verification specialist. Analyze the following resume for education-related fraud indicators.

        Look for these specific patterns:
        1. Degree mills or non-accredited institutions
        2. Impossible graduation timelines
        3. Too many degrees in short timeframes
        4. Degrees from geographically scattered locations without explanation
        5. Inconsistent naming of institutions
        6. Unrealistic GPAs or honors
        7. Field of study mismatches with career
        8. Missing or vague graduation years

        Resume Text:
        {resume_text}

        Provide your analysis in JSON format:
        {{
            "risk_level": "low|medium|high|critical",
            "confidence": 0.85,
            "findings": ["specific finding 1", "specific finding 2"],
            "evidence": {{
                "suspicious_institutions": ["institution1", "institution2"],
                "timeline_issues": ["issue1", "issue2"],
                "degree_inconsistencies": ["inconsistency1", "inconsistency2"]
            }},
            "recommendations": ["recommendation 1", "recommendation 2"],
            "explanation": "Brief explanation of the analysis"
        }}
        """

        start_time = time.time()

        try:
            response = self._make_api_call(prompt, "education_fraud")
            result = self._parse_json_response(response, AnalysisType.EDUCATION_VERIFICATION)
            result.processing_time = time.time() - start_time
            return result

        except Exception as e:
            logger.error(f"Education fraud analysis failed: {e}")
            return self._create_error_result(AnalysisType.EDUCATION_VERIFICATION, str(e))

    def analyze_skills_fraud(self, resume_text: str, job_description: Optional[str] = None) -> GeminiAnalysisResult:
        """Analyze skills-related fraud patterns"""

        prompt = f"""
        You are an expert technical skills assessment specialist. Analyze the following resume for skills-related fraud indicators.

        Look for these specific patterns:
        1. Too many unrelated skills claimed
        2. Advanced skills without corresponding experience
        3. Buzzword overload without substance
        4. Contradictory skill combinations
        5. Skills that don't match job history
        6. Unrealistic proficiency claims
        7. Missing foundational skills for advanced claims
        8. Generic skill descriptions

        Resume Text:
        {resume_text}

        {"Job Description for Context: " + job_description if job_description else ""}

        Provide your analysis in JSON format:
        {{
            "risk_level": "low|medium|high|critical",
            "confidence": 0.85,
            "findings": ["specific finding 1", "specific finding 2"],
            "evidence": {{
                "overinflated_skills": ["skill1", "skill2"],
                "missing_foundations": ["foundation1", "foundation2"],
                "inconsistent_claims": ["claim1", "claim2"]
            }},
            "recommendations": ["recommendation 1", "recommendation 2"],
            "explanation": "Brief explanation of the analysis"
        }}
        """

        start_time = time.time()

        try:
            response = self._make_api_call(prompt, "skills_fraud")
            result = self._parse_json_response(response, AnalysisType.SKILLS_ASSESSMENT)
            result.processing_time = time.time() - start_time
            return result

        except Exception as e:
            logger.error(f"Skills fraud analysis failed: {e}")
            return self._create_error_result(AnalysisType.SKILLS_ASSESSMENT, str(e))

    def analyze_writing_style(self, resume_text: str) -> GeminiAnalysisResult:
        """Analyze writing style for fraud indicators"""

        prompt = f"""
        You are an expert writing style analyst specializing in resume authenticity. Analyze the writing style of this resume for fraud indicators.

        Look for these specific patterns:
        1. Inconsistent writing quality across sections
        2. Multiple writing styles suggesting different authors
        3. Copy-pasted content from templates
        4. Unnatural language patterns
        5. Inconsistent terminology usage
        6. Grammar/style patterns that don't match claimed background
        7. Generic or templated language
        8. Sudden changes in complexity or sophistication

        Resume Text:
        {resume_text}

        Provide your analysis in JSON format:
        {{
            "risk_level": "low|medium|high|critical",
            "confidence": 0.85,
            "findings": ["specific finding 1", "specific finding 2"],
            "evidence": {{
                "style_inconsistencies": ["inconsistency1", "inconsistency2"],
                "template_indicators": ["indicator1", "indicator2"],
                "language_patterns": ["pattern1", "pattern2"]
            }},
            "recommendations": ["recommendation 1", "recommendation 2"],
            "explanation": "Brief explanation of the analysis"
        }}
        """

        start_time = time.time()

        try:
            response = self._make_api_call(prompt, "writing_style")
            result = self._parse_json_response(response, AnalysisType.WRITING_STYLE)
            result.processing_time = time.time() - start_time
            return result

        except Exception as e:
            logger.error(f"Writing style analysis failed: {e}")
            return self._create_error_result(AnalysisType.WRITING_STYLE, str(e))

    def analyze_content_originality(self, resume_text: str, reference_texts: Optional[List[str]] = None) -> GeminiAnalysisResult:
        """Analyze content originality and detect plagiarism"""

        reference_context = ""
        if reference_texts:
            reference_context = f"""
            Reference texts to check against:
            {chr(10).join(reference_texts[:3])}  # Limit to first 3 references
            """

        prompt = f"""
        You are an expert plagiarism detection specialist. Analyze this resume for content originality issues.

        Look for these specific patterns:
        1. Generic job descriptions that could be copied
        2. Overly polished language inconsistent with claimed background
        3. Industry jargon used incorrectly or out of context
        4. Repetitive or templated achievement statements
        5. Content that seems copied from job postings
        6. Unrealistic or exaggerated accomplishments
        7. Inconsistent detail levels across sections

        Resume Text:
        {resume_text}

        {reference_context}

        Provide your analysis in JSON format:
        {{
            "risk_level": "low|medium|high|critical",
            "confidence": 0.85,
            "findings": ["specific finding 1", "specific finding 2"],
            "evidence": {{
                "generic_content": ["content1", "content2"],
                "potential_plagiarism": ["text1", "text2"],
                "inconsistent_detail": ["section1", "section2"]
            }},
            "recommendations": ["recommendation 1", "recommendation 2"],
            "explanation": "Brief explanation of the analysis"
        }}
        """

        start_time = time.time()

        try:
            response = self._make_api_call(prompt, "content_originality")
            result = self._parse_json_response(response, AnalysisType.CONTENT_ORIGINALITY)
            result.processing_time = time.time() - start_time
            return result

        except Exception as e:
            logger.error(f"Content originality analysis failed: {e}")
            return self._create_error_result(AnalysisType.CONTENT_ORIGINALITY, str(e))

    def comprehensive_fraud_analysis(self, resume_text: str, job_description: Optional[str] = None) -> GeminiAnalysisResult:
        """Perform comprehensive fraud analysis covering all aspects"""

        prompt = f"""
        You are the world's leading expert in resume fraud detection. Perform a comprehensive analysis of this resume for ALL types of fraud indicators.

        Analyze these categories systematically:
        1. EXPERIENCE FRAUD: Unrealistic titles, inflated responsibilities, impossible progression
        2. EDUCATION FRAUD: Degree mills, timeline issues, too many degrees
        3. SKILLS FRAUD: Overinflated abilities, missing foundations, contradictory claims
        4. TIMELINE FRAUD: Overlapping positions, gaps, inconsistent dates
        5. CONTENT FRAUD: Plagiarism, templates, generic descriptions
        6. CONSISTENCY FRAUD: Writing style changes, terminology inconsistencies
        7. ACHIEVEMENT FRAUD: Unrealistic accomplishments, impossible metrics
        8. LOCATION FRAUD: Geographic inconsistencies, impossible relocations

        Resume Text:
        {resume_text}

        {"Job Description for Context: " + job_description if job_description else ""}

        Provide a comprehensive analysis in JSON format:
        {{
            "overall_risk_level": "low|medium|high|critical",
            "overall_confidence": 0.85,
            "category_scores": {{
                "experience": 0.7,
                "education": 0.5,
                "skills": 0.8,
                "timeline": 0.6,
                "content": 0.4,
                "consistency": 0.3,
                "achievements": 0.9,
                "location": 0.2
            }},
            "key_findings": ["most important finding 1", "most important finding 2"],
            "detailed_evidence": {{
                "high_risk_indicators": ["indicator1", "indicator2"],
                "medium_risk_indicators": ["indicator1", "indicator2"],
                "supporting_evidence": ["evidence1", "evidence2"]
            }},
            "recommendations": {{
                "immediate_actions": ["action1", "action2"],
                "verification_needed": ["verify1", "verify2"],
                "interview_focus": ["focus1", "focus2"]
            }},
            "hiring_recommendation": "proceed|investigate|reject",
            "explanation": "Comprehensive explanation of the analysis and reasoning"
        }}
        """

        start_time = time.time()

        try:
            response = self._make_api_call(prompt, "comprehensive_fraud")
            result = self._parse_comprehensive_response(response)
            result.processing_time = time.time() - start_time
            return result

        except Exception as e:
            logger.error(f"Comprehensive fraud analysis failed: {e}")
            return self._create_error_result(AnalysisType.COMPREHENSIVE_FRAUD, str(e))

    def batch_analyze(self, resume_texts: List[str], job_description: Optional[str] = None) -> List[GeminiAnalysisResult]:
        """Perform batch analysis on multiple resumes"""

        results = []

        with ThreadPoolExecutor(max_workers=3) as executor:  # Limit concurrent requests
            futures = []

            for i, resume_text in enumerate(resume_texts):
                future = executor.submit(
                    self.comprehensive_fraud_analysis,
                    resume_text,
                    job_description
                )
                futures.append((i, future))

            for i, future in futures:
                try:
                    result = future.result(timeout=60)  # 60 second timeout
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch analysis failed for resume {i}: {e}")
                    results.append(self._create_error_result(AnalysisType.COMPREHENSIVE_FRAUD, str(e)))

        return results

    def _parse_json_response(self, response: str, analysis_type: AnalysisType) -> GeminiAnalysisResult:
        """Parse JSON response from Gemini"""

        try:
            # Clean the response - sometimes Gemini includes extra text
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")

            # Map risk level
            risk_level_str = data.get('risk_level', 'medium').lower()
            risk_level = RiskLevel(risk_level_str) if risk_level_str in [r.value for r in RiskLevel] else RiskLevel.MEDIUM

            return GeminiAnalysisResult(
                analysis_type=analysis_type,
                risk_level=risk_level,
                confidence=float(data.get('confidence', 0.5)),
                findings=data.get('findings', []),
                evidence=data.get('evidence', {}),
                recommendations=data.get('recommendations', []),
                raw_response=response,
                processing_time=0.0  # Will be set by caller
            )

        except Exception as e:
            logger.error(f"Failed to parse Gemini response: {e}")
            logger.debug(f"Raw response: {response[:500]}...")

            # Fallback parsing
            return self._fallback_parse_response(response, analysis_type)

    def _parse_comprehensive_response(self, response: str) -> GeminiAnalysisResult:
        """Parse comprehensive analysis response"""

        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")

            # Map overall risk level
            risk_level_str = data.get('overall_risk_level', 'medium').lower()
            risk_level = RiskLevel(risk_level_str) if risk_level_str in [r.value for r in RiskLevel] else RiskLevel.MEDIUM

            # Extract findings from comprehensive analysis
            findings = data.get('key_findings', [])

            # Combine evidence from detailed analysis
            evidence = {
                'category_scores': data.get('category_scores', {}),
                'detailed_evidence': data.get('detailed_evidence', {}),
                'hiring_recommendation': data.get('hiring_recommendation', 'investigate')
            }

            # Extract recommendations
            recommendations_data = data.get('recommendations', {})
            recommendations = (
                recommendations_data.get('immediate_actions', []) +
                recommendations_data.get('verification_needed', []) +
                recommendations_data.get('interview_focus', [])
            )

            return GeminiAnalysisResult(
                analysis_type=AnalysisType.COMPREHENSIVE_FRAUD,
                risk_level=risk_level,
                confidence=float(data.get('overall_confidence', 0.5)),
                findings=findings,
                evidence=evidence,
                recommendations=recommendations,
                raw_response=response,
                processing_time=0.0
            )

        except Exception as e:
            logger.error(f"Failed to parse comprehensive response: {e}")
            return self._fallback_parse_response(response, AnalysisType.COMPREHENSIVE_FRAUD)

    def _fallback_parse_response(self, response: str, analysis_type: AnalysisType) -> GeminiAnalysisResult:
        """Fallback parsing when JSON parsing fails"""

        # Simple keyword-based risk assessment
        response_lower = response.lower()

        risk_keywords = {
            'critical': ['critical', 'severe', 'major fraud', 'definitely fraudulent'],
            'high': ['high risk', 'significant concerns', 'likely fraud', 'suspicious'],
            'medium': ['moderate risk', 'some concerns', 'questionable', 'investigate'],
            'low': ['low risk', 'minor concerns', 'acceptable', 'legitimate']
        }

        risk_level = RiskLevel.MEDIUM  # Default
        for level, keywords in risk_keywords.items():
            if any(keyword in response_lower for keyword in keywords):
                risk_level = RiskLevel(level)
                break

        # Extract simple findings
        findings = []
        if 'unrealistic' in response_lower:
            findings.append("Unrealistic claims detected")
        if 'inconsistent' in response_lower:
            findings.append("Inconsistencies found")
        if 'suspicious' in response_lower:
            findings.append("Suspicious patterns identified")

        return GeminiAnalysisResult(
            analysis_type=analysis_type,
            risk_level=risk_level,
            confidence=0.3,  # Low confidence due to parsing issues
            findings=findings or ["Analysis completed with parsing issues"],
            evidence={'parsing_error': True, 'raw_response_preview': response[:200]},
            recommendations=["Manual review recommended due to parsing issues"],
            raw_response=response,
            processing_time=0.0
        )

    def _create_error_result(self, analysis_type: AnalysisType, error_message: str) -> GeminiAnalysisResult:
        """Create error result when analysis fails"""

        return GeminiAnalysisResult(
            analysis_type=analysis_type,
            risk_level=RiskLevel.MEDIUM,  # Default to medium due to uncertainty
            confidence=0.0,
            findings=[f"Analysis failed: {error_message}"],
            evidence={'error': error_message},
            recommendations=["Manual review required due to analysis failure"],
            raw_response="",
            processing_time=0.0
        )

    def get_analysis_summary(self, results: List[GeminiAnalysisResult]) -> Dict[str, Any]:
        """Generate summary from multiple analysis results"""

        if not results:
            return {'error': 'No results to summarize'}

        # Calculate overall risk
        risk_scores = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        avg_risk_score = sum(risk_scores.get(r.risk_level.value, 2) for r in results) / len(results)

        if avg_risk_score >= 3.5:
            overall_risk = RiskLevel.CRITICAL
        elif avg_risk_score >= 2.5:
            overall_risk = RiskLevel.HIGH
        elif avg_risk_score >= 1.5:
            overall_risk = RiskLevel.MEDIUM
        else:
            overall_risk = RiskLevel.LOW

        # Aggregate findings
        all_findings = []
        for result in results:
            all_findings.extend(result.findings)

        # Aggregate recommendations
        all_recommendations = []
        for result in results:
            all_recommendations.extend(result.recommendations)

        return {
            'overall_risk_level': overall_risk.value,
            'average_confidence': sum(r.confidence for r in results) / len(results),
            'total_analyses': len(results),
            'risk_distribution': {
                level.value: sum(1 for r in results if r.risk_level == level)
                for level in RiskLevel
            },
            'key_findings': list(set(all_findings))[:10],  # Top 10 unique findings
            'recommendations': list(set(all_recommendations))[:10],  # Top 10 unique recommendations
            'processing_time_total': sum(r.processing_time for r in results),
            'analysis_types_performed': [r.analysis_type.value for r in results]
        }


# Example usage and testing
if __name__ == "__main__":
    # This would be used for testing the module
    pass
