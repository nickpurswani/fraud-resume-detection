"""
Report Generator for Fraudulent Candidate Detection Tool

This module provides comprehensive reporting capabilities including:
- Executive summary reports
- Detailed technical analysis reports
- Visual analytics and charts
- Multi-format export (JSON, HTML, PDF)
- Batch reporting for multiple candidates
- Comparison reports
- Alert generation
- Custom report templates
"""

import logging
import json
import os
import base64
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from jinja2 import Template, Environment, FileSystemLoader
import pdfkit
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)

class ReportType(Enum):
    """Report type enumeration"""
    EXECUTIVE_SUMMARY = "executive_summary"
    DETAILED_ANALYSIS = "detailed_analysis"
    COMPARISON = "comparison"
    BATCH_SUMMARY = "batch_summary"
    ALERT = "alert"
    DASHBOARD = "dashboard"

class ReportFormat(Enum):
    """Report format enumeration"""
    JSON = "json"
    HTML = "html"
    PDF = "pdf"
    CSV = "csv"
    EXCEL = "xlsx"

class RiskLevel(Enum):
    """Risk level for alerts"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ReportMetadata:
    """Report metadata structure"""
    report_id: str
    report_type: ReportType
    timestamp: datetime
    generated_by: str
    candidate_count: int
    analysis_version: str
    configuration: Dict[str, Any]

class ReportGenerator:
    """Comprehensive report generator for fraud detection results"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize report generator

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.output_dir = Path(self.config.get('output_dir', 'reports'))
        self.templates_dir = Path(self.config.get('templates_dir', 'templates'))

        # Create directories if they don't exist
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.templates_dir.mkdir(exist_ok=True, parents=True)

        # Initialize template environment
        self.template_env = Environment(
            loader=FileSystemLoader(str(self.templates_dir))
        )

        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # Color schemes for different risk levels
        self.risk_colors = {
            'low': '#28a745',
            'medium': '#ffc107',
            'high': '#fd7e14',
            'critical': '#dc3545'
        }

    def generate_executive_summary(self, analysis_results: Dict[str, Any],
                                 candidate_name: str = "Unknown") -> Dict[str, Any]:
        """
        Generate executive summary report

        Args:
            analysis_results: Fraud analysis results
            candidate_name: Name of the candidate

        Returns:
            Executive summary report
        """
        logger.info(f"Generating executive summary for {candidate_name}")

        # Extract key metrics
        fraud_flags = analysis_results.get('fraud_flags', [])
        risk_assessment = analysis_results.get('risk_assessment', {})
        confidence_scores = analysis_results.get('confidence_scores', {})

        # Calculate summary statistics
        total_flags = len(fraud_flags)
        critical_flags = sum(1 for flag in fraud_flags
                           if getattr(flag, 'risk_level', 'medium') == 'critical')
        high_flags = sum(1 for flag in fraud_flags
                        if getattr(flag, 'risk_level', 'medium') == 'high')

        overall_risk = risk_assessment.get('overall_risk', 'unknown')
        risk_score = risk_assessment.get('risk_score', 0.0)

        # Generate recommendation
        recommendation = self._generate_hiring_recommendation(
            overall_risk, total_flags, critical_flags, confidence_scores
        )

        summary = {
            'report_metadata': {
                'report_id': f"ES_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'report_type': ReportType.EXECUTIVE_SUMMARY.value,
                'candidate_name': candidate_name,
                'generated_at': datetime.now().isoformat(),
                'analysis_date': analysis_results.get('timestamp', datetime.now().isoformat())
            },
            'key_findings': {
                'overall_risk_level': overall_risk,
                'risk_score': round(risk_score, 2),
                'total_fraud_flags': total_flags,
                'critical_issues': critical_flags,
                'high_priority_issues': high_flags,
                'authenticity_confidence': round(
                    confidence_scores.get('overall_authenticity', 0.0), 2
                )
            },
            'recommendation': {
                'decision': recommendation['decision'],
                'rationale': recommendation['rationale'],
                'next_steps': recommendation['next_steps'],
                'urgency': recommendation['urgency']
            },
            'top_concerns': self._extract_top_concerns(fraud_flags, limit=3),
            'verification_requirements': self._extract_verification_requirements(fraud_flags),
            'summary_charts': self._generate_summary_charts_data(analysis_results)
        }

        return summary

    def generate_detailed_report(self, analysis_results: Dict[str, Any],
                               linkedin_results: Optional[Dict[str, Any]] = None,
                               fit_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive detailed analysis report

        Args:
            analysis_results: Main fraud analysis results
            linkedin_results: LinkedIn verification results
            fit_results: Job fit analysis results

        Returns:
            Detailed analysis report
        """
        logger.info("Generating detailed analysis report")

        report = {
            'report_metadata': {
                'report_id': f"DA_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'report_type': ReportType.DETAILED_ANALYSIS.value,
                'generated_at': datetime.now().isoformat(),
                'sections_included': ['fraud_analysis']
            },
            'fraud_analysis': self._format_fraud_analysis(analysis_results),
            'risk_breakdown': self._generate_risk_breakdown(analysis_results),
            'timeline_analysis': self._analyze_timeline_consistency(analysis_results),
            'statistical_analysis': self._generate_statistical_analysis(analysis_results)
        }

        # Add LinkedIn analysis if available
        if linkedin_results:
            report['linkedin_verification'] = self._format_linkedin_analysis(linkedin_results)
            report['report_metadata']['sections_included'].append('linkedin_verification')

        # Add fit analysis if available
        if fit_results:
            report['job_fit_analysis'] = self._format_fit_analysis(fit_results)
            report['report_metadata']['sections_included'].append('job_fit_analysis')

        # Add visualizations
        report['visualizations'] = self._generate_detailed_charts(
            analysis_results, linkedin_results, fit_results
        )

        return report

    def generate_comparison_report(self, candidates_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comparison report for multiple candidates

        Args:
            candidates_results: List of candidate analysis results

        Returns:
            Comparison report
        """
        logger.info(f"Generating comparison report for {len(candidates_results)} candidates")

        # Extract comparison metrics
        comparison_data = []
        for result in candidates_results:
            candidate_data = {
                'candidate_id': result.get('candidate_id', 'unknown'),
                'candidate_name': result.get('candidate_name', 'Unknown'),
                'overall_risk': result.get('risk_assessment', {}).get('overall_risk', 'unknown'),
                'risk_score': result.get('risk_assessment', {}).get('risk_score', 0.0),
                'total_flags': len(result.get('fraud_flags', [])),
                'critical_flags': sum(1 for flag in result.get('fraud_flags', [])
                                    if getattr(flag, 'risk_level', 'medium') == 'critical'),
                'authenticity_score': result.get('confidence_scores', {}).get('overall_authenticity', 0.0),
                'linkedin_verified': result.get('linkedin_results', {}).get('verification_status') == 'verified',
                'job_fit_score': result.get('fit_results', {}).get('overall_score', 0.0)
            }
            comparison_data.append(candidate_data)

        # Rank candidates
        ranked_candidates = sorted(
            comparison_data,
            key=lambda x: (x['risk_score'], x['total_flags'])
        )

        # Generate insights
        insights = self._generate_comparison_insights(comparison_data)

        report = {
            'report_metadata': {
                'report_id': f"CR_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'report_type': ReportType.COMPARISON.value,
                'generated_at': datetime.now().isoformat(),
                'candidates_analyzed': len(candidates_results)
            },
            'ranking': [
                {
                    'rank': i + 1,
                    'candidate_id': candidate['candidate_id'],
                    'candidate_name': candidate['candidate_name'],
                    'risk_level': candidate['overall_risk'],
                    'risk_score': round(candidate['risk_score'], 3),
                    'recommendation': self._get_candidate_recommendation(candidate)
                }
                for i, candidate in enumerate(ranked_candidates)
            ],
            'statistics': {
                'total_candidates': len(candidates_results),
                'high_risk_candidates': sum(1 for c in comparison_data if c['risk_score'] > 0.7),
                'verified_linkedin_profiles': sum(1 for c in comparison_data if c['linkedin_verified']),
                'average_risk_score': round(np.mean([c['risk_score'] for c in comparison_data]), 3),
                'candidates_with_critical_flags': sum(1 for c in comparison_data if c['critical_flags'] > 0)
            },
            'insights': insights,
            'comparative_charts': self._generate_comparison_charts(comparison_data)
        }

        return report

    def generate_batch_summary(self, batch_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate batch processing summary report

        Args:
            batch_results: Batch processing results

        Returns:
            Batch summary report
        """
        logger.info("Generating batch summary report")

        results = batch_results.get('results', [])

        # Calculate aggregate statistics
        stats = {
            'total_processed': len(results),
            'successful_analysis': sum(1 for r in results if r.get('status') == 'success'),
            'failed_analysis': sum(1 for r in results if r.get('status') == 'error'),
            'high_risk_detected': sum(1 for r in results
                                    if r.get('risk_assessment', {}).get('overall_risk') in ['high', 'critical']),
            'fraud_flags_total': sum(len(r.get('fraud_flags', [])) for r in results),
            'linkedin_verified': sum(1 for r in results
                                   if r.get('linkedin_results', {}).get('verification_status') == 'verified')
        }

        # Risk distribution
        risk_distribution = {}
        for result in results:
            risk_level = result.get('risk_assessment', {}).get('overall_risk', 'unknown')
            risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + 1

        summary = {
            'report_metadata': {
                'report_id': f"BS_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'report_type': ReportType.BATCH_SUMMARY.value,
                'generated_at': datetime.now().isoformat(),
                'batch_id': batch_results.get('batch_id', 'unknown')
            },
            'processing_summary': stats,
            'risk_distribution': risk_distribution,
            'alerts_generated': self._generate_batch_alerts(results),
            'performance_metrics': {
                'success_rate': round(stats['successful_analysis'] / stats['total_processed'], 3) if stats['total_processed'] > 0 else 0,
                'fraud_detection_rate': round(stats['high_risk_detected'] / stats['successful_analysis'], 3) if stats['successful_analysis'] > 0 else 0,
                'average_processing_time': batch_results.get('average_processing_time', 0)
            },
            'batch_charts': self._generate_batch_charts(results)
        }

        return summary

    def export_report(self, report_data: Dict[str, Any],
                     filename: str, format_type: ReportFormat) -> str:
        """
        Export report in specified format

        Args:
            report_data: Report data to export
            filename: Output filename (without extension)
            format_type: Export format

        Returns:
            Path to exported file
        """
        logger.info(f"Exporting report to {format_type.value} format")

        if format_type == ReportFormat.JSON:
            return self._export_json(report_data, filename)
        elif format_type == ReportFormat.HTML:
            return self._export_html(report_data, filename)
        elif format_type == ReportFormat.PDF:
            return self._export_pdf(report_data, filename)
        elif format_type == ReportFormat.CSV:
            return self._export_csv(report_data, filename)
        elif format_type == ReportFormat.EXCEL:
            return self._export_excel(report_data, filename)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")

    def _generate_hiring_recommendation(self, overall_risk: str, total_flags: int,
                                      critical_flags: int, confidence_scores: Dict[str, float]) -> Dict[str, Any]:
        """Generate hiring recommendation based on analysis"""

        authenticity = confidence_scores.get('overall_authenticity', 0.0)

        if critical_flags > 0:
            decision = "REJECT"
            rationale = f"Critical fraud indicators detected ({critical_flags} critical flags)"
            urgency = "immediate"
            next_steps = ["Do not proceed with hiring", "Document findings", "Alert HR security"]
        elif overall_risk == 'high' or total_flags > 5:
            decision = "INVESTIGATE"
            rationale = f"High risk profile with {total_flags} fraud flags requiring investigation"
            urgency = "high"
            next_steps = ["Conduct thorough background check", "Verify all claims", "Extended interview process"]
        elif overall_risk == 'medium' or total_flags > 2:
            decision = "PROCEED WITH CAUTION"
            rationale = f"Moderate concerns identified, verification recommended"
            urgency = "medium"
            next_steps = ["Verify key claims", "Reference checks", "Probationary period if hired"]
        elif authenticity > 0.8 and total_flags <= 1:
            decision = "PROCEED"
            rationale = "Low fraud risk, candidate appears authentic"
            urgency = "normal"
            next_steps = ["Standard interview process", "Routine reference checks"]
        else:
            decision = "MANUAL REVIEW"
            rationale = "Mixed indicators require human judgment"
            urgency = "medium"
            next_steps = ["Senior reviewer assessment", "Additional verification as needed"]

        return {
            'decision': decision,
            'rationale': rationale,
            'next_steps': next_steps,
            'urgency': urgency
        }

    def _extract_top_concerns(self, fraud_flags: List[Any], limit: int = 3) -> List[Dict[str, Any]]:
        """Extract top fraud concerns"""
        if not fraud_flags:
            return []

        # Sort by severity and confidence
        sorted_flags = sorted(
            fraud_flags,
            key=lambda x: (getattr(x, 'severity_score', 0.5), getattr(x, 'confidence', 0.5)),
            reverse=True
        )

        concerns = []
        for flag in sorted_flags[:limit]:
            concerns.append({
                'type': getattr(flag, 'fraud_type', 'unknown'),
                'description': getattr(flag, 'description', 'No description'),
                'severity': getattr(flag, 'risk_level', 'medium'),
                'confidence': getattr(flag, 'confidence', 0.5)
            })

        return concerns

    def _extract_verification_requirements(self, fraud_flags: List[Any]) -> List[str]:
        """Extract verification requirements based on fraud flags"""
        requirements = set()

        for flag in fraud_flags:
            flag_type = getattr(flag, 'fraud_type', '')
            if 'experience' in flag_type:
                requirements.add("Employment verification")
            elif 'education' in flag_type:
                requirements.add("Educational credential verification")
            elif 'skill' in flag_type:
                requirements.add("Technical skills assessment")
            elif 'timeline' in flag_type:
                requirements.add("Detailed timeline clarification")
            elif 'plagiarism' in flag_type:
                requirements.add("Content originality investigation")

        return list(requirements)

    def _generate_summary_charts_data(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate chart data for executive summary"""
        fraud_flags = analysis_results.get('fraud_flags', [])
        confidence_scores = analysis_results.get('confidence_scores', {})

        # Risk level distribution
        risk_levels = {}
        for flag in fraud_flags:
            risk = getattr(flag, 'risk_level', 'medium')
            risk_levels[risk] = risk_levels.get(risk, 0) + 1

        # Confidence scores radar chart data
        confidence_categories = ['Experience', 'Education', 'Skills', 'Timeline', 'Content']
        confidence_values = [
            confidence_scores.get('experience_authenticity', 1.0),
            confidence_scores.get('education_validity', 1.0),
            confidence_scores.get('skills_alignment', 1.0),
            confidence_scores.get('timeline_consistency', 1.0),
            confidence_scores.get('content_originality', 1.0)
        ]

        return {
            'risk_distribution': risk_levels,
            'confidence_radar': {
                'categories': confidence_categories,
                'values': confidence_values
            }
        }

    def _format_fraud_analysis(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Format fraud analysis for detailed report"""
        fraud_flags = analysis_results.get('fraud_flags', [])

        # Group flags by type
        flags_by_type = {}
        for flag in fraud_flags:
            flag_type = getattr(flag, 'fraud_type', 'unknown')
            if flag_type not in flags_by_type:
                flags_by_type[flag_type] = []

            flags_by_type[flag_type].append({
                'description': getattr(flag, 'description', ''),
                'risk_level': getattr(flag, 'risk_level', 'medium'),
                'confidence': getattr(flag, 'confidence', 0.5),
                'evidence': getattr(flag, 'evidence', {}),
                'recommendation': getattr(flag, 'recommendation', '')
            })

        return {
            'total_flags': len(fraud_flags),
            'flags_by_type': flags_by_type,
            'risk_assessment': analysis_results.get('risk_assessment', {}),
            'confidence_scores': analysis_results.get('confidence_scores', {}),
            'detailed_analysis': analysis_results.get('detailed_analysis', {})
        }

    def _generate_risk_breakdown(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed risk breakdown"""
        fraud_flags = analysis_results.get('fraud_flags', [])

        breakdown = {
            'by_severity': {'critical': 0, 'high': 0, 'medium': 0, 'low': 0},
            'by_category': {},
            'risk_factors': []
        }

        for flag in fraud_flags:
            # Count by severity
            severity = getattr(flag, 'risk_level', 'medium')
            if severity in breakdown['by_severity']:
                breakdown['by_severity'][severity] += 1

            # Count by category
            category = getattr(flag, 'fraud_type', 'unknown')
            breakdown['by_category'][category] = breakdown['by_category'].get(category, 0) + 1

            # Add risk factors
            if severity in ['high', 'critical']:
                breakdown['risk_factors'].append({
                    'factor': getattr(flag, 'description', ''),
                    'impact': severity,
                    'evidence': getattr(flag, 'evidence', {})
                })

        return breakdown

    def _analyze_timeline_consistency(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze timeline consistency"""
        structured_info = analysis_results.get('detailed_analysis', {}).get('structured_info', {})
        experiences = structured_info.get('experiences', [])
        education = structured_info.get('education', [])

        timeline_analysis = {
            'total_experience_years': 0.0,
            'experience_gaps': [],
            'overlapping_periods': [],
            'education_work_alignment': 'unknown'
        }

        # Calculate total experience
        for exp in experiences:
            duration = getattr(exp, 'duration_months', 0) or 0
            timeline_analysis['total_experience_years'] += duration / 12.0

        # Detect gaps and overlaps (simplified)
        timeline_analysis['consistency_score'] = 0.8  # Placeholder

        return timeline_analysis

    def _generate_statistical_analysis(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate statistical analysis of the results"""
        fraud_flags = analysis_results.get('fraud_flags', [])

        if not fraud_flags:
            return {'message': 'No statistical analysis available - no fraud flags detected'}

        # Calculate statistics
        severities = [getattr(flag, 'severity_score', 0.5) for flag in fraud_flags]
        confidences = [getattr(flag, 'confidence', 0.5) for flag in fraud_flags]

        stats = {
            'fraud_flags_statistics': {
                'count': len(fraud_flags),
                'avg_severity': np.mean(severities) if severities else 0,
                'max_severity': np.max(severities) if severities else 0,
                'avg_confidence': np.mean(confidences) if confidences else 0,
                'std_confidence': np.std(confidences) if confidences else 0
            },
            'distribution_analysis': {
                'severity_quartiles': np.percentile(severities, [25, 50, 75]).tolist() if severities else [0, 0, 0],
                'confidence_quartiles': np.percentile(confidences, [25, 50, 75]).tolist() if confidences else [0, 0, 0]
            }
        }

        return stats

    def _format_linkedin_analysis(self, linkedin_results: Dict[str, Any]) -> Dict[str, Any]:
        """Format LinkedIn analysis for report"""
        return {
            'verification_status': linkedin_results.get('verification_status', 'unknown'),
            'match_score': linkedin_results.get('overall_match_score', 0.0),
            'discrepancies_found': len(linkedin_results.get('discrepancies', [])),
            'profile_completeness': linkedin_results.get('profile_completeness', 'unknown'),
            'key_discrepancies': linkedin_results.get('discrepancies', [])[:5]  # Top 5
        }

    def _format_fit_analysis(self, fit_results: Dict[str, Any]) -> Dict[str, Any]:
        """Format job fit analysis for report"""
        return {
            'overall_fit_score': fit_results.get('overall_score', 0.0),
            'fit_level': fit_results.get('fit_level', 'unknown'),
            'qualification_status': fit_results.get('qualification_status', 'unknown'),
            'key_strengths': fit_results.get('strengths', [])[:3],
            'main_gaps': fit_results.get('gaps', [])[:3],
            'red_flags': fit_results.get('red_flags', [])
        }

    def _generate_detailed_charts(self, analysis_results: Dict[str, Any],
                                linkedin_results: Optional[Dict[str, Any]] = None,
                                fit_results: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """Generate charts for detailed report"""
        charts = {}

        # Risk assessment chart
        risk_data = analysis_results.get('risk_assessment', {})
        if risk_data:
            fig = self._create_risk_gauge(risk_data.get('risk_score', 0.0))
            charts['risk_gauge'] = self._fig_to_base64(fig)

        # Fraud flags breakdown
        fraud_flags = analysis_results.get('fraud_flags', [])
        if fraud_flags:
            fig = self._create_fraud_flags_chart(fraud_flags)
            charts['fraud_flags_breakdown'] = self._fig_to_base64(fig)

        # Confidence scores radar
        confidence_scores = analysis_results.get('confidence_scores', {})
        if confidence_scores:
            fig = self._create_confidence_radar(confidence_scores)
            charts['confidence_radar'] = self._fig_to_base64(fig)

        return charts

    def _generate_comparison_insights(self, comparison_data: List[Dict[str, Any]]) -> List[str]:
        """Generate insights from comparison data"""
        insights = []

        risk_scores = [c['risk_score'] for c in comparison_data]
        avg_risk = np.mean(risk_scores)

        if avg_risk > 0.7:
            insights.append("High average risk score indicates potential systemic issues in candidate pool")

        high_risk_count = sum(1 for c in comparison_data if c['risk_score'] > 0.7)
        if high_risk_count > len(comparison_data) * 0.3:
            insights.append(f"{high_risk_count} candidates show high fraud risk - review sourcing channels")

        linkedin_verified = sum(1 for c in comparison_data if c['linkedin_verified'])
        if linkedin_verified < len(comparison_data) * 0.5:
            insights.append("Low LinkedIn verification rate - consider requiring profiles for applications")

        return insights

    def _get_candidate_recommendation(self, candidate_data: Dict[str, Any]) -> str:
        """Get recommendation for individual candidate in comparison"""
        if candidate_data['critical_flags'] > 0:
            return "REJECT"
        elif candidate_data['risk_score'] > 0.7:
            return "HIGH RISK"
        elif candidate_data['risk_score'] > 0.4:
            return "INVESTIGATE"
        else:
            return "LOW RISK"

    def _generate_comparison_charts(self, comparison_data: List[Dict[str, Any]]) -> Dict[str, str]:
        """Generate charts for comparison report"""
        charts = {}

        # Risk score distribution
        fig = self._create_risk_distribution_chart(comparison_data)
        charts['risk_distribution'] = self._fig_to_base64(fig)

        # Candidate ranking
        fig = self._create_candidate_ranking_chart(comparison_data)
        charts['candidate_ranking'] = self._fig_to_base64(fig)

        return charts

    def _generate_batch_alerts(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate alerts from batch results"""
        alerts = []

        # High risk candidates alert
        high_risk_count = sum(1 for r in results
                            if r.get('risk_assessment', {}).get('overall_risk') in ['high', 'critical'])

        if high_risk_count > 0:
            alerts.append({
                'type': 'high_risk_candidates',
                'level': 'critical' if high_risk_count > len(results) * 0.2 else 'high',
                'message': f"{high_risk_count} high-risk candidates detected in batch",
                'action_required': True
            })

        # Processing errors alert
        error_count = sum(1 for r in results if r.get('status') == 'error')
        if error_count > 0:
            alerts.append({
                'type': 'processing_errors',
                'level': 'medium',
                'message': f"{error_count} candidates failed processing",
                'action_required': False
            })

        return alerts

    def _generate_batch_charts(self, results: List[Dict[str, Any]]) -> Dict[str, str]:
        """Generate charts for batch summary"""
        charts = {}

        # Processing status pie chart
        fig = self._create_batch_status_chart(results)
        charts['processing_status'] = self._fig_to_base64(fig)

        # Risk level distribution
        fig = self._create_batch_risk_chart(results)
        charts['risk_distribution'] = self._fig_to_base64(fig)

        return charts

    def _create_risk_gauge(self, risk_score: float) -> go.Figure:
        """Create risk gauge chart"""
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Fraud Risk Score"},
            gauge = {
                'axis': {'range': [None, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.3], 'color': "lightgreen"},
                    {'range': [0.3, 0.7], 'color': "yellow"},
                    {'range': [0.7, 1], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.8
                }
            }
        ))
        return fig

    def _create_fraud_flags_chart(self, fraud_flags: List[Any]) -> go.Figure:
        """Create fraud flags breakdown chart"""
        flag_types = {}
        for flag in fraud_flags:
            flag_type = getattr(flag, 'fraud_type', 'unknown')
            flag_types[flag_type] = flag_types.get(flag_type, 0) + 1

        fig = go.Figure(data=[go.Bar(
            x=list(flag_types.keys()),
            y=list(flag_types.values()),
            marker_color=['red' if v > 2 else 'orange' if v > 1 else 'yellow' for v in flag_types.values()]
        )])

        fig.update_layout(
            title="Fraud Flags by Type",
            xaxis_title="Flag Type",
            yaxis_title="Count"
        )
        return fig

    def _create_confidence_radar(self, confidence_scores: Dict[str, float]) -> go.Figure:
        """Create confidence scores radar chart"""
        categories = ['Experience', 'Education', 'Skills', 'Timeline', 'Content']
        values = [
            confidence_scores.get('experience_authenticity', 1.0),
            confidence_scores.get('education_validity', 1.0),
            confidence_scores.get('skills_alignment', 1.0),
            confidence_scores.get('timeline_consistency', 1.0),
            confidence_scores.get('content_originality', 1.0)
        ]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Confidence Scores'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Authenticity Confidence by Category"
        )
        return fig

    def _create_risk_distribution_chart(self, comparison_data: List[Dict[str, Any]]) -> go.Figure:
        """Create risk distribution chart for comparison"""
        risk_scores = [c['risk_score'] for c in comparison_data]

        fig = go.Figure(data=[go.Histogram(
            x=risk_scores,
            nbinsx=10,
            marker_color='rgba(255, 100, 102, 0.7)',
            marker_line=dict(color='rgba(255, 100, 102, 1.0)', width=1)
        )])

        fig.update_layout(
            title="Risk Score Distribution",
            xaxis_title="Risk Score",
            yaxis_title="Number of Candidates"
        )
        return fig

    def _create_candidate_ranking_chart(self, comparison_data: List[Dict[str, Any]]) -> go.Figure:
        """Create candidate ranking chart"""
        sorted_data = sorted(comparison_data, key=lambda x: x['risk_score'])

        colors = ['green' if score < 0.3 else 'yellow' if score < 0.7 else 'red'
                 for score in [c['risk_score'] for c in sorted_data]]

        fig = go.Figure(data=[go.Bar(
            x=[f"Candidate {i+1}" for i in range(len(sorted_data))],
            y=[c['risk_score'] for c in sorted_data],
            marker_color=colors
        )])

        fig.update_layout(
            title="Candidate Risk Ranking",
            xaxis_title="Candidates",
            yaxis_title="Risk Score"
        )
        return fig

    def _create_batch_status_chart(self, results: List[Dict[str, Any]]) -> go.Figure:
        """Create batch processing status chart"""
        status_counts = {}
        for result in results:
            status = result.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1

        fig = go.Figure(data=[go.Pie(
            labels=list(status_counts.keys()),
            values=list(status_counts.values()),
            hole=0.4
        )])

        fig.update_layout(title="Batch Processing Status")
        return fig

    def _create_batch_risk_chart(self, results: List[Dict[str, Any]]) -> go.Figure:
        """Create batch risk distribution chart"""
        risk_levels = {}
        for result in results:
            risk_level = result.get('risk_assessment', {}).get('overall_risk', 'unknown')
            risk_levels[risk_level] = risk_levels.get(risk_level, 0) + 1

        colors = [self.risk_colors.get(level, 'gray') for level in risk_levels.keys()]

        fig = go.Figure(data=[go.Bar(
            x=list(risk_levels.keys()),
            y=list(risk_levels.values()),
            marker_color=colors
        )])

        fig.update_layout(
            title="Risk Level Distribution",
            xaxis_title="Risk Level",
            yaxis_title="Number of Candidates"
        )
        return fig

    def _fig_to_base64(self, fig: go.Figure) -> str:
        """Convert plotly figure to base64 string"""
        img_bytes = fig.to_image(format="png", width=800, height=600)
        img_base64 = base64.b64encode(img_bytes).decode()
        return f"data:image/png;base64,{img_base64}"

    def _export_json(self, report_data: Dict[str, Any], filename: str) -> str:
        """Export report as JSON"""
        filepath = self.output_dir / f"{filename}.json"

        # Convert datetime objects to strings for JSON serialization
        def json_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2, default=json_serializer)

        logger.info(f"JSON report exported to {filepath}")
        return str(filepath)

    def _export_html(self, report_data: Dict[str, Any], filename: str) -> str:
        """Export report as HTML"""
        filepath = self.output_dir / f"{filename}.html"

        # Load HTML template
        template_content = self._get_html_template()
        template = Template(template_content)

        html_content = template.render(report=report_data)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"HTML report exported to {filepath}")
        return str(filepath)

    def _export_pdf(self, report_data: Dict[str, Any], filename: str) -> str:
        """Export report as PDF"""
        # First create HTML
        html_filepath = self._export_html(report_data, f"{filename}_temp")
        pdf_filepath = self.output_dir / f"{filename}.pdf"

        try:
            # Convert HTML to PDF using pdfkit
            pdfkit.from_file(html_filepath, str(pdf_filepath))

            # Clean up temporary HTML file
            os.remove(html_filepath)

            logger.info(f"PDF report exported to {pdf_filepath}")
            return str(pdf_filepath)
        except Exception as e:
            logger.error(f"PDF export failed: {e}")
            # Return HTML file if PDF fails
            return html_filepath

    def _export_csv(self, report_data: Dict[str, Any], filename: str) -> str:
        """Export report data as CSV"""
        filepath = self.output_dir / f"{filename}.csv"

        # Extract tabular data from report
        rows = []
        report_type = report_data.get('report_metadata', {}).get('report_type', '')

        if report_type == 'comparison':
            # Export ranking data
            ranking = report_data.get('ranking', [])
            for item in ranking:
                rows.append({
                    'Rank': item.get('rank', ''),
                    'Candidate': item.get('candidate_name', ''),
                    'Risk Level': item.get('risk_level', ''),
                    'Risk Score': item.get('risk_score', ''),
                    'Recommendation': item.get('recommendation', '')
                })
        else:
            # Export fraud flags data
            fraud_analysis = report_data.get('fraud_analysis', {})
            flags_by_type = fraud_analysis.get('flags_by_type', {})

            for flag_type, flags in flags_by_type.items():
                for flag in flags:
                    rows.append({
                        'Flag Type': flag_type,
                        'Description': flag.get('description', ''),
                        'Risk Level': flag.get('risk_level', ''),
                        'Confidence': flag.get('confidence', '')
                    })

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(filepath, index=False)
        else:
            # Create empty CSV with headers
            with open(filepath, 'w') as f:
                f.write("No tabular data available for this report type\n")

        logger.info(f"CSV report exported to {filepath}")
        return str(filepath)

    def _export_excel(self, report_data: Dict[str, Any], filename: str) -> str:
        """Export report as Excel workbook"""
        filepath = self.output_dir / f"{filename}.xlsx"

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = []
            if 'key_findings' in report_data:
                findings = report_data['key_findings']
                for key, value in findings.items():
                    summary_data.append({'Metric': key.replace('_', ' ').title(), 'Value': value})

            if summary_data:
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

            # Fraud flags sheet
            fraud_flags = []
            fraud_analysis = report_data.get('fraud_analysis', {})
            flags_by_type = fraud_analysis.get('flags_by_type', {})

            for flag_type, flags in flags_by_type.items():
                for flag in flags:
                    fraud_flags.append({
                        'Type': flag_type,
                        'Description': flag.get('description', ''),
                        'Risk Level': flag.get('risk_level', ''),
                        'Confidence': flag.get('confidence', ''),
                        'Recommendation': flag.get('recommendation', '')
                    })

            if fraud_flags:
                pd.DataFrame(fraud_flags).to_excel(writer, sheet_name='Fraud Flags', index=False)

            # Ranking sheet (for comparison reports)
            if 'ranking' in report_data:
                ranking_df = pd.DataFrame(report_data['ranking'])
                ranking_df.to_excel(writer, sheet_name='Ranking', index=False)

        logger.info(f"Excel report exported to {filepath}")
        return str(filepath)

    def _get_html_template(self) -> str:
        """Get HTML template for report generation"""
        template_file = self.templates_dir / "report_template.html"

        if template_file.exists():
            with open(template_file, 'r', encoding='utf-8') as f:
                return f.read()

        # Default template if file doesn't exist
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Fraud Detection Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #f8f9fa; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; }
                .risk-high { color: #dc3545; font-weight: bold; }
                .risk-medium { color: #fd7e14; font-weight: bold; }
                .risk-low { color: #28a745; font-weight: bold; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .chart { text-align: center; margin: 20px 0; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Fraud Detection Report</h1>
                <p><strong>Generated:</strong> {{ report.report_metadata.generated_at }}</p>
                <p><strong>Report ID:</strong> {{ report.report_metadata.report_id }}</p>
            </div>

            <div class="section">
                <h2>Executive Summary</h2>
                {% if report.key_findings %}
                    <p><strong>Overall Risk:</strong>
                        <span class="risk-{{ report.key_findings.overall_risk_level }}">
                            {{ report.key_findings.overall_risk_level|upper }}
                        </span>
                    </p>
                    <p><strong>Risk Score:</strong> {{ report.key_findings.risk_score }}/1.0</p>
                    <p><strong>Total Fraud Flags:</strong> {{ report.key_findings.total_fraud_flags }}</p>
                {% endif %}
            </div>

            <div class="section">
                <h2>Recommendation</h2>
                {% if report.recommendation %}
                    <p><strong>Decision:</strong> {{ report.recommendation.decision }}</p>
                    <p><strong>Rationale:</strong> {{ report.recommendation.rationale }}</p>
                    <ul>
                    {% for step in report.recommendation.next_steps %}
                        <li>{{ step }}</li>
                    {% endfor %}
                    </ul>
                {% endif %}
            </div>

            <div class="section">
                <h2>Top Concerns</h2>
                {% if report.top_concerns %}
                    <ul>
                    {% for concern in report.top_concerns %}
                        <li>
                            <strong>{{ concern.type }}:</strong> {{ concern.description }}
                            (Severity: {{ concern.severity }}, Confidence: {{ concern.confidence }})
                        </li>
                    {% endfor %}
                    </ul>
                {% endif %}
            </div>
        </body>
        </html>
        """


if __name__ == "__main__":
    # Test the report generator
    generator = ReportGenerator()

    # Sample analysis results for testing
    sample_results = {
        'timestamp': datetime.now().isoformat(),
        'fraud_flags': [
            type('Flag', (), {
                'fraud_type': 'experience_inconsistency',
                'risk_level': 'high',
                'confidence': 0.85,
                'description': 'Unrealistic experience duration',
                'severity_score': 0.8,
                'evidence': {'duration': '120 months'},
                'recommendation': 'Verify employment dates'
            })()
        ],
        'risk_assessment': {
            'overall_risk': 'high',
            'risk_score': 0.75,
            'total_flags': 3
        },
        'confidence_scores': {
            'overall_authenticity': 0.4,
            'experience_authenticity': 0.3,
            'education_validity': 0.8
        }
    }

    # Generate executive summary
    summary = generator.generate_executive_summary(sample_results, "Test Candidate")
    print("Executive Summary generated")
    print(f"Recommendation: {summary['recommendation']['decision']}")

    # Test export
    json_path = generator.export_report(summary, "test_summary", ReportFormat.JSON)
    print(f"Report exported to: {json_path}")
