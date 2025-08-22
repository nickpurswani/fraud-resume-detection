"""
Fraudulent Candidate Detection Tool - Streamlit Application

A comprehensive web application for detecting fraudulent patterns in resumes
and candidate profiles using AI, NLP, and data verification techniques.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import io
from datetime import datetime
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
from typing import Dict, List, Any, Optional

# Import our fraud detection modules
try:
    from src.fraud_detector import FraudDetector, FraudFlag, RiskLevel
    from src.nlp_analyzer import NLPAnalyzer
    from src.linkedin_verifier import LinkedInVerifier
    from src.fit_scorer import FitScorer
    from src.report_generator import ReportGenerator, ReportFormat
    from src.utils import TextExtractor, validate_file_upload, create_sample_data
    from config import Config
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Fraud Detection Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #ffffff;
        border: 2px solid #e1e8ed;
        padding: 1.2rem;
        border-radius: 0.8rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        min-height: 100px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-container h3 {
        color: #1f2937;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    .metric-container p {
        color: #374151;
        font-size: 1.3rem;
        font-weight: 700;
        margin: 0;
    }
    .risk-high {
        color: #dc3545;
        font-weight: bold;
    }
    .risk-medium {
        color: #fd7e14;
        font-weight: bold;
    }
    .risk-low {
        color: #28a745;
        font-weight: bold;
    }
    .risk-critical {
        color: #6f42c1;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stAlert > div {
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'current_candidate' not in st.session_state:
    st.session_state.current_candidate = None

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize fraud detection components"""
    try:
        config = Config()
        validation_result = config.validate_config()

        if validation_result['warnings']:
            for warning in validation_result['warnings']:
                st.sidebar.warning(warning)

        detector = FraudDetector(config.__dict__)
        linkedin_verifier = LinkedInVerifier(config.LINKEDIN_API_KEY, config.__dict__)
        fit_scorer = FitScorer(config.__dict__)
        report_generator = ReportGenerator(config.__dict__)
        text_extractor = TextExtractor()

        return detector, linkedin_verifier, fit_scorer, report_generator, text_extractor
    except Exception as e:
        st.error(f"Error initializing components: {e}")
        return None, None, None, None, None

detector, linkedin_verifier, fit_scorer, report_generator, text_extractor = initialize_components()

if not all([detector, fit_scorer, report_generator, text_extractor]):
    st.stop()

def main():
    """Main application function"""

    # Header
    st.markdown('<h1 class="main-header">🔍 Fraudulent Candidate Detection Tool</h1>',
                unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 1.1rem;">Powered by Google Gemini AI for Advanced Fraud Detection</p>',
                unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["Single Candidate Analysis", "Batch Processing", "Comparison Analysis",
         "LinkedIn Verification", "Settings", "Help & Documentation"]
    )

    if page == "Single Candidate Analysis":
        single_candidate_analysis()
    elif page == "Batch Processing":
        batch_processing()
    elif page == "Comparison Analysis":
        comparison_analysis()
    elif page == "LinkedIn Verification":
        linkedin_verification()
    elif page == "Settings":
        settings_page()
    elif page == "Help & Documentation":
        help_documentation()

def single_candidate_analysis():
    """Single candidate analysis interface"""
    st.header("📄 Single Candidate Analysis")

    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Upload Resume", "Paste Text"])

    # Initialize session state variables for single candidate analysis
    st.session_state.setdefault('single_resume_text', None)
    st.session_state.setdefault('single_candidate_name', "Unknown Candidate")
    st.session_state.setdefault('single_uploaded_file_info', None) # Stores name, size, size_mb if uploaded
    st.session_state.setdefault('single_source_type', None) # 'upload' or 'paste'

    with tab1:
        st.subheader("Upload Resume File")
        uploaded_file = st.file_uploader(
            "Choose a resume file",
            type=['pdf', 'docx', 'txt', 'doc'],
            help="Supported formats: PDF, DOCX, TXT, DOC",
            key="single_resume_uploader" # Unique key for this uploader
        )

        if uploaded_file is not None:
            # Check if this file is different from the one currently stored in session state
            file_info = st.session_state.single_uploaded_file_info
            is_same_file = (file_info and
                            file_info.get('name') == uploaded_file.name and
                            file_info.get('size') == uploaded_file.size)

            if not is_same_file or st.session_state.single_source_type != 'upload':
                # Process new file or if source type changed
                temp_file_path = f"temp_{uploaded_file.name}"
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                validation = validate_file_upload(temp_file_path)

                if validation['valid']:
                    st.session_state.single_resume_text = text_extractor.extract_text(temp_file_path)
                    st.session_state.single_candidate_name = uploaded_file.name.split('.')[0]
                    st.session_state.single_uploaded_file_info = {
                        'name': uploaded_file.name,
                        'size': uploaded_file.size,
                        'size_mb': validation['size_mb']
                    }
                    st.session_state.single_source_type = 'upload'
                    st.success(f"File uploaded successfully ({validation['size_mb']:.2f} MB)")
                else:
                    st.error(validation['error'])
                    st.session_state.single_resume_text = None
                    st.session_state.single_candidate_name = "Unknown Candidate"
                    st.session_state.single_uploaded_file_info = None
                    st.session_state.single_source_type = None

                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
            elif is_same_file and st.session_state.single_source_type == 'upload':
                # Same file re-uploaded, and already processed, just display success
                file_info = st.session_state.single_uploaded_file_info
                st.success(f"File '{file_info['name']}' uploaded successfully ({file_info['size_mb']:.2f} MB)")
        else:
            # If uploader is empty, clear file-related state if previous source was 'upload'
            if st.session_state.single_source_type == 'upload':
                st.session_state.single_resume_text = None
                st.session_state.single_candidate_name = "Unknown Candidate"
                st.session_state.single_uploaded_file_info = None
                st.session_state.single_source_type = None

    with tab2:
        st.subheader("Paste Resume Text")
        # Initialize text area with session state content if the source was 'paste'
        default_paste_value = st.session_state.single_resume_text if st.session_state.single_source_type == 'paste' else ""
        pasted_text_input = st.text_area(
            "Paste resume text here",
            height=300,
            placeholder="Paste the candidate's resume text here...",
            value=default_paste_value,
            key="single_paste_text_area" # Unique key for paste text area
        )

        if pasted_text_input:
            # If text is entered and it's new or the source type has changed
            if (st.session_state.single_source_type != 'paste' or
                st.session_state.single_resume_text != pasted_text_input):

                st.session_state.single_resume_text = pasted_text_input
                st.session_state.single_candidate_name = "Pasted Text Candidate"
                st.session_state.single_source_type = 'paste'
                st.session_state.single_uploaded_file_info = None # Clear file upload info when pasting text
                st.success("Resume text pasted successfully.")
            elif st.session_state.single_source_type == 'paste' and st.session_state.single_resume_text == pasted_text_input:
                # If same text and source, just re-display success on rerun
                st.success("Resume text pasted successfully.")
        else:
            # If text area is empty, and the current source was 'paste', clear the state
            if st.session_state.single_source_type == 'paste':
                st.session_state.single_resume_text = None
                st.session_state.single_candidate_name = "Unknown Candidate"
                st.session_state.single_source_type = None
                st.session_state.single_uploaded_file_info = None
            # If source was 'upload' or None, do not clear paste-specific state.
                st.session_state.single_uploaded_file_info = None

    st.markdown("---") # Separator
    # Always display candidate name input. Its value comes from session state.
    # User can override it, which updates session state.
    candidate_name_from_state = st.session_state.single_candidate_name
    if candidate_name_from_state == "Unknown Candidate" and st.session_state.single_source_type == 'paste':
        candidate_name_from_state = "Pasted Text Candidate" # Default for pasted if not explicitly set
    elif candidate_name_from_state == "Unknown Candidate":
        candidate_name_from_state = "Manual Entry" # Generic default if truly unknown

    candidate_name_input = st.text_input(
        "Candidate Name",
        value=candidate_name_from_state,
        key="single_candidate_name_final_input" # Unique key for this input
    )
    # Update session state if the user has changed the name via this input
    if st.session_state.single_candidate_name != candidate_name_input:
        st.session_state.single_candidate_name = candidate_name_input


    # Job description input (optional)
    st.subheader("📋 Job Description (Optional)")
    job_description = st.text_area(
        "Enter job description for fit analysis:",
        height=150,
        placeholder="About the job\nAre you a passionate Software Developer with expertise in Python, Generative AI Tools, Vibe Coding, JavaScript, Cloud Computing, React, and Generative AI Development? Rahe Solutions is looking for a talented individual like you to join our dynamic team!\n... (rest of your job description)"
    )

    # LinkedIn profile input (optional)
    st.subheader("🔗 LinkedIn Profile (Optional)")
    linkedin_url = st.text_input("LinkedIn Profile URL:", placeholder="https://www.linkedin.com/in/username")

    st.subheader("🔧 Analysis Options")
    col1, col2, col3 = st.columns(3)
    with col1:
        include_linkedin = st.checkbox("LinkedIn Verification", value=bool(linkedin_url))
    with col2:
        include_fit = st.checkbox("Job Fit Analysis", value=bool(job_description))

    # Analysis button
    if st.button("🚀 Analyze Candidate with Gemini AI", type="primary", use_container_width=True):
        # Use the latest values from session state for analysis
        resume_text = st.session_state.single_resume_text
        candidate_name = st.session_state.single_candidate_name

        if not resume_text:
            st.error("Please provide a resume either by uploading a file or pasting text.")
            return

        # Run analysis
        run_single_analysis(resume_text, candidate_name, job_description, linkedin_url,
                            include_linkedin=include_linkedin, include_fit=include_fit)

def run_single_analysis(resume_text, candidate_name, job_description=None,
                       linkedin_url=None, include_nlp=True, include_linkedin=False,
                       include_fit=False):
    """Run comprehensive analysis on a single candidate"""

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Step 1: AI-powered fraud detection
        status_text.text("🤖 Running Gemini AI fraud detection analysis...")
        progress_bar.progress(20)

        # Prepare reference texts for plagiarism detection
        reference_texts = []  # In production, load from database

        analysis_results = detector.analyze_resume(
            resume_text,
            job_description,
            reference_texts
        )

        # Step 2: LinkedIn verification
        linkedin_results = None
        if include_linkedin and linkedin_verifier and linkedin_url:
            status_text.text("🔗 Verifying LinkedIn profile...")
            progress_bar.progress(40)

            try:
                # Extract name for LinkedIn search
                personal_info = analysis_results.get('detailed_analysis', {}).get('nlp', {}).get('analysis_results', {}).get('personal_info', {})
                names = personal_info.get('names', [candidate_name])
                search_name = names[0] if names else candidate_name

                profiles = linkedin_verifier.search_profile_by_name(search_name)
                if profiles:
                    profile = linkedin_verifier.get_profile_details(profiles[0]['id'])
                    if profile:
                        linkedin_results = linkedin_verifier.verify_against_resume(
                            analysis_results.get('detailed_analysis', {}).get('structured_info', {}),
                            profile
                        )
            except Exception as e:
                st.warning(f"LinkedIn verification failed: {e}")

        # Step 3: Job fit analysis
        fit_results = None
        if include_fit and job_description:
            status_text.text("📊 Analyzing job fit...")
            progress_bar.progress(60)

            try:
                fit_analysis = fit_scorer.calculate_fit_score(
                    analysis_results.get('detailed_analysis', {}).get('structured_info', {}),
                    job_description
                )
                fit_results = {
                    'overall_score': fit_analysis.overall_score,
                    'fit_level': fit_analysis.fit_level.value,
                    'qualification_status': fit_analysis.qualification_status.value,
                    'strengths': fit_analysis.strengths,
                    'gaps': fit_analysis.gaps,
                    'red_flags': fit_analysis.red_flags,
                    'recommendations': fit_analysis.recommendations
                }
            except Exception as e:
                st.warning(f"Fit analysis failed: {e}")

        # Step 4: Generate reports
        status_text.text("📋 Generating comprehensive report...")
        progress_bar.progress(80)

        # Store results in session state
        st.session_state.analysis_results = {
            'candidate_name': candidate_name,
            'fraud_analysis': analysis_results,
            'linkedin_results': linkedin_results,
            'fit_results': fit_results,
            'timestamp': datetime.now().isoformat()
        }

        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")

        # Display results
        display_analysis_results(st.session_state.analysis_results)

        # Show Gemini AI insights if available
        if 'gemini_analysis' in st.session_state.analysis_results.get('fraud_analysis', {}):
            display_gemini_insights(st.session_state.analysis_results['fraud_analysis']['gemini_analysis'])

    except Exception as e:
        st.error(f"Analysis failed: {e}")
        progress_bar.empty()
        status_text.empty()

def display_analysis_results(results):
    """Display comprehensive analysis results"""

    st.success("🎉 Analysis Complete!")

    fraud_analysis = results.get('fraud_analysis', {})
    linkedin_results = results.get('linkedin_results')
    fit_results = results.get('fit_results')

    # Executive Summary
    st.header("📊 Executive Summary")

    col1, col2, col3, col4 = st.columns(4)

    risk_assessment = fraud_analysis.get('risk_assessment', {})
    overall_risk = risk_assessment.get('overall_risk', 'unknown')
    risk_score = risk_assessment.get('risk_score', 0.0)
    total_flags = len(fraud_analysis.get('fraud_flags', []))

    with col1:
        risk_class = f"risk-{overall_risk}"
        st.markdown(f'''
            <div class="metric-container">
                <h3>🚨 Risk Level</h3>
                <p class="{risk_class}">{overall_risk.upper()}</p>
            </div>
        ''', unsafe_allow_html=True)

    with col2:
        score_color = "#dc3545" if risk_score > 0.7 else "#fd7e14" if risk_score > 0.4 else "#28a745"
        st.markdown(f'''
            <div class="metric-container">
                <h3>📊 Risk Score</h3>
                <p style="color: {score_color};">{risk_score:.2f}/1.0</p>
            </div>
        ''', unsafe_allow_html=True)

    with col3:
        flag_color = "#dc3545" if total_flags > 3 else "#fd7e14" if total_flags > 1 else "#28a745"
        st.markdown(f'''
            <div class="metric-container">
                <h3>🚩 Fraud Flags</h3>
                <p style="color: {flag_color};">{total_flags}</p>
            </div>
        ''', unsafe_allow_html=True)

    with col4:
        authenticity = fraud_analysis.get('confidence_scores', {}).get('overall_authenticity', 0.0)
        auth_color = "#28a745" if authenticity > 0.7 else "#fd7e14" if authenticity > 0.4 else "#dc3545"
        st.markdown(f'''
            <div class="metric-container">
                <h3>✅ Authenticity</h3>
                <p style="color: {auth_color};">{authenticity:.2f}/1.0</p>
            </div>
        ''', unsafe_allow_html=True)

    # Recommendation
    st.subheader("🎯 Hiring Recommendation")

    if overall_risk == 'critical' or risk_score > 0.8:
        st.error("❌ **DO NOT HIRE** - Critical fraud indicators detected")
    elif overall_risk == 'high' or total_flags > 3:
        st.warning("⚠️ **INVESTIGATE FURTHER** - High risk candidate requiring verification")
    elif overall_risk == 'medium':
        st.info("🔍 **PROCEED WITH CAUTION** - Some concerns identified")
    else:
        st.success("✅ **LOW RISK** - Candidate appears authentic")

    # Detailed Analysis Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🚨 Fraud Flags", "📈 Risk Analysis", "🔬 NLP Analysis", "🤖 Gemini AI", "🔗 LinkedIn", "📋 Job Fit", "📄 Report"
    ])

    with tab1:
        display_fraud_flags(fraud_analysis.get('fraud_flags', []))

    with tab2:
        display_risk_analysis(fraud_analysis)

    with tab3:
        display_nlp_results(fraud_analysis.get('detailed_analysis', {}).get('nlp'))

    with tab4:
        display_gemini_insights(results.get('fraud_analysis', {}).get('gemini_analysis'))

    with tab5:
        display_linkedin_results(linkedin_results)

    with tab6:
        display_fit_results(fit_results)

    with tab7:
        display_report_options(results)

def display_fraud_flags(fraud_flags):
    """Display fraud flags analysis"""
    st.subheader("🚨 Detected Fraud Flags")

    if not fraud_flags:
        st.success("✅ No fraud flags detected!")
        return

    # Group flags by risk level
    flags_by_risk = {}
    for flag in fraud_flags:
        risk_level = getattr(flag, 'risk_level', RiskLevel.MEDIUM)
        risk_key = risk_level.value if hasattr(risk_level, 'value') else str(risk_level)

        if risk_key not in flags_by_risk:
            flags_by_risk[risk_key] = []
        flags_by_risk[risk_key].append(flag)

    # Display flags by risk level
    risk_order = ['critical', 'high', 'medium', 'low']
    for risk_level in risk_order:
        if risk_level in flags_by_risk:
            flags = flags_by_risk[risk_level]

            if risk_level == 'critical':
                st.error(f"🔴 **Critical Issues ({len(flags)})**")
            elif risk_level == 'high':
                st.warning(f"🟠 **High Risk Issues ({len(flags)})**")
            elif risk_level == 'medium':
                st.info(f"🟡 **Medium Risk Issues ({len(flags)})**")
            else:
                st.info(f"🟢 **Low Risk Issues ({len(flags)})**")

            for i, flag in enumerate(flags):
                with st.expander(f"{i+1}. {getattr(flag, 'description', 'No description')}"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write(f"**Type:** {getattr(flag, 'fraud_type', 'Unknown')}")
                        st.write(f"**Confidence:** {getattr(flag, 'confidence', 0.0):.2f}")

                    with col2:
                        st.write(f"**Severity Score:** {getattr(flag, 'severity_score', 0.0):.2f}")
                        st.write(f"**Risk Level:** {getattr(flag, 'risk_level', 'Unknown')}")

                    if hasattr(flag, 'evidence') and flag.evidence:
                        st.write("**Evidence:**")
                        st.json(flag.evidence)

                    if hasattr(flag, 'recommendation'):
                        st.write(f"**Recommendation:** {flag.recommendation}")

def display_risk_analysis(fraud_analysis):
    """Display risk analysis with visualizations"""
    st.subheader("📈 Risk Analysis")

    risk_assessment = fraud_analysis.get('risk_assessment', {})
    confidence_scores = fraud_analysis.get('confidence_scores', {})

    col1, col2 = st.columns(2)

    with col1:
        # Risk gauge
        risk_score = risk_assessment.get('risk_score', 0.0)

        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Overall Risk Score"},
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
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        # Confidence radar chart
        categories = ['Experience', 'Education', 'Skills', 'Timeline', 'Content']
        values = [
            confidence_scores.get('experience_authenticity', 1.0),
            confidence_scores.get('education_validity', 1.0),
            confidence_scores.get('skills_alignment', 1.0),
            confidence_scores.get('timeline_consistency', 1.0),
            confidence_scores.get('content_originality', 1.0)
        ]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Confidence Scores'
        ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            showlegend=False,
            title="Authenticity Confidence",
            height=300
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Risk breakdown
    st.subheader("Risk Breakdown")

    critical_flags = risk_assessment.get('critical_flags', 0)
    high_flags = risk_assessment.get('high_flags', 0)
    medium_flags = risk_assessment.get('medium_flags', 0)
    low_flags = risk_assessment.get('low_flags', 0)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Critical", critical_flags)
    with col2:
        st.metric("High Risk", high_flags)
    with col3:
        st.metric("Medium Risk", medium_flags)
    with col4:
        st.metric("Low Risk", low_flags)

def display_linkedin_results(linkedin_results):
    """Display LinkedIn verification results"""
    st.subheader("🔗 LinkedIn Profile Verification")

    if not linkedin_results:
        st.info("LinkedIn verification was not performed or no profile found.")
        return

    verification_status = linkedin_results.get('verification_status', 'unknown')
    match_score = linkedin_results.get('overall_match_score', 0.0)

    col1, col2 = st.columns(2)

    with col1:
        if verification_status == 'verified':
            st.success(f"✅ **Profile Verified** (Match: {match_score:.2f})")
        elif verification_status == 'partial':
            st.warning(f"⚠️ **Partial Match** (Score: {match_score:.2f})")
        else:
            st.error(f"❌ **Verification Failed** (Score: {match_score:.2f})")

    with col2:
        st.metric("Match Score", f"{match_score:.2f}/1.0")

    # Discrepancies
    discrepancies = linkedin_results.get('discrepancies', [])
    if discrepancies:
        st.subheader("Identified Discrepancies")

        for i, disc in enumerate(discrepancies[:5]):  # Show top 5
            severity = getattr(disc, 'severity', 'medium')
            description = getattr(disc, 'description', 'No description')

            if severity == 'critical':
                st.error(f"🔴 {description}")
            elif severity == 'high':
                st.warning(f"🟠 {description}")
            else:
                st.info(f"🟡 {description}")

def display_fit_results(fit_results):
    """Display job fit analysis results"""
    st.subheader("📋 Job Fit Analysis")

    if not fit_results:
        st.info("Job fit analysis was not performed. Please provide a job description.")
        return

    overall_score = fit_results.get('overall_score', 0.0)
    fit_level = fit_results.get('fit_level', 'unknown')
    qualification_status = fit_results.get('qualification_status', 'unknown')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Fit Score", f"{overall_score:.2f}/1.0")

    with col2:
        fit_color = {
            'excellent': 'success',
            'good': 'success',
            'average': 'info',
            'poor': 'warning',
            'mismatch': 'error'
        }.get(fit_level, 'info')

        if fit_color == 'success':
            st.success(f"✅ {fit_level.title()} Fit")
        elif fit_color == 'warning':
            st.warning(f"⚠️ {fit_level.title()} Fit")
        elif fit_color == 'error':
            st.error(f"❌ {fit_level.title()} Fit")
        else:
            st.info(f"ℹ️ {fit_level.title()} Fit")

    with col3:
        st.write(f"**Status:** {qualification_status.title()}")

    # Strengths and gaps
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💪 Strengths")
        strengths = fit_results.get('strengths', [])
        if strengths:
            for strength in strengths:
                st.success(f"✅ {strength}")
        else:
            st.info("No specific strengths identified")

    with col2:
        st.subheader("⚠️ Gaps")
        gaps = fit_results.get('gaps', [])
        if gaps:
            for gap in gaps:
                st.warning(f"⚠️ {gap}")
        else:
            st.success("No significant gaps identified")

    # Red flags
    red_flags = fit_results.get('red_flags', [])
    if red_flags:
        st.subheader("🚩 Fit-Related Red Flags")
        for flag in red_flags:
            st.error(f"🚩 {flag}")

def display_report_options(results):
    """Display report generation and export options"""
    st.subheader("📄 Generate Reports")

    col1, col2 = st.columns(2)

    with col1:
        report_type = st.selectbox(
            "Report Type",
            ["Executive Summary", "Detailed Analysis", "Comprehensive Report"]
        )

    with col2:
        export_format = st.selectbox(
            "Export Format",
            ["JSON", "HTML", "PDF", "Excel"]
        )

    if st.button("Generate Report", type="primary"):
        try:
            if report_type == "Executive Summary":
                report_data = report_generator.generate_executive_summary(
                    results['fraud_analysis'],
                    results['candidate_name']
                )
            elif report_type == "Detailed Analysis":
                report_data = report_generator.generate_detailed_report(
                    results['fraud_analysis'],
                    results.get('linkedin_results'),
                    results.get('fit_results')
                )
            else:  # Comprehensive Report
                # Combine all analyses
                report_data = {
                    'executive_summary': report_generator.generate_executive_summary(
                        results['fraud_analysis'],
                        results['candidate_name']
                    ),
                    'detailed_analysis': report_generator.generate_detailed_report(
                        results['fraud_analysis'],
                        results.get('linkedin_results'),
                        results.get('fit_results')
                    )
                }

            # Export report
            filename = f"fraud_report_{results['candidate_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            format_enum = getattr(ReportFormat, export_format.upper())

            file_path = report_generator.export_report(report_data, filename, format_enum)

            st.success(f"Report generated successfully: {file_path}")

            # Provide download link
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    file_data = f.read()

                st.download_button(
                    label=f"📥 Download {export_format.upper()} Report",
                    data=file_data,
                    file_name=os.path.basename(file_path),
                    mime=get_mime_type(export_format)
                )

        except Exception as e:
            st.error(f"Report generation failed: {e}")

def get_mime_type(format_type):
    """Get MIME type for download"""
    mime_types = {
        'JSON': 'application/json',
        'HTML': 'text/html',
        'PDF': 'application/pdf',
        'Excel': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    }
    return mime_types.get(format_type, 'application/octet-stream')

def batch_processing():
    """Batch processing interface"""
    st.header("📁 Batch Processing")
    st.write("Upload multiple resumes for batch fraud detection analysis.")

    # File uploader for multiple files
    uploaded_files = st.file_uploader(
        "Choose multiple resume files",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True,
        help="Upload multiple resumes for batch analysis"
    )

    if uploaded_files:
        st.success(f"📁 {len(uploaded_files)} files uploaded")

        # Job description for batch fit analysis
        job_description = st.text_area(
            "Job Description (Optional - for fit analysis):",
            height=100,
            placeholder="Enter job description for batch fit analysis..."
        )

        # Processing options
        col1, col2 = st.columns(2)

        with col1:
            include_linkedin = st.checkbox("Include LinkedIn Verification", value=False)
        with col2:
            include_fit = st.checkbox("Include Job Fit Analysis", value=bool(job_description))

        if st.button("🚀 Start Batch Processing", type="primary"):
            run_batch_processing(uploaded_files, job_description, include_linkedin, include_fit)

def run_batch_processing(uploaded_files, job_description=None, include_linkedin=False, include_fit=False):
    """Run batch processing on multiple files"""

    progress_bar = st.progress(0)
    status_text = st.empty()

    results = []
    total_files = len(uploaded_files)

    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name} ({i+1}/{total_files})...")

        try:
            # Extract text
            temp_file_path = f"temp_batch_{uploaded_file.name}"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            resume_text = text_extractor.extract_text(temp_file_path)
            candidate_name = uploaded_file.name.split('.')[0]

            # Run analysis
            analysis_result = detector.analyze_resume(resume_text, job_description)

            # Optional LinkedIn verification
            linkedin_result = None
            if include_linkedin:
                # Simplified for batch processing
                pass

            # Optional fit analysis
            fit_result = None
            if include_fit and job_description:
                fit_analysis = fit_scorer.calculate_fit_score(
                    analysis_result.get('detailed_analysis', {}).get('structured_info', {}),
                    job_description
                )
                fit_result = {
                    'overall_score': fit_analysis.overall_score,
                    'fit_level': fit_analysis.fit_level.value,
                    'qualification_status': fit_analysis.qualification_status.value
                }

            results.append({
                'candidate_name': candidate_name,
                'file_name': uploaded_file.name,
                'fraud_analysis': analysis_result,
                'linkedin_results': linkedin_result,
                'fit_results': fit_result,
                'status': 'success'
            })

            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

        except Exception as e:
            results.append({
                'candidate_name': uploaded_file.name.split('.')[0],
                'file_name': uploaded_file.name,
                'error': str(e),
                'status': 'error'
            })

        # Update progress
        progress_bar.progress((i + 1) / total_files)

    status_text.text("✅ Batch processing complete!")

    # Display results
    display_batch_results(results, job_description)


def display_batch_results(results, job_description=None):
    """Display batch processing results"""
    st.subheader("📊 Batch Processing Results")

    # Summary statistics
    total_processed = len(results)
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = total_processed - successful

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Processed", total_processed)
    with col2:
        st.metric("Successful", successful)
    with col3:
        st.metric("Failed", failed)

    if successful > 0:
        # Risk distribution
        risk_levels = {}
        for result in results:
            if result['status'] == 'success':
                risk = result['fraud_analysis'].get('risk_assessment', {}).get('overall_risk', 'unknown')
                risk_levels[risk] = risk_levels.get(risk, 0) + 1

        # Display risk distribution chart
        if risk_levels:
            fig = px.pie(
                values=list(risk_levels.values()),
                names=list(risk_levels.keys()),
                title="Risk Level Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Detailed results table
        st.subheader("Detailed Results")

        table_data = []
        for result in results:
            if result['status'] == 'success':
                fraud_analysis = result['fraud_analysis']
                risk_assessment = fraud_analysis.get('risk_assessment', {})

                table_data.append({
                    'Candidate': result['candidate_name'],
                    'Risk Level': risk_assessment.get('overall_risk', 'unknown').title(),
                    'Risk Score': f"{risk_assessment.get('risk_score', 0.0):.2f}",
                    'Total Flags': len(fraud_analysis.get('fraud_flags', [])),
                    'Authenticity': f"{fraud_analysis.get('confidence_scores', {}).get('overall_authenticity', 0.0):.2f}"
                })

        if table_data:
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True)

        # Export batch results
        st.subheader("Export Batch Results")
        if st.button("📥 Export Batch Report"):
            try:
                batch_report = report_generator.generate_batch_summary({
                    'results': results,
                    'batch_id': f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'timestamp': datetime.now().isoformat()
                })

                filename = f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                file_path = report_generator.export_report(batch_report, filename, ReportFormat.JSON)

                with open(file_path, 'rb') as f:
                    st.download_button(
                        label="📥 Download Batch Report",
                        data=f.read(),
                        file_name=os.path.basename(file_path),
                        mime="application/json"
                    )

            except Exception as e:
                st.error(f"Export failed: {e}")


def comparison_analysis():
    """Comparison analysis interface"""
    st.header("⚖️ Comparison Analysis")
    st.write("Compare multiple candidates side by side.")

    # Check if we have batch results to compare
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = []

    # Option to load previous batch results or upload new files
    tab1, tab2 = st.tabs(["Use Previous Batch", "Upload New Files"])

    candidates_data = []

    with tab1:
        if st.session_state.batch_results:
            st.success(f"Found {len(st.session_state.batch_results)} candidates from previous batch")
            if st.button("Use Previous Batch Results"):
                candidates_data = st.session_state.batch_results
        else:
            st.info("No previous batch results found. Use the 'Upload New Files' tab.")

    with tab2:
        uploaded_files = st.file_uploader(
            "Upload resumes for comparison",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True
        )

        if uploaded_files and len(uploaded_files) >= 2:
            job_description = st.text_area(
                "Job Description (for fit comparison):",
                height=100
            )

            if st.button("Analyze for Comparison"):
                candidates_data = process_comparison_files(uploaded_files, job_description)

    if candidates_data and len(candidates_data) >= 2:
        display_comparison_results(candidates_data)
    elif len(candidates_data) == 1:
        st.warning("Need at least 2 candidates for comparison analysis.")


def process_comparison_files(uploaded_files, job_description=None):
    """Process files for comparison analysis"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    candidates_data = []

    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name}...")

        try:
            # Extract text
            temp_file_path = f"temp_compare_{uploaded_file.name}"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            resume_text = text_extractor.extract_text(temp_file_path)
            candidate_name = uploaded_file.name.split('.')[0]

            # Run analysis
            fraud_analysis = detector.analyze_resume(resume_text, job_description)

            # Optional fit analysis
            fit_result = None
            if job_description:
                fit_analysis = fit_scorer.calculate_fit_score(
                    fraud_analysis.get('detailed_analysis', {}).get('structured_info', {}),
                    job_description
                )
                fit_result = {
                    'overall_score': fit_analysis.overall_score,
                    'fit_level': fit_analysis.fit_level.value,
                    'qualification_status': fit_analysis.qualification_status.value
                }

            candidates_data.append({
                'candidate_name': candidate_name,
                'fraud_analysis': fraud_analysis,
                'fit_results': fit_result
            })

            # Clean up
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")

        progress_bar.progress((i + 1) / len(uploaded_files))

    status_text.text("✅ Comparison analysis complete!")
    return candidates_data


def display_comparison_results(candidates_data):
    """Display comparison analysis results"""
    st.subheader("📊 Candidate Comparison")

    # Create comparison table
    comparison_data = []
    for candidate in candidates_data:
        fraud_analysis = candidate['fraud_analysis']
        fit_results = candidate.get('fit_results')
        risk_assessment = fraud_analysis.get('risk_assessment', {})

        comparison_data.append({
            'Candidate': candidate['candidate_name'],
            'Risk Level': risk_assessment.get('overall_risk', 'unknown').title(),
            'Risk Score': risk_assessment.get('risk_score', 0.0),
            'Total Flags': len(fraud_analysis.get('fraud_flags', [])),
            'Authenticity': fraud_analysis.get('confidence_scores', {}).get('overall_authenticity', 0.0),
            'Fit Score': fit_results.get('overall_score', 0.0) if fit_results else 0.0,
            'Fit Level': fit_results.get('fit_level', 'N/A') if fit_results else 'N/A'
        })

    # Sort by risk score (ascending - lower risk is better)
    comparison_data.sort(key=lambda x: x['Risk Score'])

    # Display ranking
    st.subheader("🏆 Candidate Ranking")
    for i, candidate in enumerate(comparison_data):
        rank = i + 1
        risk_class = f"risk-{candidate['Risk Level'].lower()}"

        col1, col2, col3, col4 = st.columns([1, 3, 2, 2])

        with col1:
            st.markdown(f"**#{rank}**")

        with col2:
            st.markdown(f"**{candidate['Candidate']}**")

        with col3:
            st.markdown(f'<span class="{risk_class}">{candidate["Risk Level"]}</span>',
                       unsafe_allow_html=True)

        with col4:
            st.write(f"Risk: {candidate['Risk Score']:.2f}")

    # Detailed comparison chart
    st.subheader("📈 Risk Comparison Chart")

    fig = go.Figure(data=[
        go.Bar(
            x=[c['Candidate'] for c in comparison_data],
            y=[c['Risk Score'] for c in comparison_data],
            marker_color=['red' if score > 0.7 else 'orange' if score > 0.3 else 'green'
                         for score in [c['Risk Score'] for c in comparison_data]]
        )
    ])

    fig.update_layout(
        title="Risk Score Comparison",
        xaxis_title="Candidates",
        yaxis_title="Risk Score"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Detailed comparison table
    st.subheader("📋 Detailed Comparison")
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True)


def linkedin_verification():
    """LinkedIn verification interface"""
    st.header("🔗 LinkedIn Profile Verification")
    st.write("Verify candidate information against LinkedIn profiles.")

    # Input options
    col1, col2 = st.columns(2)

    with col1:
        candidate_name = st.text_input("Candidate Name", placeholder="John Smith")
        company_name = st.text_input("Current/Recent Company (Optional)", placeholder="Google")

    with col2:
        linkedin_url = st.text_input("LinkedIn Profile URL (Optional)",
                                   placeholder="https://www.linkedin.com/in/username")

    # Resume input
    st.subheader("📄 Resume Information")
    resume_input_method = st.radio(
        "Resume Input Method:",
        ["Upload File", "Paste Text"]
    )

    resume_text = None

    if resume_input_method == "Upload File":
        uploaded_file = st.file_uploader("Upload Resume", type=['pdf', 'docx', 'txt'])
        if uploaded_file:
            temp_file_path = f"temp_linkedin_{uploaded_file.name}"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            resume_text = text_extractor.extract_text(temp_file_path)

            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    else:
        resume_text = st.text_area("Paste Resume Text", height=200)

    # Verification button
    if st.button("🔍 Verify LinkedIn Profile", type="primary"):
        if not candidate_name:
            st.error("Please provide a candidate name.")
            return

        if not resume_text:
            st.error("Please provide resume information.")
            return

        run_linkedin_verification(candidate_name, company_name, linkedin_url, resume_text)


def run_linkedin_verification(candidate_name, company_name, linkedin_url, resume_text):
    """Run LinkedIn verification process"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Step 1: Parse resume
        status_text.text("📝 Analyzing resume...")
        progress_bar.progress(25)

        fraud_analysis = detector.analyze_resume(resume_text)
        structured_info = fraud_analysis.get('detailed_analysis', {}).get('structured_info', {})

        # Step 2: Search LinkedIn profile
        status_text.text("🔍 Searching LinkedIn profile...")
        progress_bar.progress(50)

        profiles = linkedin_verifier.search_profile_by_name(candidate_name, company_name)

        if not profiles:
            st.warning("❌ No LinkedIn profiles found matching the search criteria.")
            progress_bar.empty()
            status_text.empty()
            return

        # Step 3: Get detailed profile
        status_text.text("📊 Analyzing LinkedIn profile...")
        progress_bar.progress(75)

        profile = linkedin_verifier.get_profile_details(profiles[0]['id'])

        if not profile:
            st.error("❌ Failed to retrieve LinkedIn profile details.")
            progress_bar.empty()
            status_text.empty()
            return

        # Step 4: Verify against resume
        status_text.text("⚖️ Comparing resume with LinkedIn...")
        progress_bar.progress(100)

        verification_results = linkedin_verifier.verify_against_resume(structured_info, profile)

        status_text.text("✅ LinkedIn verification complete!")

        # Display results
        display_linkedin_verification_results(verification_results, profile)

    except Exception as e:
        st.error(f"LinkedIn verification failed: {e}")
        progress_bar.empty()
        status_text.empty()


def display_linkedin_verification_results(verification_results, profile):
    """Display LinkedIn verification results"""
    st.subheader("🔗 LinkedIn Verification Results")

    # Overall verification status
    verification_status = verification_results.get('verification_status', 'unknown')
    match_score = verification_results.get('overall_match_score', 0.0)

    col1, col2, col3 = st.columns(3)

    with col1:
        if verification_status.value == 'verified':
            st.success("✅ **VERIFIED**")
        elif verification_status.value == 'partial':
            st.warning("⚠️ **PARTIAL MATCH**")
        else:
            st.error("❌ **NOT VERIFIED**")

    with col2:
        st.metric("Match Score", f"{match_score:.2f}/1.0")

    with col3:
        confidence = verification_results.get('confidence_scores', {}).get('overall_confidence', 0.0)
        st.metric("Confidence", f"{confidence:.2f}/1.0")

    # Profile information
    st.subheader("👤 LinkedIn Profile Information")

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Name:** {profile.full_name}")
        st.write(f"**Headline:** {profile.headline}")
        st.write(f"**Location:** {profile.location}")

    with col2:
        st.write(f"**Experience Entries:** {len(profile.experience)}")
        st.write(f"**Education Entries:** {len(profile.education)}")
        st.write(f"**Skills Listed:** {len(profile.skills)}")

    # Discrepancies
    discrepancies = verification_results.get('discrepancies', [])
    if discrepancies:
        st.subheader("⚠️ Identified Discrepancies")

        for disc in discrepancies:
            severity = getattr(disc, 'severity', 'medium')
            description = getattr(disc, 'description', 'No description')

            if severity == 'critical':
                st.error(f"🔴 **Critical:** {description}")
            elif severity == 'high':
                st.warning(f"🟠 **High:** {description}")
            else:
                st.info(f"🟡 **Medium:** {description}")

            # Show more details in expander
            with st.expander("View Details"):
                st.write(f"**Resume Value:** {getattr(disc, 'resume_value', 'N/A')}")
                st.write(f"**LinkedIn Value:** {getattr(disc, 'linkedin_value', 'N/A')}")
                st.write(f"**Confidence:** {getattr(disc, 'confidence', 0.0):.2f}")
                st.write(f"**Suggestion:** {getattr(disc, 'suggestion', 'No suggestion')}")
    else:
        st.success("✅ No significant discrepancies found!")


def settings_page():
    """Settings and configuration page"""
    st.header("⚙️ Settings & Configuration")

    # API Configuration
    st.subheader("🔑 API Configuration")

    col1, col2 = st.columns(2)

    with col1:
        gemini_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            help="Required for advanced AI-powered fraud detection"
        )

    with col2:
        linkedin_key = st.text_input(
            "LinkedIn API Key",
            type="password",
            help="Required for LinkedIn profile verification"
        )

    # Detection Thresholds
    st.subheader("🎯 Detection Thresholds")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Fraud Detection Thresholds**")
        experience_threshold = st.slider("Experience Inconsistency", 0.0, 1.0, 0.7, 0.1)
        education_threshold = st.slider("Education Mismatch", 0.0, 1.0, 0.6, 0.1)
        skill_threshold = st.slider("Skill-Experience Gap", 0.0, 1.0, 0.8, 0.1)

    with col2:
        st.write("**Verification Thresholds**")
        plagiarism_threshold = st.slider("Plagiarism Similarity", 0.0, 1.0, 0.85, 0.05)
        timeline_threshold = st.slider("Timeline Inconsistency", 0.0, 1.0, 0.7, 0.1)
        location_threshold = st.slider("Location Discrepancy", 0.0, 1.0, 0.6, 0.1)

    # Processing Options
    st.subheader("⚡ Processing Options")

    col1, col2 = st.columns(2)

    with col1:
        max_file_size = st.number_input("Max File Size (MB)", min_value=1, max_value=50, value=10)
        batch_size = st.number_input("Batch Processing Size", min_value=1, max_value=100, value=20)

    with col2:
        enable_caching = st.checkbox("Enable Caching", value=True)
        enable_logging = st.checkbox("Enable Detailed Logging", value=True)

    # Export Settings
    st.subheader("📤 Export Settings")

    default_format = st.selectbox(
        "Default Export Format",
        ["JSON", "HTML", "PDF", "Excel"]
    )

    include_charts = st.checkbox("Include Charts in Reports", value=True)
    include_raw_data = st.checkbox("Include Raw Analysis Data", value=False)

    # Save settings
    if st.button("💾 Save Settings", type="primary"):
        settings = {
            'api_keys': {
                'gemini': gemini_key,
                'linkedin': linkedin_key
            },
            'thresholds': {
                'experience_inconsistency': experience_threshold,
                'education_mismatch': education_threshold,
                'skill_experience_gap': skill_threshold,
                'plagiarism_similarity': plagiarism_threshold,
                'timeline_inconsistency': timeline_threshold,
                'location_discrepancy': location_threshold
            },
            'processing': {
                'max_file_size_mb': max_file_size,
                'batch_size': batch_size,
                'enable_caching': enable_caching,
                'enable_logging': enable_logging
            },
            'export': {
                'default_format': default_format,
                'include_charts': include_charts,
                'include_raw_data': include_raw_data
            }
        }

        # Save to session state (in production, save to file or database)
        st.session_state.user_settings = settings
        st.success("✅ Settings saved successfully!")


def help_documentation():
    """Help and documentation page"""
    st.header("📚 Help & Documentation")

    # Create tabs for different help sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Getting Started", "Features", "Fraud Detection", "Best Practices", "FAQ"
    ])

    with tab1:
        st.subheader("🚀 Getting Started")
        st.markdown("""
        ### Welcome to the Fraudulent Candidate Detection Tool!

        This comprehensive tool helps you identify potential fraud in resumes and candidate profiles using advanced AI and NLP techniques.

        #### Quick Start:
        1. **Single Analysis**: Upload a resume or paste text, optionally add a job description
        2. **Batch Processing**: Upload multiple resumes for bulk analysis
        3. **LinkedIn Verification**: Cross-check resume information with LinkedIn profiles
        4. **Comparison Analysis**: Compare multiple candidates side-by-side

        #### Supported File Formats:
        - PDF (.pdf)
        - Microsoft Word (.docx, .doc)
        - Plain Text (.txt)

        #### Key Features:
        - ✅ Google Gemini AI-powered fraud detection
        - ✅ LinkedIn profile verification
        - ✅ Job fit analysis
        - ✅ Comprehensive reporting
        - ✅ Batch processing capabilities
        - ✅ Advanced natural language understanding
        """)

    with tab2:
        st.subheader("🔧 Features Overview")
        st.markdown("""
        ### Core Features

        #### 1. Fraud Detection Engine
        - Experience inconsistencies
        - Education mismatches
        - Timeline anomalies
        - Skill-experience gaps
        - Content plagiarism detection
        - Career progression analysis

        #### 2. LinkedIn Verification
        - Profile matching and verification
        - Experience cross-validation
        - Skills comparison
        - Timeline consistency checks

        #### 3. Job Fit Analysis
        - Skills matching
        - Experience level assessment
        - Education requirements verification
        - Qualification scoring

        #### 4. Advanced AI Analysis with Gemini
        - Intelligent fraud pattern recognition
        - Context-aware inconsistency detection
        - Natural language understanding
        - Advanced reasoning and analysis

        #### 5. Comprehensive Reporting
        - Executive summaries
        - Detailed analysis reports
        - Visual analytics and charts
        - Multiple export formats
        """)

    with tab3:
        st.subheader("🔍 Fraud Detection Indicators")
        st.markdown("""
        ### What We Detect

        #### Experience-Related Fraud
        - **Inflated Job Titles**: Senior roles with insufficient experience
        - **Unrealistic Durations**: Impossibly long tenures
        - **Timeline Inconsistencies**: Overlapping employment periods
        - **Career Progression Anomalies**: Unusual jumps in responsibility

        #### Education-Related Fraud
        - **Degree Mills**: Non-accredited institutions
        - **Timeline Mismatches**: Graduation dates inconsistent with experience
        - **Overqualification**: Degrees that don't match career trajectory

        #### Skills-Related Fraud
        - **Experience Gaps**: Advanced skills claimed without relevant experience
        - **Impossible Combinations**: Contradictory skill claims
        - **Buzzword Overload**: Excessive use of trendy terms

        #### Content-Related Fraud
        - **Plagiarism**: Copied job descriptions or achievements
        - **Template Resumes**: Generic, non-personalized content
        - **Inconsistent Writing**: Varying quality suggesting multiple authors

        ### Risk Levels
        - 🟢 **Low Risk**: Minor concerns, proceed with standard process
        - 🟡 **Medium Risk**: Some inconsistencies, additional verification recommended
        - 🟠 **High Risk**: Significant concerns, thorough investigation required
        - 🔴 **Critical Risk**: Major fraud indicators, recommend rejection
        """)

    with tab4:
        st.subheader("✅ Best Practices")
        st.markdown("""
        ### Recommended Workflow

        #### 1. Initial Screening
        - Run fraud detection on all incoming resumes
        - Focus on high-risk candidates first
        - Use batch processing for efficiency

        #### 2. Verification Process
        - Verify LinkedIn profiles for medium+ risk candidates
        - Cross-check education credentials
        - Contact previous employers for high-risk cases

        #### 3. Interview Preparation
        - Use fit analysis to identify focus areas
        - Prepare specific questions about flagged items
        - Document all verification attempts

        #### 4. Decision Making
        - Consider all factors, not just fraud scores
        - Use human judgment for edge cases
        - Maintain audit trail of decisions

        ### Integration Tips
        - Integrate with your ATS for seamless workflow
        - Set up automated alerts for critical cases
        - Regular review and calibration of thresholds
        - Train hiring team on tool interpretation

        ### Legal Considerations
        - Ensure compliance with local employment laws
        - Maintain candidate privacy and data protection
        - Document decision rationale
        - Provide opportunity for candidate explanation
        """)

    with tab5:
        st.subheader("❓ Frequently Asked Questions")
        st.markdown("""
        ### Common Questions

        **Q: How accurate is the fraud detection?**
        A: The tool provides indicators and risk scores. It should be used as a screening aid, not a definitive judgment. Always combine with human review.

        **Q: Can I customize the detection thresholds?**
        A: Yes! Go to Settings to adjust thresholds based on your organization's risk tolerance.

        **Q: What if a candidate doesn't have a LinkedIn profile?**
        A: LinkedIn verification is optional. The tool can still perform comprehensive fraud detection without it.

        **Q: How should I handle false positives?**
        A: Review flagged items with candidates during interviews. Many "inconsistencies" have legitimate explanations.

        **Q: Is my data secure?**
        A: Yes, all processing is done locally. We recommend reviewing your organization's data handling policies.

        **Q: Can I export reports for record-keeping?**
        A: Absolutely! Reports can be exported in JSON, HTML, PDF, and Excel formats.

        **Q: What file formats are supported?**
        A: PDF, DOCX, DOC, and TXT files are supported. Maximum file size is configurable (default 10MB).

        **Q: How do I interpret risk scores?**
        A: Risk scores range from 0.0 (no risk) to 1.0 (maximum risk). Use the risk level categories as primary indicators.

        ### Need More Help?
        - Check the tooltips and help text throughout the application
        - Review the detailed analysis reports for explanations
        - Consider the specific evidence provided with each flag
        """)

    # Sample data section
    st.subheader("📝 Try It Out!")
    if st.button("Load Sample Data"):
        sample_resume, sample_job_desc = create_sample_data()

        st.success("Sample data loaded! You can copy this text to try the Gemini AI-powered tool.")

        col1, col2 = st.columns(2)

        with col1:
            st.text_area("Sample Resume", sample_resume, height=300)

        with col2:
            st.text_area("Sample Job Description", sample_job_desc, height=300)

def display_nlp_results(nlp_analysis):
    """Display NLP analysis results"""
    st.subheader("🔬 Natural Language Processing Analysis")

    if not nlp_analysis:
        st.info("NLP analysis data is not available.")
        return

    # Analysis results overview
    analysis_results = nlp_analysis.get('analysis_results', {})

    if analysis_results:
        col1, col2, col3 = st.columns(3)

        with col1:
            personal_info = analysis_results.get('personal_info', {})
            st.markdown("""
                <div class="metric-container">
                    <h3>👤 Personal Information</h3>
                    <p>Names: {}</p>
                    <p>Emails: {}</p>
                    <p>Phones: {}</p>
                </div>
            """.format(
                len(personal_info.get('names', [])),
                len(personal_info.get('emails', [])),
                len(personal_info.get('phones', []))
            ), unsafe_allow_html=True)

        with col2:
            work_exp = analysis_results.get('work_experience', [])
            education = analysis_results.get('education', [])
            st.markdown(f"""
                <div class="metric-container">
                    <h3>🏢 Experience & Education</h3>
                    <p>Work Experience: {len(work_exp)}</p>
                    <p>Education: {len(education)}</p>
                </div>
            """, unsafe_allow_html=True)

        with col3:
            skills = analysis_results.get('skills', [])
            certifications = analysis_results.get('certifications', [])
            st.markdown(f"""
                <div class="metric-container">
                    <h3>🎯 Skills & Certifications</h3>
                    <p>Skills: {len(skills)}</p>
                    <p>Certifications: {len(certifications)}</p>
                </div>
            """, unsafe_allow_html=True)

    # Text statistics
    text_stats = nlp_analysis.get('text_statistics', {})
    if text_stats:
        st.subheader("📝 Text Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Word Count", text_stats.get('word_count', 0))
        with col2:
            st.metric("Sentence Count", text_stats.get('sentence_count', 0))
        with col3:
            st.metric("Reading Level", f"{text_stats.get('flesch_reading_ease', 0):.1f}")
        with col4:
            st.metric("Avg Sentence Length", f"{text_stats.get('avg_sentence_length', 0):.1f}")

    # Language quality
    language_quality = nlp_analysis.get('language_quality', {})
    if language_quality:
        st.subheader("📊 Language Quality Analysis")

        quality_metrics = {
            'Grammar Score': language_quality.get('grammar_score', 0),
            'Vocabulary Richness': language_quality.get('vocabulary_richness', 0),
            'Coherence Score': language_quality.get('coherence_score', 0),
            'Professional Tone': language_quality.get('professional_tone_score', 0)
        }

        for metric, score in quality_metrics.items():
            progress_color = "#28a745" if score > 0.7 else "#fd7e14" if score > 0.4 else "#dc3545"
            st.progress(score, text=f"{metric}: {score:.2f}")

    # Sentiment analysis
    sentiment = nlp_analysis.get('sentiment_analysis', {})
    if sentiment:
        st.subheader("😊 Sentiment Analysis")
        sentiment_label = sentiment.get('label', 'neutral')
        sentiment_score = sentiment.get('score', 0.0)

        if sentiment_label == 'POSITIVE':
            st.success(f"✅ Positive sentiment detected (confidence: {sentiment_score:.2f})")
        elif sentiment_label == 'NEGATIVE':
            st.error(f"❌ Negative sentiment detected (confidence: {sentiment_score:.2f})")
        else:
            st.info(f"😐 Neutral sentiment detected (confidence: {sentiment_score:.2f})")


def display_gemini_insights(gemini_analysis):
    """Display Gemini AI specific insights"""
    st.subheader("🤖 Gemini AI Advanced Analysis")

    if not gemini_analysis:
        st.info("Gemini AI analysis was not performed. Configure your Gemini API key in Settings.")
        return

    if 'error' in gemini_analysis:
        st.error(f"Gemini AI analysis failed: {gemini_analysis['error']}")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("AI Risk Level", gemini_analysis.get('risk_level', 'unknown').title())

    with col2:
        st.metric("AI Confidence", f"{gemini_analysis.get('confidence', 0.0):.2f}")

    with col3:
        processing_time = gemini_analysis.get('processing_time', 0.0)
        st.metric("Analysis Time", f"{processing_time:.1f}s")

    # AI Findings
    findings = gemini_analysis.get('findings', [])
    if findings:
        st.subheader("🔍 AI-Detected Patterns")
        for i, finding in enumerate(findings, 1):
            st.write(f"**{i}.** {finding}")

    # Evidence
    evidence = gemini_analysis.get('evidence', {})
    if evidence and evidence != {'parsing_error': True}:
        st.subheader("📋 Supporting Evidence")

        # Category scores if available
        if 'category_scores' in evidence:
            st.write("**Risk by Category:**")
            category_scores = evidence['category_scores']
            for category, score in category_scores.items():
                risk_color = "🔴" if score > 0.7 else "🟡" if score > 0.4 else "🟢"
                st.write(f"{risk_color} **{category.title()}**: {score:.2f}")

        # Hiring recommendation if available
        if 'hiring_recommendation' in evidence:
            recommendation = evidence['hiring_recommendation']
            if recommendation == 'reject':
                st.error(f"🚫 **AI Recommendation**: Do not proceed with hiring")
            elif recommendation == 'investigate':
                st.warning(f"⚠️ **AI Recommendation**: Investigate further before proceeding")
            else:
                st.success(f"✅ **AI Recommendation**: Proceed with standard process")

    # Recommendations
    recommendations = gemini_analysis.get('recommendations', [])
    if recommendations:
        st.subheader("💡 AI Recommendations")
        for rec in recommendations:
            st.info(f"• {rec}")

    # Show raw insights in expander
    with st.expander("View Raw AI Analysis"):
        st.json(gemini_analysis)


if __name__ == "__main__":
    main()
