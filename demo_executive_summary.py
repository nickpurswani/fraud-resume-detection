#!/usr/bin/env python3
"""
Executive Summary Demo Script for Fraudulent Candidate Detection Tool

This script demonstrates the improved executive summary cards with enhanced styling
and functionality. It shows how different risk scenarios are displayed in the UI.
"""

import streamlit as st
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import Config
from src.fraud_detector import FraudDetector, RiskLevel

# Page configuration
st.set_page_config(
    page_title="Executive Summary Demo",
    page_icon="📊",
    layout="wide"
)

# Custom CSS (same as main app)
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
    .demo-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def create_sample_analysis_results(risk_level, risk_score, total_flags, authenticity):
    """Create sample analysis results for demo"""
    return {
        'candidate_name': f'Demo Candidate ({risk_level.upper()})',
        'fraud_analysis': {
            'risk_assessment': {
                'overall_risk': risk_level,
                'risk_score': risk_score,
                'total_flags': total_flags,
                'critical_flags': 1 if risk_level == 'critical' else 0,
                'high_flags': 1 if risk_level == 'high' else 0,
                'medium_flags': 1 if risk_level == 'medium' else 0,
                'low_flags': 1 if risk_level == 'low' else 0
            },
            'confidence_scores': {
                'overall_authenticity': authenticity,
                'experience_authenticity': 0.8,
                'education_validity': 0.9,
                'skills_alignment': 0.7
            },
            'fraud_flags': [f'Demo Flag {i+1}' for i in range(total_flags)]
        },
        'timestamp': datetime.now().isoformat()
    }

def display_executive_summary_demo(results):
    """Display the executive summary with improved styling"""

    fraud_analysis = results.get('fraud_analysis', {})
    risk_assessment = fraud_analysis.get('risk_assessment', {})
    overall_risk = risk_assessment.get('overall_risk', 'unknown')
    risk_score = risk_assessment.get('risk_score', 0.0)
    total_flags = len(fraud_analysis.get('fraud_flags', []))

    col1, col2, col3, col4 = st.columns(4)

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

def display_hiring_recommendation(risk_level, risk_score, total_flags):
    """Display hiring recommendation"""
    st.subheader("🎯 Hiring Recommendation")

    if risk_level == 'critical' or risk_score > 0.8:
        st.error("❌ **DO NOT HIRE** - Critical fraud indicators detected")
    elif risk_level == 'high' or total_flags > 3:
        st.warning("⚠️ **INVESTIGATE FURTHER** - High risk candidate requiring verification")
    elif risk_level == 'medium':
        st.info("🔍 **PROCEED WITH CAUTION** - Some concerns identified")
    else:
        st.success("✅ **LOW RISK** - Candidate appears authentic")

def main():
    """Main demo application"""

    st.markdown('<h1 class="main-header">📊 Executive Summary Demo</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 1.1rem;">Demonstrating Enhanced Executive Summary Cards</p>', unsafe_allow_html=True)

    st.markdown("---")

    # Demo scenarios
    scenarios = {
        "Low Risk Candidate": {
            'risk_level': 'low',
            'risk_score': 0.2,
            'total_flags': 1,
            'authenticity': 0.95
        },
        "Medium Risk Candidate": {
            'risk_level': 'medium',
            'risk_score': 0.5,
            'total_flags': 2,
            'authenticity': 0.75
        },
        "High Risk Candidate": {
            'risk_level': 'high',
            'risk_score': 0.8,
            'total_flags': 4,
            'authenticity': 0.45
        },
        "Critical Risk Candidate": {
            'risk_level': 'critical',
            'risk_score': 0.95,
            'total_flags': 6,
            'authenticity': 0.25
        }
    }

    # Scenario selector
    st.subheader("🎭 Select Demo Scenario")
    selected_scenario = st.selectbox(
        "Choose a risk scenario to demonstrate:",
        list(scenarios.keys())
    )

    scenario_data = scenarios[selected_scenario]

    # Create sample results
    demo_results = create_sample_analysis_results(
        scenario_data['risk_level'],
        scenario_data['risk_score'],
        scenario_data['total_flags'],
        scenario_data['authenticity']
    )

    st.markdown("---")

    # Display executive summary
    st.header("📊 Executive Summary")
    display_executive_summary_demo(demo_results)

    # Display recommendation
    display_hiring_recommendation(
        scenario_data['risk_level'],
        scenario_data['risk_score'],
        scenario_data['total_flags']
    )

    # Show improvements
    st.markdown("---")
    st.subheader("🎨 Visual Improvements")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="demo-section">
            <h4>✨ Enhanced Styling Features</h4>
            <ul>
                <li>🎨 White background with subtle shadows</li>
                <li>🔲 Rounded corners and borders</li>
                <li>📊 Color-coded risk indicators</li>
                <li>📱 Responsive column layout</li>
                <li>🎯 Centered content alignment</li>
                <li>📏 Consistent card heights</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="demo-section">
            <h4>🔧 Functional Improvements</h4>
            <ul>
                <li>🚨 Dynamic risk level colors</li>
                <li>📈 Smart score color coding</li>
                <li>🚩 Flag count indicators</li>
                <li>✅ Authenticity confidence display</li>
                <li>🎯 Clear hiring recommendations</li>
                <li>📊 Improved visual hierarchy</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Technical details
    st.markdown("---")
    st.subheader("🔍 Technical Details")

    with st.expander("View CSS Implementation"):
        st.code("""
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

.risk-high { color: #dc3545; font-weight: bold; }
.risk-medium { color: #fd7e14; font-weight: bold; }
.risk-low { color: #28a745; font-weight: bold; }
.risk-critical { color: #6f42c1; font-weight: bold; }
        """, language="css")

    with st.expander("View Analysis Data Structure"):
        st.json({
            "fraud_analysis": {
                "risk_assessment": {
                    "overall_risk": scenario_data['risk_level'],
                    "risk_score": scenario_data['risk_score'],
                    "total_flags": scenario_data['total_flags']
                },
                "confidence_scores": {
                    "overall_authenticity": scenario_data['authenticity'],
                    "experience_authenticity": 0.8,
                    "education_validity": 0.9
                },
                "fraud_flags": ["Sample fraud flags..."]
            }
        })

    # Usage instructions
    st.markdown("---")
    st.subheader("📚 Usage Instructions")

    st.info("""
    **To use the executive summary in the main application:**

    1. Run `streamlit run app.py`
    2. Navigate to "Single Candidate Analysis"
    3. Upload a resume file or paste resume text
    4. Click "🚀 Analyze Candidate with Gemini AI"
    5. View the enhanced executive summary cards

    The cards will automatically display with appropriate colors based on risk levels
    and provide clear visual feedback about candidate authenticity.
    """)

if __name__ == "__main__":
    main()
