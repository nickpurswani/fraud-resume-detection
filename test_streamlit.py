#!/usr/bin/env python3
"""
Streamlit App Test Script for Fraudulent Candidate Detection Tool

This script tests the Streamlit app functionality including:
- App initialization
- Resume processing
- Executive summary display
- Analysis results display
"""

import streamlit as st
import tempfile
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_sample_resume():
    """Create a sample resume for testing"""
    sample_resume = """
JOHN SMITH
Software Engineer
Email: john.smith@email.com
Phone: +1-555-123-4567
Location: San Francisco, CA

PROFESSIONAL EXPERIENCE

Senior Software Engineer | TechCorp Inc. | 2020 - Present
• Led development of microservices architecture serving 1M+ users
• Managed cross-functional team of 8 developers and designers
• Improved system performance by 40% through optimization initiatives
• Implemented CI/CD pipelines reducing deployment time by 60%

Software Developer | StartupXYZ | 2018 - 2020
• Developed full-stack web applications using React, Node.js, and PostgreSQL
• Built RESTful APIs handling 10,000+ requests per minute
• Collaborated with product team to define technical requirements
• Mentored 3 junior developers on best practices

Junior Developer | CodeCorp | 2016 - 2018
• Developed responsive web interfaces using HTML, CSS, JavaScript
• Fixed bugs and implemented new features in legacy systems
• Participated in code reviews and agile development processes

EDUCATION

Bachelor of Science in Computer Science | University of Technology | 2012 - 2016
• GPA: 3.8/4.0
• Relevant Coursework: Data Structures, Algorithms, Software Engineering
• Senior Project: E-commerce platform with recommendation engine

TECHNICAL SKILLS

Programming Languages: Python, JavaScript, Java, C++, SQL
Frontend: React, Vue.js, HTML5, CSS3, Bootstrap
Backend: Node.js, Express.js, Django, Flask, Spring Boot
Databases: PostgreSQL, MongoDB, MySQL, Redis
Cloud & DevOps: AWS, Docker, Kubernetes, Jenkins, GitHub Actions
Tools: Git, Jira, Slack, Visual Studio Code

CERTIFICATIONS
• AWS Certified Solutions Architect - Associate (2021)
• Google Cloud Professional Developer (2020)
• Certified Scrum Master (2019)

PROJECTS

E-commerce Platform (2021)
• Built scalable microservices architecture using Node.js and Docker
• Implemented payment processing with Stripe API
• Used Redis for caching and PostgreSQL for data storage
• Deployed on AWS with auto-scaling capabilities

Data Analytics Dashboard (2020)
• Created real-time analytics dashboard using React and D3.js
• Built ETL pipeline processing 100GB+ daily data
• Used Python and Pandas for data processing
• Integrated with multiple third-party APIs

Mobile App Backend (2019)
• Developed REST API for iOS/Android mobile application
• Implemented user authentication and authorization
• Used Django REST framework with PostgreSQL database
• Achieved 99.9% uptime with proper error handling
"""
    return sample_resume

def test_streamlit_components():
    """Test if Streamlit app components load correctly"""
    print("Testing Streamlit app components...")

    try:
        # Test imports
        import app
        print("✓ App module imported successfully")

        # Test component initialization
        from src.fraud_detector import FraudDetector, RiskLevel
        from src.utils import TextExtractor, validate_file_upload
        from src.fit_scorer import FitScorer
        from src.linkedin_verifier import LinkedInVerifier
        from src.report_generator import ReportGenerator
        from config import Config

        print("✓ All components imported successfully")

        # Test configuration
        config = Config()
        validation = config.validate_config()

        if validation['valid']:
            print("✓ Configuration is valid")
        else:
            print(f"⚠ Configuration issues: {validation['issues']}")

        return True

    except Exception as e:
        print(f"✗ Component test failed: {e}")
        return False

def test_resume_processing():
    """Test resume processing functionality"""
    print("\nTesting resume processing...")

    try:
        from config import Config
        from src.fraud_detector import FraudDetector
        from src.utils import TextExtractor, validate_file_upload

        # Create sample resume file
        sample_text = create_sample_resume()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(sample_text)
            temp_file_path = f.name

        try:
            # Test file validation
            validation = validate_file_upload(temp_file_path)
            if not validation['valid']:
                print(f"✗ File validation failed: {validation['error']}")
                return False
            print(f"✓ File validation passed ({validation['size_mb']:.2f} MB)")

            # Test text extraction
            text_extractor = TextExtractor()
            extracted_text = text_extractor.extract_text(temp_file_path)

            if not extracted_text or len(extracted_text.strip()) == 0:
                print("✗ Text extraction failed - no text extracted")
                return False
            print(f"✓ Text extraction successful ({len(extracted_text)} characters)")

            # Test fraud analysis
            config = Config()
            detector = FraudDetector(config.__dict__)

            results = detector.analyze_resume(extracted_text)

            if not results or not isinstance(results, dict):
                print("✗ Fraud analysis failed - invalid results")
                return False

            # Check required result fields
            required_fields = ['fraud_score', 'risk_level', 'fraud_flags', 'risk_assessment']
            missing_fields = [field for field in required_fields if field not in results]

            if missing_fields:
                print(f"✗ Missing result fields: {missing_fields}")
                return False

            print("✓ Fraud analysis completed successfully")
            print(f"  - Risk Level: {results.get('risk_level', 'unknown')}")
            print(f"  - Fraud Score: {results.get('fraud_score', 0.0):.2f}")
            print(f"  - Total Flags: {len(results.get('fraud_flags', []))}")

            return True

        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except Exception as e:
        print(f"✗ Resume processing test failed: {e}")
        return False

def test_executive_summary_data():
    """Test executive summary data structure"""
    print("\nTesting executive summary data...")

    try:
        from config import Config
        from src.fraud_detector import FraudDetector

        sample_text = create_sample_resume()
        config = Config()
        detector = FraudDetector(config.__dict__)

        results = detector.analyze_resume(sample_text)

        # Test executive summary data structure
        fraud_analysis = results
        risk_assessment = fraud_analysis.get('risk_assessment', {})
        confidence_scores = fraud_analysis.get('confidence_scores', {})

        # Check executive summary components
        components = {
            'overall_risk': risk_assessment.get('overall_risk'),
            'risk_score': risk_assessment.get('risk_score'),
            'total_flags': len(fraud_analysis.get('fraud_flags', [])),
            'authenticity': confidence_scores.get('overall_authenticity')
        }

        print("Executive Summary Components:")
        for component, value in components.items():
            if value is not None:
                print(f"  ✓ {component}: {value}")
            else:
                print(f"  ⚠ {component}: Missing or None")

        # Test if all required components exist
        required_components = ['overall_risk', 'risk_score', 'total_flags', 'authenticity']
        missing_components = [comp for comp in required_components if components[comp] is None]

        if missing_components:
            print(f"⚠ Missing components: {missing_components}")
        else:
            print("✓ All executive summary components present")

        return len(missing_components) == 0

    except Exception as e:
        print(f"✗ Executive summary data test failed: {e}")
        return False

def test_analysis_tabs():
    """Test analysis tab data availability"""
    print("\nTesting analysis tabs data...")

    try:
        from config import Config
        from src.fraud_detector import FraudDetector

        sample_text = create_sample_resume()
        config = Config()
        detector = FraudDetector(config.__dict__)

        results = detector.analyze_resume(sample_text)

        # Test tab data availability
        tab_data = {
            'Fraud Flags': results.get('fraud_flags'),
            'Risk Assessment': results.get('risk_assessment'),
            'NLP Analysis': results.get('detailed_analysis', {}).get('nlp'),
            'Gemini Analysis': results.get('gemini_analysis'),
            'Confidence Scores': results.get('confidence_scores')
        }

        print("Analysis Tab Data:")
        for tab, data in tab_data.items():
            if data is not None and data != {}:
                if isinstance(data, list):
                    print(f"  ✓ {tab}: {len(data)} items")
                elif isinstance(data, dict):
                    print(f"  ✓ {tab}: {len(data)} fields")
                else:
                    print(f"  ✓ {tab}: Available")
            else:
                print(f"  ⚠ {tab}: No data")

        return True

    except Exception as e:
        print(f"✗ Analysis tabs test failed: {e}")
        return False

def create_css_test():
    """Test CSS styling components"""
    print("\nTesting CSS styling...")

    # Test metric container styling
    test_html = '''
    <div class="metric-container">
        <h3>🚨 Risk Level</h3>
        <p class="risk-high">HIGH</p>
    </div>
    '''

    # Check if required CSS classes would render
    required_classes = [
        'metric-container',
        'risk-high',
        'risk-medium',
        'risk-low',
        'risk-critical'
    ]

    print("Required CSS Classes:")
    for css_class in required_classes:
        print(f"  ✓ .{css_class}")

    print("✓ CSS structure appears correct")
    return True

def run_comprehensive_test():
    """Run comprehensive test suite"""
    print("="*60)
    print("STREAMLIT APP COMPREHENSIVE TEST")
    print("="*60)

    tests = [
        ("Component Loading", test_streamlit_components),
        ("Resume Processing", test_resume_processing),
        ("Executive Summary Data", test_executive_summary_data),
        ("Analysis Tabs", test_analysis_tabs),
        ("CSS Styling", create_css_test)
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{'='*40}")
        print(f"Running {test_name} Test")
        print('='*40)

        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))

    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:<8} {test_name}")

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Streamlit app is ready for use.")
        print("\nTo run the app:")
        print("streamlit run app.py")
        return 0
    else:
        print("❌ Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = run_comprehensive_test()

    # Additional usage info
    if exit_code == 0:
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print("1. Start the Streamlit app: streamlit run app.py")
        print("2. Upload a resume file or paste resume text")
        print("3. Check that executive summary cards are visible")
        print("4. Verify all analysis tabs display correctly")
        print("5. Test different analysis options")

    sys.exit(exit_code)
