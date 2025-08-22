#!/usr/bin/env python3
"""
Basic functionality test for Fraudulent Candidate Detection Tool

This script tests core functionality to identify any configuration or import issues.
"""

import sys
import os
import tempfile
from pathlib import Path

def test_imports():
    """Test all necessary imports"""
    print("Testing imports...")

    try:
        import streamlit as st
        print("✓ Streamlit imported successfully")
    except ImportError as e:
        print(f"✗ Streamlit import failed: {e}")
        return False

    try:
        from src.fraud_detector import FraudDetector, RiskLevel
        print("✓ FraudDetector imported successfully")
    except ImportError as e:
        print(f"✗ FraudDetector import failed: {e}")
        return False

    try:
        from src.utils import TextExtractor, validate_file_upload
        print("✓ TextExtractor imported successfully")
    except ImportError as e:
        print(f"✗ TextExtractor import failed: {e}")
        return False

    try:
        from src.fit_scorer import FitScorer
        print("✓ FitScorer imported successfully")
    except ImportError as e:
        print(f"✗ FitScorer import failed: {e}")
        return False

    try:
        from src.linkedin_verifier import LinkedInVerifier
        print("✓ LinkedInVerifier imported successfully")
    except ImportError as e:
        print(f"✗ LinkedInVerifier import failed: {e}")
        return False

    try:
        from src.report_generator import ReportGenerator
        print("✓ ReportGenerator imported successfully")
    except ImportError as e:
        print(f"✗ ReportGenerator import failed: {e}")
        return False

    try:
        from config import Config
        print("✓ Config imported successfully")
    except ImportError as e:
        print(f"✗ Config import failed: {e}")
        return False

    return True

def test_config():
    """Test configuration"""
    print("\nTesting configuration...")

    try:
        from config import Config
        config = Config()
        validation = config.validate_config()

        print(f"✓ Configuration validation: {'Passed' if validation['valid'] else 'Failed'}")

        if validation['warnings']:
            print("⚠ Warnings:")
            for warning in validation['warnings']:
                print(f"  - {warning}")

        if validation['issues']:
            print("✗ Issues:")
            for issue in validation['issues']:
                print(f"  - {issue}")

        return validation['valid']
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_text_extraction():
    """Test text extraction functionality"""
    print("\nTesting text extraction...")

    try:
        from src.utils import TextExtractor

        # Create a sample text file
        sample_text = """John Doe
Software Engineer
Experience: 5 years in Python development
Skills: Python, JavaScript, React, Node.js
Education: BS Computer Science, University of Technology

Professional Experience:
- Senior Software Engineer at Tech Corp (2019-2024)
- Software Developer at StartupXYZ (2017-2019)
- Junior Developer at CodeCorp (2015-2017)

Projects:
- E-commerce platform using React and Node.js
- Data analytics dashboard with Python and Flask
- Mobile app backend with Django REST framework"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(sample_text)
            temp_file_path = f.name

        try:
            extracted_text = TextExtractor.extract_text(temp_file_path)

            if extracted_text and len(extracted_text.strip()) > 0:
                print("✓ Text extraction successful")
                print(f"  Extracted {len(extracted_text)} characters")
                return True
            else:
                print("✗ Text extraction returned empty result")
                return False
        finally:
            os.unlink(temp_file_path)

    except Exception as e:
        print(f"✗ Text extraction test failed: {e}")
        return False

def test_file_validation():
    """Test file validation"""
    print("\nTesting file validation...")

    try:
        from src.utils import validate_file_upload

        # Create a sample file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Sample resume content for testing file validation")
            temp_file_path = f.name

        try:
            validation = validate_file_upload(temp_file_path)

            if validation['valid']:
                print("✓ File validation successful")
                print(f"  File size: {validation['size_mb']:.2f} MB")
                print(f"  Extension: {validation['extension']}")
                return True
            else:
                print(f"✗ File validation failed: {validation['error']}")
                return False
        finally:
            os.unlink(temp_file_path)

    except Exception as e:
        print(f"✗ File validation test failed: {e}")
        return False

def test_component_initialization():
    """Test component initialization"""
    print("\nTesting component initialization...")

    try:
        from config import Config
        from src.fraud_detector import FraudDetector
        from src.fit_scorer import FitScorer
        from src.report_generator import ReportGenerator
        from src.utils import TextExtractor
        from src.linkedin_verifier import LinkedInVerifier

        config = Config()
        config_dict = config.__dict__

        # Test FraudDetector
        try:
            detector = FraudDetector(config_dict)
            print("✓ FraudDetector initialized successfully")
        except Exception as e:
            print(f"✗ FraudDetector initialization failed: {e}")
            return False

        # Test FitScorer
        try:
            fit_scorer = FitScorer(config_dict)
            print("✓ FitScorer initialized successfully")
        except Exception as e:
            print(f"✗ FitScorer initialization failed: {e}")
            return False

        # Test ReportGenerator
        try:
            report_generator = ReportGenerator(config_dict)
            print("✓ ReportGenerator initialized successfully")
        except Exception as e:
            print(f"✗ ReportGenerator initialization failed: {e}")
            return False

        # Test TextExtractor
        try:
            text_extractor = TextExtractor()
            print("✓ TextExtractor initialized successfully")
        except Exception as e:
            print(f"✗ TextExtractor initialization failed: {e}")
            return False

        # Test LinkedInVerifier
        try:
            linkedin_verifier = LinkedInVerifier(config.LINKEDIN_API_KEY, config_dict)
            print("✓ LinkedInVerifier initialized successfully")
        except Exception as e:
            print(f"⚠ LinkedInVerifier initialization warning: {e}")
            # LinkedIn verifier might fail due to missing API key, which is expected

        return True

    except Exception as e:
        print(f"✗ Component initialization test failed: {e}")
        return False

def test_basic_analysis():
    """Test basic fraud analysis"""
    print("\nTesting basic analysis...")

    try:
        from config import Config
        from src.fraud_detector import FraudDetector

        config = Config()
        detector = FraudDetector(config.__dict__)

        sample_resume = """John Doe
Senior Software Engineer
Email: john.doe@email.com
Phone: +1-555-123-4567

Professional Experience:
Senior Software Engineer - Tech Corp (2020-2024)
- Led development of microservices architecture
- Managed team of 5 developers
- Improved system performance by 40%

Software Developer - StartupXYZ (2018-2020)
- Developed web applications using React and Node.js
- Implemented CI/CD pipelines
- Built RESTful APIs

Education:
Bachelor of Science in Computer Science
University of Technology (2014-2018)

Skills:
Python, JavaScript, React, Node.js, Docker, AWS, PostgreSQL"""

        # Run basic analysis
        results = detector.analyze_resume(sample_resume)

        if results and isinstance(results, dict):
            print("✓ Basic analysis completed successfully")

            # Check for key components in results
            expected_keys = ['fraud_score', 'risk_level', 'fraud_flags']
            for key in expected_keys:
                if key in results:
                    print(f"  ✓ Found {key} in results")
                else:
                    print(f"  ⚠ Missing {key} in results")

            print(f"  Risk Level: {results.get('risk_level', 'Unknown')}")
            print(f"  Fraud Score: {results.get('fraud_score', 'Unknown')}")

            return True
        else:
            print("✗ Basic analysis returned invalid results")
            return False

    except Exception as e:
        print(f"✗ Basic analysis test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Fraudulent Candidate Detection Tool - Basic Test ===\n")

    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_config),
        ("Text Extraction Test", test_text_extraction),
        ("File Validation Test", test_file_validation),
        ("Component Initialization Test", test_component_initialization),
        ("Basic Analysis Test", test_basic_analysis),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running {test_name}")
        print('='*50)

        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print('='*50)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:<8} {test_name}")

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
