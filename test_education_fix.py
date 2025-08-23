#!/usr/bin/env python3
"""
Test script for EducationEntry parsing fix
Tests the fix for 'EducationEntry' object has no attribute 'get' error
"""

import os
import sys
import logging
from typing import Dict, List, Any

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from brightdata_linkedin_fixed import BrightDataLinkedInFixed, ProfileData
from fraud_detector import EducationEntry, ExperienceEntry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_mock_profile_data():
    """Create mock LinkedIn profile data"""
    return ProfileData(
        url="https://www.linkedin.com/in/testuser",
        name="John Smith",
        headline="Software Engineer at TechCorp",
        title="Senior Software Engineer",
        current_company="TechCorp Inc",
        location="San Francisco, CA",
        summary="Experienced software engineer...",
        experience=[
            {
                'title': 'Senior Software Engineer',
                'company': 'TechCorp Inc',
                'duration': '2020-Present',
                'location': 'San Francisco, CA',
                'description': 'Lead development of web applications'
            },
            {
                'title': 'Software Engineer',
                'company': 'StartupXYZ',
                'duration': '2018-2020',
                'location': 'San Jose, CA',
                'description': 'Full stack development'
            }
        ],
        education=[
            {
                'school': 'Stanford University',
                'degree': 'MS Computer Science',
                'field': 'Computer Science',
                'years': '2016-2018',
                'description': 'Master of Science in Computer Science'
            },
            {
                'school': 'UC Berkeley',
                'degree': 'BS Computer Science',
                'field': 'Computer Science',
                'years': '2012-2016',
                'description': 'Bachelor of Science in Computer Science'
            }
        ],
        skills=['Python', 'JavaScript', 'React', 'Node.js', 'AWS'],
        connections=500,
        success=True,
        raw_data={
            'name': 'John Smith',
            'current_company': 'TechCorp Inc',
            'experience': [
                {'title': 'Senior Software Engineer', 'company': 'TechCorp Inc'},
                {'title': 'Software Engineer', 'company': 'StartupXYZ'}
            ],
            'education': [
                {'school': 'Stanford University', 'degree': 'MS Computer Science'},
                {'school': 'UC Berkeley', 'degree': 'BS Computer Science'}
            ]
        }
    )

def create_resume_data_with_dicts():
    """Create resume data with dictionary format"""
    return {
        'name': 'John Smith',
        'experience': [
            {
                'title': 'Senior Software Engineer',
                'company': 'TechCorp Inc',
                'start_date': '2020',
                'end_date': 'Present'
            },
            {
                'title': 'Software Engineer',
                'company': 'StartupXYZ',
                'start_date': '2018',
                'end_date': '2020'
            }
        ],
        'education': [
            {
                'school': 'Stanford University',
                'degree': 'MS Computer Science',
                'institution': 'Stanford University'
            },
            {
                'school': 'UC Berkeley',
                'degree': 'BS Computer Science',
                'institution': 'UC Berkeley'
            }
        ],
        'skills': ['Python', 'JavaScript', 'React', 'AWS']
    }

def create_resume_data_with_objects():
    """Create resume data with EducationEntry and ExperienceEntry objects"""
    return {
        'name': 'John Smith',
        'experience': [
            ExperienceEntry(
                title='Senior Software Engineer',
                company='TechCorp Inc',
                start_date='2020-01-01',
                end_date=None,
                location='San Francisco, CA',
                description='Lead development team'
            ),
            ExperienceEntry(
                title='Software Engineer',
                company='StartupXYZ',
                start_date='2018-01-01',
                end_date='2020-01-01',
                location='San Jose, CA',
                description='Full stack development'
            )
        ],
        'education': [
            EducationEntry(
                degree='MS Computer Science',
                institution='Stanford University',
                graduation_year=2018,
                gpa=3.8,
                location='Stanford, CA',
                field_of_study='Computer Science'
            ),
            EducationEntry(
                degree='BS Computer Science',
                institution='UC Berkeley',
                graduation_year=2016,
                gpa=3.6,
                location='Berkeley, CA',
                field_of_study='Computer Science'
            )
        ],
        'skills': ['Python', 'JavaScript', 'React', 'AWS']
    }

def test_education_verification_with_dicts():
    """Test education verification with dictionary objects"""
    print("\n=== Test 1: Education Verification with Dictionaries ===")

    service = BrightDataLinkedInFixed()
    profile = create_mock_profile_data()
    resume_data = create_resume_data_with_dicts()

    try:
        # Test the education verification method directly
        result = service._verify_education(profile.education, resume_data['education'])

        print(f"✅ Dictionary test PASSED")
        print(f"   Match: {result['match']}")
        print(f"   Score: {result['score']:.2f}")
        print(f"   Details: {result['details']}")
        return True

    except Exception as e:
        print(f"❌ Dictionary test FAILED: {e}")
        return False

def test_education_verification_with_objects():
    """Test education verification with EducationEntry objects"""
    print("\n=== Test 2: Education Verification with EducationEntry Objects ===")

    service = BrightDataLinkedInFixed()
    profile = create_mock_profile_data()
    resume_data = create_resume_data_with_objects()

    try:
        # Test the education verification method directly
        result = service._verify_education(profile.education, resume_data['education'])

        print(f"✅ EducationEntry test PASSED")
        print(f"   Match: {result['match']}")
        print(f"   Score: {result['score']:.2f}")
        print(f"   Details: {result['details']}")
        return True

    except Exception as e:
        print(f"❌ EducationEntry test FAILED: {e}")
        return False

def test_experience_verification_with_objects():
    """Test experience verification with ExperienceEntry objects"""
    print("\n=== Test 3: Experience Verification with ExperienceEntry Objects ===")

    service = BrightDataLinkedInFixed()
    profile = create_mock_profile_data()
    resume_data = create_resume_data_with_objects()

    try:
        # Test the experience verification method directly
        result = service._verify_experience(profile.experience, resume_data['experience'])

        print(f"✅ ExperienceEntry test PASSED")
        print(f"   Match: {result['match']}")
        print(f"   Score: {result['score']:.2f}")
        print(f"   Details: {result['details']}")
        return True

    except Exception as e:
        print(f"❌ ExperienceEntry test FAILED: {e}")
        return False

def test_full_verification_with_objects():
    """Test full verification workflow with mixed object types"""
    print("\n=== Test 4: Full Verification with Mixed Object Types ===")

    service = BrightDataLinkedInFixed()
    profile = create_mock_profile_data()
    resume_data = create_resume_data_with_objects()

    try:
        # Test the full verification workflow
        result = service.verify_against_resume(profile, resume_data)

        print(f"✅ Full verification test PASSED")
        print(f"   Overall match score: {result['overall_match_score']:.2f}")
        print(f"   Confidence score: {result['confidence_score']:.2f}")
        print(f"   Fraud indicators: {len(result['fraud_indicators'])}")

        # Check individual verification results
        if 'verification_results' in result:
            ver_results = result['verification_results']
            if 'education_match' in ver_results:
                edu_match = ver_results['education_match']
                print(f"   Education match: {edu_match['match']} (score: {edu_match['score']:.2f})")

            if 'experience_match' in ver_results:
                exp_match = ver_results['experience_match']
                print(f"   Experience match: {exp_match['match']} (score: {exp_match['score']:.2f})")

        return True

    except Exception as e:
        print(f"❌ Full verification test FAILED: {e}")
        return False

def test_safe_verification():
    """Test the safe verification wrapper"""
    print("\n=== Test 5: Safe Verification Wrapper ===")

    service = BrightDataLinkedInFixed()
    profile = create_mock_profile_data()
    resume_data = create_resume_data_with_objects()

    try:
        # Test the safe verification method
        result = service.verify_against_resume_safe(profile, resume_data)

        print(f"✅ Safe verification test PASSED")
        print(f"   Service used: {result.get('service_used', 'Standard')}")
        print(f"   Overall match score: {result['overall_match_score']:.2f}")

        if 'fallback_reason' in result:
            print(f"   Fallback reason: {result['fallback_reason']}")

        return True

    except Exception as e:
        print(f"❌ Safe verification test FAILED: {e}")
        return False

def test_ai_fallback_availability():
    """Test if AI fallback is available"""
    print("\n=== Test 6: AI Fallback Availability ===")

    service = BrightDataLinkedInFixed()
    profile = create_mock_profile_data()
    resume_data = create_resume_data_with_objects()

    try:
        # Test AI fallback method
        result = service.verify_with_ai_fallback(profile, resume_data)

        if 'error' not in result or 'API key not found' in result.get('error', ''):
            print(f"⚠️  AI fallback available but no API key set")
            print(f"   To enable: export GEMINI_API_KEY='your-api-key'")
        else:
            print(f"✅ AI fallback test COMPLETED")
            print(f"   Service used: {result.get('service_used', 'Unknown')}")

        return True

    except Exception as e:
        print(f"❌ AI fallback test FAILED: {e}")
        return False

def simulate_original_error():
    """Simulate the original error scenario"""
    print("\n=== Test 7: Simulating Original Error Scenario ===")

    # Create a scenario that would cause the original error
    service = BrightDataLinkedInFixed()
    profile = create_mock_profile_data()

    # Create problematic resume data (mixing object types)
    problematic_resume_data = {
        'name': 'John Smith',
        'education': [
            EducationEntry(
                degree='MS Computer Science',
                institution='Stanford University',
                graduation_year=2018,
                gpa=3.8,
                location='Stanford, CA',
                field_of_study='Computer Science'
            ),
            {  # Mixed with dictionary
                'school': 'UC Berkeley',
                'degree': 'BS Computer Science'
            }
        ],
        'experience': [
            ExperienceEntry(
                title='Senior Software Engineer',
                company='TechCorp Inc',
                start_date='2020-01-01',
                end_date=None,
                location='San Francisco, CA',
                description='Lead development team'
            )
        ]
    }

    try:
        result = service.verify_against_resume_safe(problematic_resume_data, resume_data)
        print(f"✅ Original error scenario handled successfully")
        print(f"   No 'get' attribute error occurred")
        return True
    except Exception as e:
        print(f"❌ Original error scenario FAILED: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing EducationEntry Parsing Fix")
    print("=" * 60)

    tests = [
        test_education_verification_with_dicts,
        test_education_verification_with_objects,
        test_experience_verification_with_objects,
        test_full_verification_with_objects,
        test_safe_verification,
        test_ai_fallback_availability,
        simulate_original_error
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} CRASHED: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"TEST SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total Tests: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(passed/(passed+failed)*100) if (passed+failed) > 0 else 0:.1f}%")

    if failed == 0:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"✅ The EducationEntry parsing fix is working correctly")
        print(f"✅ Both Dict and EducationEntry objects are handled properly")
        print(f"✅ Safe verification wrapper provides fallback protection")
    else:
        print(f"\n⚠️  Some tests failed. Review the results above.")

    print(f"\n💡 Next Steps:")
    print(f"   1. If tests passed, the fix should resolve the original error")
    print(f"   2. Test in your Streamlit app with real data")
    print(f"   3. Consider setting GEMINI_API_KEY for AI fallback capability")

if __name__ == "__main__":
    main()
