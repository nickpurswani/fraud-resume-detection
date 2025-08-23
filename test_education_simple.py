#!/usr/bin/env python3
"""
Simple test for EducationEntry parsing fix without problematic imports
Tests the fix for 'EducationEntry' object has no attribute 'get' error
"""

import os
import sys
import logging

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from brightdata_linkedin_fixed import BrightDataLinkedInFixed, ProfileData

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockEducationEntry:
    """Mock EducationEntry object that mimics the real one"""
    def __init__(self, degree, institution, graduation_year=None, gpa=None, location=None, field_of_study=None):
        self.degree = degree
        self.institution = institution
        self.graduation_year = graduation_year
        self.gpa = gpa
        self.location = location
        self.field_of_study = field_of_study

    def __str__(self):
        return f"MockEducationEntry(degree='{self.degree}', institution='{self.institution}')"

class MockExperienceEntry:
    """Mock ExperienceEntry object that mimics the real one"""
    def __init__(self, title, company, start_date=None, end_date=None, location=None, description=None):
        self.title = title
        self.company = company
        self.start_date = start_date
        self.end_date = end_date
        self.location = location
        self.description = description
        self.position = title  # Alternative attribute name

    def __str__(self):
        return f"MockExperienceEntry(title='{self.title}', company='{self.company}')"

def create_mock_profile_data():
    """Create mock LinkedIn profile data"""
    return ProfileData(
        url="https://www.linkedin.com/in/testuser",
        name="John Smith",
        headline="Software Engineer at TechCorp",
        title="Senior Software Engineer",
        current_company="TechCorp Inc",
        location="San Francisco, CA",
        experience=[
            {
                'title': 'Senior Software Engineer',
                'company': 'TechCorp Inc',
                'duration': '2020-Present'
            },
            {
                'title': 'Software Engineer',
                'company': 'StartupXYZ',
                'duration': '2018-2020'
            }
        ],
        education=[
            {
                'school': 'Stanford University',
                'degree': 'MS Computer Science',
                'field': 'Computer Science'
            },
            {
                'school': 'UC Berkeley',
                'degree': 'BS Computer Science',
                'field': 'Computer Science'
            }
        ],
        skills=['Python', 'JavaScript', 'React'],
        success=True,
        raw_data={
            'name': 'John Smith',
            'current_company': 'TechCorp Inc'
        }
    )

def test_education_with_dictionaries():
    """Test education verification with dictionary objects (should work)"""
    print("\n=== Test 1: Education with Dictionaries ===")

    service = BrightDataLinkedInFixed()
    profile = create_mock_profile_data()

    resume_education_dicts = [
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
    ]

    try:
        result = service._verify_education(profile.education, resume_education_dicts)
        print(f"✅ Dictionary test PASSED")
        print(f"   Score: {result['score']:.2f}")
        print(f"   Details: {result['details']}")
        return True
    except Exception as e:
        print(f"❌ Dictionary test FAILED: {e}")
        return False

def test_education_with_mock_objects():
    """Test education verification with MockEducationEntry objects (the fix)"""
    print("\n=== Test 2: Education with Mock Objects ===")

    service = BrightDataLinkedInFixed()
    profile = create_mock_profile_data()

    resume_education_objects = [
        MockEducationEntry(
            degree='MS Computer Science',
            institution='Stanford University',
            graduation_year=2018
        ),
        MockEducationEntry(
            degree='BS Computer Science',
            institution='UC Berkeley',
            graduation_year=2016
        )
    ]

    try:
        result = service._verify_education(profile.education, resume_education_objects)
        print(f"✅ Mock object test PASSED")
        print(f"   Score: {result['score']:.2f}")
        print(f"   Details: {result['details']}")
        return True
    except Exception as e:
        print(f"❌ Mock object test FAILED: {e}")
        print(f"   This indicates the fix didn't work: {e}")
        return False

def test_experience_with_mock_objects():
    """Test experience verification with MockExperienceEntry objects"""
    print("\n=== Test 3: Experience with Mock Objects ===")

    service = BrightDataLinkedInFixed()
    profile = create_mock_profile_data()

    resume_experience_objects = [
        MockExperienceEntry(
            title='Senior Software Engineer',
            company='TechCorp Inc',
            start_date='2020-01-01'
        ),
        MockExperienceEntry(
            title='Software Engineer',
            company='StartupXYZ',
            start_date='2018-01-01',
            end_date='2020-01-01'
        )
    ]

    try:
        result = service._verify_experience(profile.experience, resume_experience_objects)
        print(f"✅ Experience mock object test PASSED")
        print(f"   Score: {result['score']:.2f}")
        print(f"   Details: {result['details']}")
        return True
    except Exception as e:
        print(f"❌ Experience mock object test FAILED: {e}")
        return False

def test_mixed_object_types():
    """Test handling mixed dictionary and object types"""
    print("\n=== Test 4: Mixed Object Types ===")

    service = BrightDataLinkedInFixed()
    profile = create_mock_profile_data()

    # Mix dictionaries and mock objects
    mixed_education = [
        {
            'school': 'Stanford University',
            'degree': 'MS Computer Science'
        },
        MockEducationEntry(
            degree='BS Computer Science',
            institution='UC Berkeley'
        )
    ]

    try:
        result = service._verify_education(profile.education, mixed_education)
        print(f"✅ Mixed types test PASSED")
        print(f"   Score: {result['score']:.2f}")
        print(f"   Details: {result['details']}")
        return True
    except Exception as e:
        print(f"❌ Mixed types test FAILED: {e}")
        return False

def test_full_verification_safe():
    """Test the safe verification wrapper"""
    print("\n=== Test 5: Safe Verification Wrapper ===")

    service = BrightDataLinkedInFixed()
    profile = create_mock_profile_data()

    resume_data = {
        'name': 'John Smith',
        'experience': [
            MockExperienceEntry('Senior Software Engineer', 'TechCorp Inc'),
            MockExperienceEntry('Software Engineer', 'StartupXYZ')
        ],
        'education': [
            MockEducationEntry('MS Computer Science', 'Stanford University'),
            MockEducationEntry('BS Computer Science', 'UC Berkeley')
        ],
        'skills': ['Python', 'JavaScript']
    }

    try:
        result = service.verify_against_resume_safe(profile, resume_data)
        print(f"✅ Safe verification test PASSED")
        print(f"   Overall score: {result['overall_match_score']:.2f}")
        print(f"   Service used: {result.get('service_used', 'Standard')}")

        if 'fallback_reason' in result:
            print(f"   Fallback used: {result['fallback_reason']}")

        return True
    except Exception as e:
        print(f"❌ Safe verification test FAILED: {e}")
        return False

def demonstrate_original_error():
    """Demonstrate what the original error looked like"""
    print("\n=== Demonstration: Original Error Scenario ===")

    # This would be the problematic code before the fix
    mock_obj = MockEducationEntry('MS Computer Science', 'Stanford University')

    print(f"Mock object: {mock_obj}")
    print(f"Has 'get' method: {hasattr(mock_obj, 'get')}")
    print(f"Has 'institution' attribute: {hasattr(mock_obj, 'institution')}")
    print(f"Institution value: {getattr(mock_obj, 'institution', 'Not found')}")

    # Show how the old code would fail
    try:
        # This is what would fail in the old code:
        value = mock_obj.get('institution')  # This would cause AttributeError
        print(f"❌ This should have failed but didn't: {value}")
    except AttributeError as e:
        print(f"✅ Expected error reproduced: {e}")
        print(f"   This is exactly the error we fixed!")

    return True

def main():
    """Run all tests"""
    print("🔧 Testing EducationEntry Parsing Fix")
    print("=" * 50)
    print("This tests the fix for: 'EducationEntry' object has no attribute 'get'")

    tests = [
        test_education_with_dictionaries,
        test_education_with_mock_objects,
        test_experience_with_mock_objects,
        test_mixed_object_types,
        test_full_verification_safe,
        demonstrate_original_error
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

    print(f"\n{'=' * 50}")
    print(f"TEST SUMMARY")
    print(f"{'=' * 50}")
    print(f"Total Tests: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"✅ The fix for 'EducationEntry' object has no attribute 'get' is working")
        print(f"✅ Both dictionary and object types are now handled properly")
        print(f"✅ The error should no longer occur in your app")

        print(f"\n🚀 Next Steps:")
        print(f"   1. Set your BRIGHTDATA_API_KEY environment variable")
        print(f"   2. Test the updated code in your Streamlit app")
        print(f"   3. The LinkedIn verification should now work without errors")

    else:
        print(f"\n⚠️  {failed} test(s) failed. The fix may need more work.")

    print(f"\n💡 About the fix:")
    print(f"   • Modified _verify_education() to handle both Dict and object types")
    print(f"   • Modified _verify_experience() to handle both Dict and object types")
    print(f"   • Added verify_against_resume_safe() with AI fallback")
    print(f"   • Uses hasattr() and getattr() instead of .get() for objects")

if __name__ == "__main__":
    main()
