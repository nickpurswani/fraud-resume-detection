#!/usr/bin/env python3
"""
Test parsing methods directly without API key requirement
Tests the fix for 'EducationEntry' object has no attribute 'get' error
"""

import os
import sys

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Mock the environment variable to bypass API key check
os.environ['BRIGHTDATA_API_KEY'] = 'test-key-for-parsing-only'

from brightdata_linkedin_fixed import BrightDataLinkedInFixed, ProfileData

class MockEducationEntry:
    """Mock EducationEntry object that mimics the real one"""
    def __init__(self, degree, institution, graduation_year=None, gpa=None, location=None, field_of_study=None):
        self.degree = degree
        self.institution = institution
        self.graduation_year = graduation_year
        self.gpa = gpa
        self.location = location
        self.field_of_study = field_of_study

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

def test_education_parsing_fix():
    """Test the education parsing fix specifically"""
    print("🔧 Testing Education Parsing Fix")
    print("=" * 50)

    # Create service instance (now possible with mock API key)
    service = BrightDataLinkedInFixed()

    # Mock LinkedIn education data (as dictionaries)
    linkedin_education = [
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
    ]

    print("\n=== Test 1: Dictionary Resume Education (Original Working Case) ===")
    resume_edu_dicts = [
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
        result = service._verify_education(linkedin_education, resume_edu_dicts)
        print(f"✅ Dictionary test PASSED")
        print(f"   Match: {result['match']}")
        print(f"   Score: {result['score']:.2f}")
        print(f"   Details: {result['details']}")
    except Exception as e:
        print(f"❌ Dictionary test FAILED: {e}")
        return False

    print("\n=== Test 2: MockEducationEntry Resume Education (The Fix) ===")
    resume_edu_objects = [
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
        result = service._verify_education(linkedin_education, resume_edu_objects)
        print(f"✅ MockEducationEntry test PASSED")
        print(f"   Match: {result['match']}")
        print(f"   Score: {result['score']:.2f}")
        print(f"   Details: {result['details']}")
        print(f"   🎉 The 'get' attribute error has been FIXED!")
    except Exception as e:
        print(f"❌ MockEducationEntry test FAILED: {e}")
        print(f"   The fix did not work properly")
        return False

    print("\n=== Test 3: Mixed Types (Real World Scenario) ===")
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
        result = service._verify_education(linkedin_education, mixed_education)
        print(f"✅ Mixed types test PASSED")
        print(f"   Score: {result['score']:.2f}")
        print(f"   Details: {result['details']}")
    except Exception as e:
        print(f"❌ Mixed types test FAILED: {e}")
        return False

    return True

def test_experience_parsing_fix():
    """Test the experience parsing fix"""
    print("\n=== Test 4: Experience Parsing Fix ===")

    service = BrightDataLinkedInFixed()

    # Mock LinkedIn experience data
    linkedin_experience = [
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
    ]

    # Test with MockExperienceEntry objects
    resume_exp_objects = [
        MockExperienceEntry(
            title='Senior Software Engineer',
            company='TechCorp Inc'
        ),
        MockExperienceEntry(
            title='Software Engineer',
            company='StartupXYZ'
        )
    ]

    try:
        result = service._verify_experience(linkedin_experience, resume_exp_objects)
        print(f"✅ Experience object test PASSED")
        print(f"   Score: {result['score']:.2f}")
        print(f"   Details: {result['details']}")
        return True
    except Exception as e:
        print(f"❌ Experience object test FAILED: {e}")
        return False

def test_demonstrate_error_scenario():
    """Demonstrate the exact error scenario that was happening"""
    print("\n=== Test 5: Demonstrating Original Error ===")

    # This is what was happening before the fix
    mock_edu = MockEducationEntry('MS Computer Science', 'Stanford University')

    print(f"MockEducationEntry object: {mock_edu}")
    print(f"Has 'degree' attribute: {hasattr(mock_edu, 'degree')}")
    print(f"Has 'institution' attribute: {hasattr(mock_edu, 'institution')}")
    print(f"Has 'get' method: {hasattr(mock_edu, 'get')}")

    # Show the error that would occur with old code
    print(f"\nWhat the OLD code would do:")
    try:
        # This is the problematic line that caused the error:
        school = mock_edu.get('school')  # AttributeError!
        print(f"❌ ERROR: This should have failed!")
    except AttributeError as e:
        print(f"✅ Expected error reproduced: {e}")

    print(f"\nWhat the NEW code does:")
    # This is how our fix handles it:
    if hasattr(mock_edu, 'get'):  # Dictionary
        school = mock_edu.get('school') or mock_edu.get('institution')
    else:  # Object with attributes
        school = getattr(mock_edu, 'institution', '')

    print(f"✅ NEW code result: school = '{school}'")
    print(f"   The fix uses hasattr() and getattr() instead of .get()")

    return True

def test_real_world_integration():
    """Test how this would work in the real application"""
    print("\n=== Test 6: Real World Integration Test ===")

    service = BrightDataLinkedInFixed()

    # Create a mock profile (what comes from LinkedIn API)
    profile = ProfileData(
        url="https://www.linkedin.com/in/testuser",
        name="John Smith",
        headline="Software Engineer at TechCorp",
        title="Senior Software Engineer",
        current_company="TechCorp Inc",
        education=[
            {
                'school': 'Stanford University',
                'degree': 'MS Computer Science'
            }
        ],
        experience=[
            {
                'title': 'Senior Software Engineer',
                'company': 'TechCorp Inc'
            }
        ],
        success=True,
        raw_data={'name': 'John Smith'}
    )

    # Resume data with mixed object types (the problematic scenario)
    resume_data = {
        'name': 'John Smith',
        'education': [
            MockEducationEntry('MS Computer Science', 'Stanford University')
        ],
        'experience': [
            MockExperienceEntry('Senior Software Engineer', 'TechCorp Inc')
        ],
        'skills': ['Python', 'JavaScript']
    }

    try:
        # Test the safe verification method
        result = service.verify_against_resume_safe(profile, resume_data)

        print(f"✅ Real world integration test PASSED")
        print(f"   Overall match score: {result['overall_match_score']:.2f}")
        print(f"   Confidence score: {result['confidence_score']:.2f}")
        print(f"   Fraud indicators: {len(result.get('fraud_indicators', []))}")

        # Check if we got verification results
        if 'verification_results' in result:
            ver_results = result['verification_results']
            if 'education_match' in ver_results:
                edu_match = ver_results['education_match']
                print(f"   Education verification worked: {edu_match.get('details', 'N/A')}")

        print(f"   🎉 No 'get' attribute errors occurred!")
        return True

    except Exception as e:
        print(f"❌ Real world integration FAILED: {e}")
        return False

def main():
    """Run all parsing tests"""
    print("🚀 LinkedIn Profile Parsing Fix Tests")
    print("🎯 Specifically testing the fix for 'EducationEntry object has no attribute get'")
    print("=" * 70)

    tests = [
        test_education_parsing_fix,
        test_experience_parsing_fix,
        test_demonstrate_error_scenario,
        test_real_world_integration
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

    print(f"\n{'=' * 70}")
    print(f"FINAL TEST RESULTS")
    print(f"{'=' * 70}")
    print(f"Tests Run: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(passed/(passed+failed)*100) if (passed+failed) > 0 else 0:.1f}%")

    if failed == 0:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"✅ The 'EducationEntry object has no attribute get' error has been FIXED")
        print(f"✅ Both dictionary and object resume data types are now supported")
        print(f"✅ The LinkedIn verification should work without parsing errors")

        print(f"\n🚀 READY FOR PRODUCTION:")
        print(f"   1. Set your real BRIGHTDATA_API_KEY environment variable")
        print(f"   2. Test in your Streamlit app")
        print(f"   3. The LinkedIn verification should now complete successfully")
        print(f"   4. If parsing still fails, AI fallback will be used automatically")

        print(f"\n🔧 WHAT WAS FIXED:")
        print(f"   • _verify_education() now handles both Dict and EducationEntry objects")
        print(f"   • _verify_experience() now handles both Dict and ExperienceEntry objects")
        print(f"   • Uses hasattr(obj, 'get') to detect dictionaries vs objects")
        print(f"   • Uses getattr(obj, 'attribute', '') for object attribute access")
        print(f"   • Added verify_against_resume_safe() with AI fallback protection")

    else:
        print(f"\n❌ SOME TESTS FAILED")
        print(f"   The fix may need additional work")
        print(f"   Review the failed tests above")

    print(f"\n💡 NEXT STEPS:")
    if failed == 0:
        print(f"   Ready to deploy! The parsing error should be resolved.")
    else:
        print(f"   Debug the failed tests before deploying to production.")

if __name__ == "__main__":
    main()
