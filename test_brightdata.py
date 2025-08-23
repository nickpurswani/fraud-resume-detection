#!/usr/bin/env python3
"""
Test script for Bright Data LinkedIn Service Integration

This script tests the Bright Data LinkedIn service functionality
including initialization, URL validation, profile extraction, and
verification against resume data.

Usage:
    python test_brightdata.py

Requirements:
    - Set BRIGHTDATA_API_KEY environment variable
    - All dependencies installed (requests, etc.)
"""

import os
import sys
import json
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import tempfile

# Add src to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from brightdata_linkedin import (
        BrightDataLinkedInService,
        LinkedInProfileData,
        BrightDataResponse,
        BrightDataStatus
    )
    from config import Config
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


class TestBrightDataService(unittest.TestCase):
    """Test cases for Bright Data LinkedIn Service"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_api_key = "test_api_key_12345"
        self.sample_urls = [
            "https://www.linkedin.com/in/elad-moshe-05a90413/",
            "https://www.linkedin.com/in/jonathan-myrvik-3baa01109",
            "https://www.linkedin.com/in/aviv-tal-75b81/",
            "https://www.linkedin.com/in/bulentakar/"
        ]

        self.sample_resume_data = {
            'name': 'John Doe',
            'location': 'San Francisco, CA',
            'experience': [
                {
                    'company': 'Tech Corp',
                    'title': 'Software Engineer',
                    'duration': '2020-2023'
                },
                {
                    'company': 'StartupXYZ',
                    'title': 'Junior Developer',
                    'duration': '2018-2020'
                }
            ],
            'education': [
                {
                    'school': 'Stanford University',
                    'degree': 'Computer Science',
                    'year': '2018'
                }
            ],
            'skills': [
                'Python', 'JavaScript', 'React', 'Node.js'
            ]
        }

        self.sample_profile_response = {
            'request_id': 'test_123',
            'data': [{
                'url': 'https://www.linkedin.com/in/test-profile/',
                'name': 'John Doe',
                'headline': 'Software Engineer at Tech Corp',
                'location': 'San Francisco Bay Area',
                'summary': 'Experienced software engineer...',
                'experience': [
                    {
                        'title': 'Software Engineer',
                        'company': 'Tech Corp',
                        'duration': '2020 - Present',
                        'location': 'San Francisco, CA'
                    }
                ],
                'education': [
                    {
                        'school': 'Stanford University',
                        'degree': 'Bachelor of Science in Computer Science',
                        'years': '2014 - 2018'
                    }
                ],
                'skills': ['Python', 'JavaScript', 'Machine Learning', 'React'],
                'connections': 500
            }],
            'errors': []
        }

    def test_service_initialization_with_api_key(self):
        """Test service initialization with API key"""
        print("🧪 Testing service initialization with API key...")

        service = BrightDataLinkedInService(api_key=self.test_api_key)

        self.assertEqual(service.api_key, self.test_api_key)
        self.assertIsNotNone(service.headers)
        self.assertIn('Authorization', service.headers)
        self.assertEqual(service.headers['Authorization'], f'Bearer {self.test_api_key}')

        print("✅ Service initialization successful")

    def test_service_initialization_without_api_key(self):
        """Test service initialization without API key raises error"""
        print("🧪 Testing service initialization without API key...")

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as context:
                BrightDataLinkedInService()

            self.assertIn("API key is required", str(context.exception))

        print("✅ Proper error handling for missing API key")

    def test_url_validation(self):
        """Test LinkedIn URL validation"""
        print("🧪 Testing URL validation...")

        service = BrightDataLinkedInService(api_key=self.test_api_key)

        # Test valid URLs
        valid_urls = [
            "https://www.linkedin.com/in/username/",
            "https://linkedin.com/in/username",
            "linkedin.com/in/username/",
            "www.linkedin.com/pub/username/1/2/3"
        ]

        for url in valid_urls:
            validated = service._validate_linkedin_urls([url])
            self.assertEqual(len(validated), 1, f"URL should be valid: {url}")

        # Test invalid URLs
        invalid_urls = [
            "https://facebook.com/username",
            "https://twitter.com/username",
            "https://linkedin.com/company/test",  # Not a profile URL
            "not-a-url-at-all"
        ]

        for url in invalid_urls:
            validated = service._validate_linkedin_urls([url])
            self.assertEqual(len(validated), 0, f"URL should be invalid: {url}")

        print("✅ URL validation working correctly")

    @patch('requests.post')
    def test_extract_profiles_success(self, mock_post):
        """Test successful profile extraction"""
        print("🧪 Testing profile extraction (mocked)...")

        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.sample_profile_response
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        service = BrightDataLinkedInService(api_key=self.test_api_key)

        # Test extraction
        result = service.extract_profiles([self.sample_urls[0]])

        self.assertIsInstance(result, BrightDataResponse)
        self.assertEqual(result.status, BrightDataStatus.COMPLETED)
        self.assertEqual(result.successful_extractions, 1)
        self.assertEqual(len(result.profiles), 1)
        self.assertTrue(result.profiles[0].success)

        print("✅ Profile extraction successful")

    @patch('requests.post')
    def test_extract_profiles_api_error(self, mock_post):
        """Test profile extraction with API error"""
        print("🧪 Testing profile extraction with API error...")

        # Mock API error
        mock_post.side_effect = Exception("API connection failed")

        service = BrightDataLinkedInService(api_key=self.test_api_key)

        result = service.extract_profiles([self.sample_urls[0]])

        self.assertIsInstance(result, BrightDataResponse)
        self.assertEqual(result.status, BrightDataStatus.FAILED)
        self.assertEqual(result.successful_extractions, 0)
        self.assertTrue(len(result.errors) > 0)

        print("✅ API error handling working correctly")

    def test_profile_data_parsing(self):
        """Test profile data parsing from API response"""
        print("🧪 Testing profile data parsing...")

        service = BrightDataLinkedInService(api_key=self.test_api_key)

        # Test with sample data
        sample_data = self.sample_profile_response['data'][0]
        profile = service._parse_profile_data(sample_data)

        self.assertIsInstance(profile, LinkedInProfileData)
        self.assertTrue(profile.success)
        self.assertEqual(profile.full_name, 'John Doe')
        self.assertEqual(profile.headline, 'Software Engineer at Tech Corp')
        self.assertIsNotNone(profile.experience)
        self.assertIsNotNone(profile.education)
        self.assertIsNotNone(profile.skills)

        print("✅ Profile data parsing successful")

    def test_resume_verification(self):
        """Test profile verification against resume"""
        print("🧪 Testing profile verification against resume...")

        service = BrightDataLinkedInService(api_key=self.test_api_key)

        # Create a sample profile
        profile = LinkedInProfileData(
            url="https://www.linkedin.com/in/test/",
            full_name="John Doe",
            headline="Software Engineer",
            location="San Francisco, CA",
            summary="Software engineer...",
            experience=[
                {
                    'title': 'Software Engineer',
                    'company': 'Tech Corp',
                    'duration': '2020-Present'
                }
            ],
            education=[
                {
                    'school': 'Stanford University',
                    'degree': 'Computer Science'
                }
            ],
            skills=['Python', 'JavaScript'],
            success=True,
            extraction_timestamp=datetime.now()
        )

        # Test verification
        results = service.verify_profile_against_resume(profile, self.sample_resume_data)

        self.assertIsInstance(results, dict)
        self.assertIn('overall_match_score', results)
        self.assertIn('name_match', results)
        self.assertIn('experience_match', results)
        self.assertIn('education_match', results)
        self.assertIn('skills_match', results)

        # Check that match scores are reasonable
        self.assertTrue(0 <= results['overall_match_score'] <= 1)

        print("✅ Profile verification working correctly")

    def test_service_status(self):
        """Test service status reporting"""
        print("🧪 Testing service status reporting...")

        service = BrightDataLinkedInService(api_key=self.test_api_key)
        status = service.get_service_status()

        self.assertIsInstance(status, dict)
        self.assertIn('service', status)
        self.assertIn('api_key_configured', status)
        self.assertIn('dataset_id', status)
        self.assertTrue(status['api_key_configured'])

        print("✅ Service status reporting working correctly")

    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        print("🧪 Testing rate limiting...")

        service = BrightDataLinkedInService(api_key=self.test_api_key)

        # Mock time to test rate limiting
        with patch('time.time') as mock_time:
            mock_time.return_value = 1000  # Fixed time

            # Set up rate limit scenario
            service.request_count = service.rate_limit - 1
            service.request_window_start = 999  # 1 second ago

            # This should not trigger rate limiting
            service._check_rate_limit()
            self.assertEqual(service.request_count, service.rate_limit)

        print("✅ Rate limiting logic working correctly")


def run_integration_test():
    """Run integration test with real API (if API key is available)"""
    print("\n" + "="*60)
    print("🚀 RUNNING INTEGRATION TEST")
    print("="*60)

    if not Config.BRIGHTDATA_API_KEY:
        print("⚠️  Integration test skipped - BRIGHTDATA_API_KEY not set")
        print("To run integration test, set the environment variable:")
        print("export BRIGHTDATA_API_KEY='your_api_key_here'")
        return

    try:
        print("🔗 Testing with real Bright Data API...")
        service = BrightDataLinkedInService()

        # Test service status
        status = service.get_service_status()
        print(f"✅ Service configured: {status['api_key_configured']}")

        # Test with a single URL (you might want to use a test profile)
        test_url = "https://www.linkedin.com/in/test-profile-that-may-not-exist/"
        print(f"🧪 Testing extraction with URL: {test_url}")

        result = service.extract_single_profile(test_url)

        if result.success:
            print("✅ Profile extraction successful!")
            print(f"   Name: {result.full_name}")
            print(f"   Headline: {result.headline}")
        else:
            print(f"⚠️  Profile extraction failed (expected for test URL): {result.error_message}")

        print("✅ Integration test completed")

    except Exception as e:
        print(f"❌ Integration test failed: {e}")


def main():
    """Main test runner"""
    print("🧪 BRIGHT DATA LINKEDIN SERVICE TESTS")
    print("="*50)

    # Run unit tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.loadTestsFromTestCase(TestBrightDataService)
    test_runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w'))

    print("🔬 Running unit tests...")
    result = test_runner.run(test_suite)

    if result.wasSuccessful():
        print("✅ All unit tests passed!")
    else:
        print(f"❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")

        for test, traceback in result.failures:
            print(f"   FAIL: {test}")

        for test, traceback in result.errors:
            print(f"   ERROR: {test}")

    # Run integration test
    run_integration_test()

    print("\n🎉 Test execution completed!")

    # Return exit code based on test results
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
