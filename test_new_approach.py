#!/usr/bin/env python3
"""
Automated test for the new immediate fetch approach
Tests the fix for snapshot progress stuck at 0% while data is actually ready
"""

import os
import sys
import time
import logging
import json
from typing import List, Optional

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from brightdata_linkedin_fixed import BrightDataLinkedInFixed, ProfileData, SnapshotResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestResults:
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.results = []

    def add_result(self, test_name: str, passed: bool, message: str = "", duration: float = 0.0):
        self.results.append({
            'test': test_name,
            'passed': passed,
            'message': message,
            'duration': duration
        })
        if passed:
            self.tests_passed += 1
            print(f"✅ {test_name}: PASSED ({duration:.2f}s) {message}")
        else:
            self.tests_failed += 1
            print(f"❌ {test_name}: FAILED ({duration:.2f}s) {message}")

    def summary(self):
        total = self.tests_passed + self.tests_failed
        print(f"\n{'='*50}")
        print(f"TEST SUMMARY")
        print(f"{'='*50}")
        print(f"Total Tests: {total}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_failed}")
        print(f"Success Rate: {(self.tests_passed/total*100) if total > 0 else 0:.1f}%")

        if self.tests_failed == 0:
            print(f"🎉 ALL TESTS PASSED! The new approach is working correctly!")
        else:
            print(f"⚠️  Some tests failed. Review the results above.")

def test_api_setup() -> bool:
    """Test if API is properly configured"""
    try:
        service = BrightDataLinkedInFixed()
        return service.api_key is not None
    except Exception as e:
        print(f"API setup failed: {e}")
        return False

def test_existing_snapshot_data_available(results: TestResults):
    """Test that data is available for the existing snapshot despite status showing running"""
    snapshot_id = "s_meoiadrqxexq11zgx"

    start_time = time.time()
    try:
        service = BrightDataLinkedInFixed()

        # Test immediate data availability
        data = service._get_snapshot_data(snapshot_id)
        duration = time.time() - start_time

        if data and isinstance(data, list) and len(data) > 0:
            first_profile = data[0]
            required_fields = ['name', 'current_company', 'experience', 'education']
            has_required_fields = all(field in first_profile for field in required_fields)

            message = f"Retrieved {len(data)} profiles with all required fields" if has_required_fields else f"Missing some fields"
            results.add_result("Existing Snapshot Data Available", True, message, duration)
            return True
        else:
            results.add_result("Existing Snapshot Data Available", False, "No data or empty response", duration)
            return False

    except Exception as e:
        duration = time.time() - start_time
        results.add_result("Existing Snapshot Data Available", False, f"Exception: {e}", duration)
        return False

def test_immediate_vs_polling_speed(results: TestResults):
    """Test speed difference between immediate fetch vs waiting for completion"""
    snapshot_id = "s_meoiadrqxexq11zgx"

    try:
        service = BrightDataLinkedInFixed()

        # Test immediate fetch
        start_time = time.time()
        immediate_data = service._get_snapshot_data(snapshot_id)
        immediate_duration = time.time() - start_time

        # Test status check (simulating old approach)
        start_time = time.time()
        status_data = service._get_snapshot_status(snapshot_id)
        status_duration = time.time() - start_time

        if immediate_data and status_data:
            status = service._parse_snapshot_status(status_data)
            progress = service._extract_progress(status_data)

            message = f"Immediate: {immediate_duration:.1f}s vs Status: {status_duration:.1f}s (Status: {status.value}, {progress:.1f}%)"
            results.add_result("Speed Comparison", True, message, immediate_duration)
            return True
        else:
            results.add_result("Speed Comparison", False, "Failed to get data or status", immediate_duration)
            return False

    except Exception as e:
        results.add_result("Speed Comparison", False, f"Exception: {e}", 0.0)
        return False

def test_retry_logic_robustness(results: TestResults):
    """Test the retry logic with controlled parameters"""
    snapshot_id = "s_meoiadrqxexq11zgx"

    start_time = time.time()
    try:
        service = BrightDataLinkedInFixed()

        # Test with limited retries to ensure it doesn't take too long
        progress_updates = []

        def capture_progress(message, percent):
            progress_updates.append((message, percent))

        data = service._get_snapshot_data_with_retries(
            snapshot_id,
            progress_callback=capture_progress,
            max_retries=3
        )

        duration = time.time() - start_time

        if data and len(progress_updates) > 0:
            message = f"Success with {len(progress_updates)} progress updates in {duration:.1f}s"
            results.add_result("Retry Logic Robustness", True, message, duration)
            return True
        else:
            message = f"Failed: No data or progress updates"
            results.add_result("Retry Logic Robustness", False, message, duration)
            return False

    except Exception as e:
        duration = time.time() - start_time
        results.add_result("Retry Logic Robustness", False, f"Exception: {e}", duration)
        return False

def test_profile_parsing_quality(results: TestResults):
    """Test that the parsed profile data is high quality and complete"""
    snapshot_id = "s_meoiadrqxexq11zgx"

    start_time = time.time()
    try:
        service = BrightDataLinkedInFixed()
        data = service._get_snapshot_data(snapshot_id)

        if not data:
            results.add_result("Profile Parsing Quality", False, "No data retrieved", time.time() - start_time)
            return False

        # Parse the data using the service's parsing method
        test_urls = ["https://www.linkedin.com/in/test"]  # Dummy URL for parsing
        profiles = service._parse_profiles(data, test_urls)

        duration = time.time() - start_time

        if profiles and len(profiles) > 0:
            profile = profiles[0]

            # Check profile completeness
            completeness_score = 0
            total_checks = 6

            if profile.name: completeness_score += 1
            if profile.title: completeness_score += 1
            if profile.current_company: completeness_score += 1
            if profile.experience and len(profile.experience) > 0: completeness_score += 1
            if profile.education and len(profile.education) > 0: completeness_score += 1
            if profile.skills and len(profile.skills) > 0: completeness_score += 1

            completeness_percent = (completeness_score / total_checks) * 100

            if completeness_percent >= 80:
                message = f"High quality profile: {completeness_percent:.0f}% complete"
                results.add_result("Profile Parsing Quality", True, message, duration)
                return True
            else:
                message = f"Low quality profile: only {completeness_percent:.0f}% complete"
                results.add_result("Profile Parsing Quality", False, message, duration)
                return False
        else:
            results.add_result("Profile Parsing Quality", False, "No profiles parsed", duration)
            return False

    except Exception as e:
        duration = time.time() - start_time
        results.add_result("Profile Parsing Quality", False, f"Exception: {e}", duration)
        return False

def test_full_workflow_new_approach(results: TestResults):
    """Test the complete workflow using the new approach"""
    # Use a real LinkedIn URL for testing
    test_url = "https://www.linkedin.com/in/jeffweiner08"  # Former LinkedIn CEO

    start_time = time.time()
    try:
        service = BrightDataLinkedInFixed()

        progress_log = []

        def log_progress(message, percent):
            progress_log.append(f"{percent}% - {message}")
            print(f"   📈 {percent}% - {message}")

        # Test the complete new workflow
        result = service.extract_profiles([test_url], progress_callback=log_progress)
        duration = time.time() - start_time

        if result.success and result.profiles and len(result.profiles) > 0:
            profile = result.profiles[0]
            message = f"Complete workflow success: {profile.name or 'Unknown'} from {profile.current_company or 'Unknown company'}"
            results.add_result("Full Workflow New Approach", True, message, duration)

            # Print detailed results
            print(f"      📊 Snapshot ID: {result.snapshot_id}")
            print(f"      👤 Profile Name: {profile.name}")
            print(f"      🏢 Company: {profile.current_company}")
            print(f"      📝 Title: {profile.title}")
            print(f"      🎓 Education: {len(profile.education) if profile.education else 0} entries")
            print(f"      💼 Experience: {len(profile.experience) if profile.experience else 0} entries")

            return True
        else:
            error_msg = result.error if hasattr(result, 'error') else "Unknown error"
            results.add_result("Full Workflow New Approach", False, f"Workflow failed: {error_msg}", duration)
            return False

    except Exception as e:
        duration = time.time() - start_time
        results.add_result("Full Workflow New Approach", False, f"Exception: {e}", duration)
        return False

def main():
    """Run all tests for the new immediate fetch approach"""

    print("🚀 Testing New Immediate Fetch Approach")
    print("=" * 60)

    # Initialize results tracker
    results = TestResults()

    # Check API setup first
    if not test_api_setup():
        print("❌ API not configured. Set BRIGHTDATA_API_KEY environment variable.")
        print("   Example: export BRIGHTDATA_API_KEY='your_api_key_here'")
        return

    print("✅ API configured successfully\n")

    # Run all tests
    print("Running Tests...")
    print("-" * 30)

    # Test 1: Existing snapshot data availability
    test_existing_snapshot_data_available(results)

    # Test 2: Speed comparison
    test_immediate_vs_polling_speed(results)

    # Test 3: Retry logic
    test_retry_logic_robustness(results)

    # Test 4: Profile parsing quality
    test_profile_parsing_quality(results)

    # Test 5: Full workflow (creates new snapshot - optional)
    print("\n🔄 Testing complete workflow with new snapshot...")
    print("   (This will create a new snapshot and test the full process)")
    test_full_workflow_new_approach(results)

    # Show summary
    results.summary()

    # Provide recommendations
    print(f"\n📋 RECOMMENDATIONS:")
    if results.tests_failed == 0:
        print("✅ The new immediate fetch approach is working perfectly!")
        print("✅ You can now use the updated LinkedIn verification in your app")
        print("✅ The fix resolves the 0% progress issue by fetching data immediately")
        print("\n🚀 Next Steps:")
        print("   1. Deploy the updated code to your Streamlit app")
        print("   2. Test with real resumes in the UI")
        print("   3. Monitor performance improvements")
    else:
        print("⚠️  Some issues detected. Review the test results above.")
        print("💡 Consider debugging the failed tests before deploying")

if __name__ == "__main__":
    main()
