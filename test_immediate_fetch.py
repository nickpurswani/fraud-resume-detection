#!/usr/bin/env python3
"""
Test script for immediate data fetching approach
This tests the new retry logic that doesn't wait for snapshot completion
"""

import os
import sys
import time
import logging
from typing import Optional

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from brightdata_linkedin_fixed import BrightDataLinkedInFixed, SnapshotStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_immediate_fetch_with_existing_snapshot():
    """Test immediate fetching with the existing snapshot ID"""

    # Your existing snapshot ID from the logs
    snapshot_id = "s_meoiadrqxexq11zgx"

    print(f"\n=== Testing Immediate Data Fetch with Snapshot {snapshot_id} ===")

    # Initialize the LinkedIn service
    service = BrightDataLinkedInFixed()

    if not service.api_key:
        print("❌ No API key found. Set BRIGHTDATA_API_KEY environment variable.")
        return False

    print(f"✅ Service initialized with API key: {service.api_key[:20]}...")

    # Test 1: Quick data check
    print(f"\n1. Quick data availability check...")
    has_data = service._quick_data_check(snapshot_id)
    print(f"   Result: {'✅ Data available' if has_data else '❌ No data yet'}")

    # Test 2: Single data fetch attempt
    print(f"\n2. Single data fetch attempt...")
    start_time = time.time()
    data = service._get_snapshot_data(snapshot_id)
    fetch_time = time.time() - start_time

    if data:
        data_info = ""
        if isinstance(data, list):
            data_info = f"List with {len(data)} items"
        elif isinstance(data, dict):
            data_info = f"Dict with keys: {list(data.keys())}"
        else:
            data_info = f"Type: {type(data)}"

        print(f"   ✅ SUCCESS: Retrieved data in {fetch_time:.2f}s")
        print(f"   📊 Data info: {data_info}")

        # Show first item preview if it's a list
        if isinstance(data, list) and len(data) > 0:
            first_item = data[0]
            if isinstance(first_item, dict):
                print(f"   🔍 First item keys: {list(first_item.keys())}")

    else:
        print(f"   ❌ FAILED: No data retrieved in {fetch_time:.2f}s")

    # Test 3: Retry logic test
    print(f"\n3. Testing retry logic with max 3 attempts...")

    def progress_callback(message, percent):
        print(f"   📈 Progress: {message} ({percent}%)")

    start_time = time.time()
    retry_data = service._get_snapshot_data_with_retries(
        snapshot_id,
        progress_callback=progress_callback,
        max_retries=3
    )
    retry_time = time.time() - start_time

    if retry_data:
        print(f"   ✅ SUCCESS: Retry logic worked in {retry_time:.2f}s")
        return True
    else:
        print(f"   ❌ FAILED: Retry logic failed after {retry_time:.2f}s")
        return False

def test_new_vs_old_approach():
    """Test creating a new snapshot and comparing approaches"""

    print(f"\n=== Testing New vs Old Approach ===")

    # Test LinkedIn URL
    test_url = "https://www.linkedin.com/in/satyanadella/"  # Microsoft CEO

    service = BrightDataLinkedInFixed()

    if not service.api_key:
        print("❌ No API key found. Skipping new snapshot test.")
        return

    print(f"🔄 Creating new snapshot for: {test_url}")

    def progress_callback(message, percent):
        print(f"📈 {message} ({percent}%)")

    # Test the new approach
    start_time = time.time()
    result = service.extract_profiles([test_url], progress_callback)
    total_time = time.time() - start_time

    print(f"\n📊 Results:")
    print(f"   Time taken: {total_time:.2f}s")
    print(f"   Status: {result.status.value}")
    print(f"   Snapshot ID: {result.snapshot_id}")
    print(f"   Profiles extracted: {len(result.profiles)}")

    if result.success:
        print(f"   ✅ SUCCESS: New approach worked!")

        for i, profile in enumerate(result.profiles):
            print(f"   👤 Profile {i+1}:")
            print(f"      URL: {profile.url}")
            print(f"      Success: {profile.success}")
            if profile.success:
                print(f"      Name: {profile.name}")
                print(f"      Title: {profile.title}")
                print(f"      Company: {profile.current_company}")
    else:
        print(f"   ❌ FAILED: {result.error}")

def test_status_vs_data_availability():
    """Test whether data is available even when status shows running"""

    snapshot_id = "s_meoiadrqxexq11zgx"
    print(f"\n=== Testing Status vs Data Availability ===")

    service = BrightDataLinkedInFixed()

    if not service.api_key:
        print("❌ No API key found.")
        return

    # Check current status
    print(f"1. Checking current snapshot status...")
    try:
        status_data = service._get_snapshot_status(snapshot_id)
        if status_data:
            status = service._parse_snapshot_status(status_data)
            progress = service._extract_progress(status_data)
            print(f"   Status: {status.value}")
            print(f"   Progress: {progress:.1f}%")
        else:
            print(f"   ❌ Could not get status")
    except Exception as e:
        print(f"   ❌ Status check failed: {e}")

    # Check data availability
    print(f"2. Checking data availability...")
    data = service._get_snapshot_data(snapshot_id)
    if data:
        print(f"   ✅ Data IS available despite status!")
        if isinstance(data, list):
            print(f"   📊 Found {len(data)} items")
        elif isinstance(data, dict):
            print(f"   📊 Found data dict")
    else:
        print(f"   ❌ No data available")

def main():
    """Run all tests"""
    print("🚀 Starting Immediate Fetch Tests")
    print("=" * 50)

    # Test 1: Existing snapshot
    success1 = test_immediate_fetch_with_existing_snapshot()

    # Test 2: Status vs Data
    test_status_vs_data_availability()

    # Test 3: New approach (optional - creates new snapshot)
    if input("\nCreate new snapshot for testing? (y/N): ").lower().startswith('y'):
        test_new_vs_old_approach()

    print(f"\n{'=' * 50}")
    if success1:
        print("🎉 Tests completed successfully!")
        print("✅ The immediate fetch approach works!")
    else:
        print("⚠️  Some tests failed - may need debugging")

if __name__ == "__main__":
    main()
