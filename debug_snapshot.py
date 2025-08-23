#!/usr/bin/env python3
"""
Debug script for Bright Data snapshot polling

This script helps debug issues with snapshot creation and polling.
It provides detailed logging and step-by-step analysis of the workflow.

Usage:
    python debug_snapshot.py
"""

import os
import sys
import json
import time
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('snapshot_debug.log')
    ]
)

try:
    from brightdata_linkedin import BrightDataLinkedInService, SnapshotStatus
    from config import Config
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def debug_snapshot_workflow():
    """Debug the complete snapshot workflow with detailed logging"""

    print("🔍 BRIGHT DATA SNAPSHOT DEBUG SCRIPT")
    print("=" * 60)

    # Check API key
    if not Config.BRIGHTDATA_API_KEY:
        print("❌ BRIGHTDATA_API_KEY not set!")
        print("Set it with: export BRIGHTDATA_API_KEY='your_key_here'")
        return False

    api_key = Config.BRIGHTDATA_API_KEY
    print(f"✅ API Key configured: {api_key[:10]}...{api_key[-4:]}")

    # Test URL - using a simple one that should work
    test_url = "https://www.linkedin.com/in/elad-moshe-05a90413/"
    print(f"📋 Test URL: {test_url}")

    try:
        # Initialize service
        print("\n🚀 STEP 1: Initialize Service")
        print("-" * 30)

        service = BrightDataLinkedInService()
        status = service.get_service_status()

        print("Service Configuration:")
        for key, value in status.items():
            print(f"  {key}: {value}")

        # Step 1: Trigger snapshot
        print(f"\n🚀 STEP 2: Trigger Snapshot")
        print("-" * 30)

        trigger_start = time.time()
        snapshot_id = service._trigger_snapshot([test_url])
        trigger_time = time.time() - trigger_start

        if not snapshot_id:
            print("❌ Failed to create snapshot!")
            return False

        print(f"✅ Snapshot created: {snapshot_id}")
        print(f"⏱️  Trigger time: {trigger_time:.2f}s")

        # Step 2: Debug polling process
        print(f"\n⏳ STEP 3: Debug Polling Process")
        print("-" * 30)

        poll_start = time.time()
        max_attempts = 30  # 5 minutes worth
        poll_interval = 10  # 10 seconds for debugging

        for attempt in range(max_attempts):
            try:
                elapsed = time.time() - poll_start
                print(f"\n📊 Polling attempt {attempt + 1}/{max_attempts} (elapsed: {elapsed:.0f}s)")

                # Get current status
                snapshot_info = service.get_snapshot_status(snapshot_id)

                print(f"Status Details:")
                print(f"  ID: {snapshot_info.snapshot_id}")
                print(f"  Status: {snapshot_info.status.value}")
                print(f"  Progress: {snapshot_info.progress_percent:.1f}%")
                print(f"  Total Records: {snapshot_info.total_records}")
                print(f"  Completed Records: {snapshot_info.completed_records}")
                print(f"  Failed Records: {snapshot_info.failed_records}")
                print(f"  Error Message: {snapshot_info.error_message}")
                print(f"  Created At: {snapshot_info.created_at}")
                print(f"  Completed At: {snapshot_info.completed_at}")

                # Check if done
                if snapshot_info.status == SnapshotStatus.COMPLETED:
                    print(f"✅ Snapshot completed after {elapsed:.0f}s!")
                    break
                elif snapshot_info.status == SnapshotStatus.FAILED:
                    print(f"❌ Snapshot failed: {snapshot_info.error_message}")
                    return False
                elif snapshot_info.status == SnapshotStatus.CANCELED:
                    print(f"❌ Snapshot canceled")
                    return False

                # Show what we're waiting for
                if snapshot_info.status == SnapshotStatus.PENDING:
                    print("🟡 Snapshot is pending - waiting to start processing")
                elif snapshot_info.status == SnapshotStatus.RUNNING:
                    if snapshot_info.progress_percent == 0.0:
                        print("🟡 Snapshot is running but no progress yet - data collection in progress")
                    else:
                        print(f"🟢 Snapshot making progress: {snapshot_info.progress_percent:.1f}%")

                # Wait before next check
                if attempt < max_attempts - 1:  # Don't wait on last attempt
                    print(f"⏳ Waiting {poll_interval}s before next check...")
                    time.sleep(poll_interval)

            except Exception as e:
                print(f"❌ Polling error on attempt {attempt + 1}: {e}")
                import traceback
                traceback.print_exc()

                if attempt < max_attempts - 1:
                    print(f"🔄 Retrying in {poll_interval}s...")
                    time.sleep(poll_interval)

        # Check final status
        final_elapsed = time.time() - poll_start
        if snapshot_info.status != SnapshotStatus.COMPLETED:
            print(f"❌ Snapshot did not complete after {final_elapsed:.0f}s")
            print(f"Final status: {snapshot_info.status.value}")
            print(f"Final progress: {snapshot_info.progress_percent:.1f}%")
            return False

        # Step 3: Try to retrieve data
        print(f"\n📥 STEP 4: Retrieve Snapshot Data")
        print("-" * 30)

        try:
            data_start = time.time()
            snapshot_data = service._get_snapshot_data(snapshot_id)
            data_time = time.time() - data_start

            print(f"✅ Data retrieved in {data_time:.2f}s")
            print(f"📊 Data type: {type(snapshot_data)}")

            if isinstance(snapshot_data, dict):
                print(f"📋 Keys: {list(snapshot_data.keys())}")
            elif isinstance(snapshot_data, list):
                print(f"📋 Items: {len(snapshot_data)}")
                if snapshot_data:
                    print(f"📋 Sample item keys: {list(snapshot_data[0].keys()) if isinstance(snapshot_data[0], dict) else 'Not a dict'}")

            # Try to parse it
            total_time = time.time() - trigger_start
            response = service._parse_snapshot_response(
                snapshot_data, [test_url], total_time, snapshot_id, snapshot_info
            )

            print(f"\n📊 FINAL RESULTS:")
            print(f"  Request ID: {response.request_id}")
            print(f"  Snapshot ID: {response.snapshot_id}")
            print(f"  Status: {response.status.value}")
            print(f"  Total Profiles: {response.total_profiles}")
            print(f"  Successful: {response.successful_extractions}")
            print(f"  Failed: {response.failed_extractions}")
            print(f"  Execution Time: {response.execution_time:.2f}s")

            if response.profiles:
                profile = response.profiles[0]
                print(f"\n👤 Profile Results:")
                print(f"  Success: {profile.success}")
                print(f"  Name: {profile.full_name}")
                print(f"  Headline: {profile.headline}")
                print(f"  Location: {profile.location}")
                if profile.experience:
                    print(f"  Experience: {len(profile.experience)} positions")
                if profile.skills:
                    print(f"  Skills: {len(profile.skills)} skills")
                if profile.error_message:
                    print(f"  Error: {profile.error_message}")

            return True

        except Exception as e:
            print(f"❌ Failed to retrieve data: {e}")
            import traceback
            traceback.print_exc()
            return False

    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_endpoints():
    """Test individual API endpoints"""

    print("\n🔧 TESTING INDIVIDUAL API ENDPOINTS")
    print("=" * 60)

    api_key = Config.BRIGHTDATA_API_KEY
    if not api_key:
        print("❌ No API key available for endpoint testing")
        return

    import requests

    headers = {'Authorization': f'Bearer {api_key}'}
    base_url = "https://api.brightdata.com/datasets/v3"

    # Test 1: Trigger endpoint
    print("🚀 Testing trigger endpoint...")
    try:
        response = requests.post(
            f"{base_url}/trigger",
            headers={**headers, 'Content-Type': 'application/json'},
            json=[{"url": "https://www.linkedin.com/in/test-profile/"}],
            params={'dataset_id': 'gd_l1viktl72bvl7bjuj0', 'include_errors': 'true'},
            timeout=30
        )
        print(f"  Status: {response.status_code}")
        print(f"  Response: {response.text[:200]}...")

        if response.status_code == 200:
            data = response.json()
            test_snapshot_id = data.get('snapshot_id') or data.get('id')

            if test_snapshot_id:
                print(f"  Snapshot ID: {test_snapshot_id}")

                # Test 2: Progress endpoint
                print(f"\n📊 Testing progress endpoint...")
                time.sleep(5)  # Wait a bit

                progress_response = requests.get(
                    f"{base_url}/progress/{test_snapshot_id}",
                    headers={'Authorization': headers['Authorization']},
                    timeout=30
                )
                print(f"  Status: {progress_response.status_code}")
                print(f"  Response: {progress_response.text[:200]}...")

    except Exception as e:
        print(f"  Error: {e}")


def main():
    """Main debug function"""

    print("Starting Bright Data snapshot debug session...")
    print(f"Timestamp: {datetime.now()}")
    print(f"Log file: snapshot_debug.log")

    # Test basic setup
    success = debug_snapshot_workflow()

    if not success:
        print("\n🔧 Running additional endpoint tests...")
        test_api_endpoints()

    print(f"\n🎯 Debug session completed!")
    print(f"Check 'snapshot_debug.log' for detailed logs")

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
