#!/usr/bin/env python3
"""
Test script for the fixed Bright Data snapshot workflow

This script tests the complete async workflow:
1. Create snapshot
2. Poll until complete
3. Retrieve and parse data

Usage:
    python test_snapshot_fixed.py
"""

import os
import sys
import json
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_fixed_snapshot_workflow():
    """Test the fixed snapshot workflow"""

    print("🔍 TESTING FIXED BRIGHT DATA SNAPSHOT WORKFLOW")
    print("=" * 60)
    print(f"Timestamp: {datetime.now()}")

    # Check API key
    api_key = os.getenv('BRIGHTDATA_API_KEY')
    if not api_key:
        print("❌ BRIGHTDATA_API_KEY not set!")
        print("Set it with: export BRIGHTDATA_API_KEY='your_key_here'")
        return False

    print(f"✅ API Key: {api_key[:10]}...{api_key[-4:]}")

    try:
        # Import the fixed service
        from brightdata_linkedin_fixed import BrightDataLinkedInFixed, SnapshotStatus

        print("✅ Fixed service imported successfully")

        # Initialize service
        service = BrightDataLinkedInFixed(api_key)
        print("✅ Service initialized")

        # Test URLs - use the working examples
        test_urls = [
            "https://www.linkedin.com/in/elad-moshe-05a90413/",
            "https://www.linkedin.com/in/jonathan-myrvik-3baa01109"
        ]

        print(f"📋 Testing with {len(test_urls)} LinkedIn URLs:")
        for i, url in enumerate(test_urls, 1):
            print(f"   {i}. {url}")

        # Progress callback for real-time updates
        def progress_callback(message, percent):
            print(f"📊 Progress {percent:3d}%: {message}")

        # Execute the complete workflow
        print(f"\n🚀 EXECUTING COMPLETE WORKFLOW")
        print("-" * 40)

        start_time = time.time()
        result = service.extract_profiles(test_urls, progress_callback)
        total_time = time.time() - start_time

        print(f"\n📈 WORKFLOW RESULTS")
        print("=" * 40)
        print(f"✅ Success: {result.success}")
        print(f"📸 Snapshot ID: {result.snapshot_id}")
        print(f"🎯 Status: {result.status.value}")
        print(f"⏱️  Total Time: {total_time:.2f} seconds")
        print(f"📊 Total Profiles: {len(result.profiles)}")

        if result.error:
            print(f"❌ Error: {result.error}")

        if result.metadata:
            print(f"📋 Metadata:")
            for key, value in result.metadata.items():
                print(f"   {key}: {value}")

        # Display profile results
        print(f"\n👤 PROFILE RESULTS")
        print("=" * 40)

        for i, profile in enumerate(result.profiles, 1):
            print(f"\nProfile {i}:")
            print(f"  URL: {profile.url}")
            print(f"  Success: {'✅' if profile.success else '❌'}")

            if profile.success:
                print(f"  Name: {profile.name or 'N/A'}")
                print(f"  Headline: {profile.headline or 'N/A'}")
                print(f"  Location: {profile.location or 'N/A'}")
                print(f"  Connections: {profile.connections or 'N/A'}")

                if profile.experience:
                    print(f"  Experience: {len(profile.experience)} positions")
                    for j, exp in enumerate(profile.experience[:2], 1):  # Show first 2
                        print(f"    {j}. {exp.get('title', 'N/A')} at {exp.get('company', 'N/A')}")

                if profile.education:
                    print(f"  Education: {len(profile.education)} entries")
                    for j, edu in enumerate(profile.education[:2], 1):  # Show first 2
                        print(f"    {j}. {edu.get('degree', 'N/A')} from {edu.get('school', 'N/A')}")

                if profile.skills:
                    print(f"  Skills: {len(profile.skills)} skills")
                    skills_preview = ', '.join(profile.skills[:5])
                    if len(profile.skills) > 5:
                        skills_preview += f" ... (+{len(profile.skills) - 5} more)"
                    print(f"    {skills_preview}")
            else:
                print(f"  Error: {profile.error}")

        # Test verification against sample resume
        if result.success and any(p.success for p in result.profiles):
            print(f"\n🔍 TESTING RESUME VERIFICATION")
            print("=" * 40)

            successful_profile = next(p for p in result.profiles if p.success)

            sample_resume = {
                'name': 'Elad Moshe',
                'experience': [
                    {'company': 'Microsoft', 'title': 'Software Engineer'},
                    {'company': 'Google', 'title': 'Developer'}
                ],
                'education': [
                    {'school': 'Tel Aviv University', 'degree': 'Computer Science'}
                ],
                'skills': ['Python', 'JavaScript', 'React', 'Node.js']
            }

            verification = service.verify_against_resume(successful_profile, sample_resume)

            print(f"📊 Verification Results:")
            print(f"   Overall Match: {verification['overall_match_score']:.2%}")
            print(f"   Confidence: {verification['confidence_score']:.2%}")
            print(f"   Service: {verification['service_used']}")

            if verification['fraud_indicators']:
                print(f"⚠️  Fraud Indicators:")
                for indicator in verification['fraud_indicators']:
                    print(f"     - {indicator}")
            else:
                print(f"✅ No fraud indicators detected")

        # Save results to file
        output_file = f"snapshot_test_results_{int(time.time())}.json"
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'test_urls': test_urls,
            'success': result.success,
            'snapshot_id': result.snapshot_id,
            'status': result.status.value,
            'total_time': total_time,
            'profiles': []
        }

        for profile in result.profiles:
            profile_data = {
                'url': profile.url,
                'success': profile.success,
                'name': profile.name,
                'headline': profile.headline,
                'location': profile.location,
                'connections': profile.connections,
                'experience_count': len(profile.experience) if profile.experience else 0,
                'education_count': len(profile.education) if profile.education else 0,
                'skills_count': len(profile.skills) if profile.skills else 0,
                'error': profile.error
            }
            output_data['profiles'].append(profile_data)

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)

        print(f"\n💾 Results saved to: {output_file}")

        return result.success

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running from the project root directory")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_steps():
    """Test individual workflow steps for debugging"""

    print(f"\n🔧 TESTING INDIVIDUAL STEPS")
    print("=" * 40)

    try:
        from brightdata_linkedin_fixed import BrightDataLinkedInFixed

        service = BrightDataLinkedInFixed()
        test_urls = ["https://www.linkedin.com/in/elad-moshe-05a90413/"]

        print("Step 1: URL Validation")
        valid_urls = service._validate_urls(test_urls)
        print(f"   Valid URLs: {len(valid_urls)}/{len(test_urls)}")

        print("Step 2: Create Snapshot")
        snapshot_id = service._create_snapshot(valid_urls)
        print(f"   Snapshot ID: {snapshot_id}")

        if snapshot_id:
            print("Step 3: Check Status")
            status_data = service._get_snapshot_status(snapshot_id)
            print(f"   Status data: {bool(status_data)}")

            if status_data:
                status = service._parse_snapshot_status(status_data)
                progress = service._extract_progress(status_data)
                print(f"   Status: {status.value}")
                print(f"   Progress: {progress:.1f}%")

        return True

    except Exception as e:
        print(f"❌ Individual steps test failed: {e}")
        return False

def main():
    """Main test runner"""

    print("🧪 BRIGHT DATA FIXED SNAPSHOT TESTS")
    print("=" * 60)

    # Test complete workflow
    workflow_success = test_fixed_snapshot_workflow()

    if not workflow_success:
        print("\n🔧 Running individual steps test for debugging...")
        test_individual_steps()

    print(f"\n🎯 TEST SUMMARY")
    print("=" * 30)
    if workflow_success:
        print("✅ All tests passed!")
        print("🎉 Fixed snapshot workflow is working correctly")
        return 0
    else:
        print("❌ Tests failed")
        print("🔍 Check the error messages above for debugging")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
