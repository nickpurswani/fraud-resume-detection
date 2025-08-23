#!/usr/bin/env python3
"""
Working Example - Bright Data LinkedIn Fixed Snapshot Workflow

This example demonstrates the complete working implementation of the Bright Data
LinkedIn service with proper snapshot handling.

Requirements:
- Set BRIGHTDATA_API_KEY environment variable
- Run from project root directory

Usage:
    python working_example.py
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def working_snapshot_example():
    """Complete working example of the snapshot workflow"""

    print("🚀 BRIGHT DATA LINKEDIN - WORKING EXAMPLE")
    print("=" * 60)
    print(f"Timestamp: {datetime.now()}")

    # Check API key
    api_key = os.getenv('BRIGHTDATA_API_KEY')
    if not api_key:
        print("❌ Missing BRIGHTDATA_API_KEY environment variable")
        print("\nTo run this example:")
        print("1. Get your Bright Data API key from https://brightdata.com")
        print("2. Set the environment variable:")
        print("   export BRIGHTDATA_API_KEY='your_api_key_here'")
        print("3. Run this script again")
        return False

    print(f"✅ API Key configured: {api_key[:10]}...{api_key[-4:]}")

    try:
        # Import the fixed service
        from brightdata_linkedin_fixed import BrightDataLinkedInFixed

        # Initialize service
        service = BrightDataLinkedInFixed(api_key)
        print("✅ Service initialized successfully")

        # Test URLs - known working LinkedIn profiles
        test_urls = [
            "https://www.linkedin.com/in/elad-moshe-05a90413/",
            "https://www.linkedin.com/in/jonathan-myrvik-3baa01109",
            "https://www.linkedin.com/in/aviv-tal-75b81/"
        ]

        print(f"\n📋 Testing with {len(test_urls)} LinkedIn URLs:")
        for i, url in enumerate(test_urls, 1):
            print(f"   {i}. {url}")

        # Progress tracking function
        progress_messages = []
        def track_progress(message: str, percent: float):
            """Track progress with timestamps"""
            timestamp = datetime.now().strftime("%H:%M:%S")
            progress_msg = f"[{timestamp}] {percent:3.0f}% - {message}"
            print(f"📊 {progress_msg}")
            progress_messages.append(progress_msg)

        # Execute the complete workflow
        print(f"\n🔄 STARTING COMPLETE WORKFLOW")
        print("-" * 50)

        start_time = time.time()
        result = service.extract_profiles(test_urls, track_progress)
        total_time = time.time() - start_time

        # Display results
        print(f"\n📈 WORKFLOW COMPLETED")
        print("=" * 50)
        print(f"✅ Success: {result.success}")
        print(f"📸 Snapshot ID: {result.snapshot_id}")
        print(f"🎯 Final Status: {result.status.value}")
        print(f"⏱️  Total Processing Time: {total_time:.2f} seconds")
        print(f"📊 Profile Results: {len(result.profiles)} total")

        if result.error:
            print(f"❌ Error: {result.error}")
            return False

        # Show metadata
        if result.metadata:
            print(f"\n📋 Processing Metadata:")
            for key, value in result.metadata.items():
                print(f"   {key}: {value}")

        # Analyze each profile
        print(f"\n👤 PROFILE ANALYSIS")
        print("=" * 50)

        successful_profiles = []
        failed_profiles = []

        for i, profile in enumerate(result.profiles, 1):
            print(f"\nProfile {i}: {profile.url}")
            print(f"Status: {'✅ Success' if profile.success else '❌ Failed'}")

            if profile.success:
                successful_profiles.append(profile)
                print(f"   Name: {profile.name or 'Not available'}")
                print(f"   Headline: {profile.headline or 'Not available'}")
                print(f"   Location: {profile.location or 'Not available'}")
                print(f"   Connections: {profile.connections or 'Not available'}")

                if profile.experience:
                    print(f"   Experience: {len(profile.experience)} positions")
                    # Show top 2 positions
                    for j, exp in enumerate(profile.experience[:2], 1):
                        title = exp.get('title', 'Unknown')
                        company = exp.get('company', 'Unknown')
                        print(f"      {j}. {title} at {company}")

                if profile.education:
                    print(f"   Education: {len(profile.education)} entries")
                    # Show top 2 education entries
                    for j, edu in enumerate(profile.education[:2], 1):
                        degree = edu.get('degree', 'Unknown')
                        school = edu.get('school', 'Unknown')
                        print(f"      {j}. {degree} from {school}")

                if profile.skills:
                    skills_preview = ', '.join(profile.skills[:5])
                    if len(profile.skills) > 5:
                        skills_preview += f" ... (+{len(profile.skills) - 5} more)"
                    print(f"   Skills ({len(profile.skills)}): {skills_preview}")

            else:
                failed_profiles.append(profile)
                print(f"   Error: {profile.error}")

        # Resume verification example (if we have successful profiles)
        if successful_profiles:
            print(f"\n🔍 RESUME VERIFICATION EXAMPLE")
            print("=" * 50)

            # Sample resume data for testing
            sample_resume = {
                'name': 'John Doe',  # This won't match exactly, showing how verification works
                'experience': [
                    {'company': 'Tech Corp', 'title': 'Software Engineer'},
                    {'company': 'Microsoft', 'title': 'Developer'},
                    {'company': 'Google', 'title': 'Senior Engineer'}
                ],
                'education': [
                    {'school': 'MIT', 'degree': 'Computer Science'},
                    {'school': 'Stanford University', 'degree': 'Masters'}
                ],
                'skills': ['Python', 'JavaScript', 'React', 'Node.js', 'AWS', 'Docker']
            }

            # Use first successful profile for verification
            test_profile = successful_profiles[0]
            print(f"Verifying LinkedIn profile against sample resume...")
            print(f"LinkedIn Profile: {test_profile.name}")
            print(f"Sample Resume: {sample_resume['name']}")

            verification = service.verify_against_resume(test_profile, sample_resume)

            print(f"\n📊 Verification Results:")
            print(f"   Overall Match Score: {verification['overall_match_score']:.1%}")
            print(f"   Confidence Score: {verification['confidence_score']:.1%}")
            print(f"   Service Used: {verification['service_used']}")

            # Show detailed verification results
            if verification.get('verification_results'):
                print(f"\n🔍 Detailed Verification:")
                for category, details in verification['verification_results'].items():
                    if isinstance(details, dict):
                        match_status = "✅" if details.get('match') else "❌"
                        score = details.get('score', 0)
                        print(f"   {match_status} {category}: {score:.1%} - {details.get('details', 'N/A')}")

            # Show fraud indicators
            fraud_indicators = verification.get('fraud_indicators', [])
            if fraud_indicators:
                print(f"\n⚠️  Fraud Risk Indicators:")
                for indicator in fraud_indicators:
                    print(f"   - {indicator}")
            else:
                print(f"\n✅ No fraud indicators detected")

        # Save detailed results to file
        output_file = f"brightdata_working_example_{int(time.time())}.json"
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'api_key_prefix': f"{api_key[:10]}...{api_key[-4:]}",
            'test_urls': test_urls,
            'workflow_results': {
                'success': result.success,
                'snapshot_id': result.snapshot_id,
                'status': result.status.value,
                'total_time': total_time,
                'metadata': result.metadata
            },
            'progress_messages': progress_messages,
            'profiles': [],
            'verification_example': None
        }

        # Add profile summaries
        for profile in result.profiles:
            profile_summary = {
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
            output_data['profiles'].append(profile_summary)

        # Add verification example if available
        if successful_profiles:
            output_data['verification_example'] = {
                'sample_resume': sample_resume,
                'verification_results': verification
            }

        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, default=str)

        print(f"\n💾 RESULTS SAVED")
        print("=" * 30)
        print(f"File: {output_file}")
        print(f"Size: {os.path.getsize(output_file):,} bytes")

        # Summary
        print(f"\n🎯 EXECUTION SUMMARY")
        print("=" * 40)
        print(f"✅ Total profiles processed: {len(result.profiles)}")
        print(f"✅ Successful extractions: {len(successful_profiles)}")
        print(f"❌ Failed extractions: {len(failed_profiles)}")
        print(f"⏱️  Average time per profile: {total_time/len(test_urls):.1f} seconds")
        print(f"📊 Success rate: {len(successful_profiles)/len(test_urls):.1%}")

        if len(successful_profiles) > 0:
            print(f"\n🎉 SUCCESS! The fixed snapshot workflow is working correctly!")
            print(f"✅ Snapshots are being created and processed")
            print(f"✅ Profile data is being extracted successfully")
            print(f"✅ Resume verification is working")
            return True
        else:
            print(f"\n⚠️  PARTIAL SUCCESS - Workflow completed but no profiles extracted")
            print(f"💡 This might be due to:")
            print(f"   - Private LinkedIn profiles")
            print(f"   - Rate limiting")
            print(f"   - Temporary service issues")
            return False

    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print(f"\nMake sure you're running from the project root directory")
        print(f"Expected file: src/brightdata_linkedin_fixed.py")
        return False

    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        import traceback
        print(f"\nFull traceback:")
        traceback.print_exc()
        return False

def quick_single_profile_test():
    """Quick test with a single profile"""

    print(f"\n🔧 QUICK SINGLE PROFILE TEST")
    print("=" * 40)

    try:
        from brightdata_linkedin_fixed import BrightDataLinkedInFixed

        service = BrightDataLinkedInFixed()
        test_url = "https://www.linkedin.com/in/elad-moshe-05a90413/"

        print(f"Testing single profile: {test_url}")

        def simple_progress(message, percent):
            if percent % 20 == 0 or percent > 90:  # Show every 20% and final stages
                print(f"   {percent:3.0f}% - {message}")

        start_time = time.time()
        profile = service.extract_single_profile(test_url, simple_progress)
        elapsed = time.time() - start_time

        if profile.success:
            print(f"✅ Single profile test successful in {elapsed:.1f}s")
            print(f"   Name: {profile.name}")
            print(f"   Headline: {profile.headline}")
            return True
        else:
            print(f"❌ Single profile test failed: {profile.error}")
            return False

    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

def main():
    """Main function"""

    print("🧪 BRIGHT DATA LINKEDIN - COMPLETE WORKING EXAMPLE")
    print("="*60)
    print("This example demonstrates the complete fixed snapshot workflow")
    print("that properly handles Bright Data's async API.")
    print()

    # Check basic requirements
    if not os.path.exists('src/brightdata_linkedin_fixed.py'):
        print("❌ Required file missing: src/brightdata_linkedin_fixed.py")
        print("Make sure you're running from the project root directory")
        return 1

    # Run the complete example
    success = working_snapshot_example()

    if not success:
        print(f"\n🔧 Running quick single profile test for debugging...")
        quick_success = quick_single_profile_test()
        if quick_success:
            print(f"\n💡 Single profile works - issue might be with batch processing")
        else:
            print(f"\n💡 Check API key and network connection")

    print(f"\n" + "="*60)
    if success:
        print("🎉 WORKING EXAMPLE COMPLETED SUCCESSFULLY!")
        print("The fixed Bright Data snapshot workflow is functioning properly.")
        print("\nNext steps:")
        print("1. Use the fixed service in your Streamlit app")
        print("2. Check the saved JSON file for detailed results")
        print("3. Integrate with your fraud detection workflow")
    else:
        print("⚠️  EXAMPLE COMPLETED WITH ISSUES")
        print("Check the error messages above for debugging information.")
        print("\nTroubleshooting:")
        print("1. Verify BRIGHTDATA_API_KEY is correct")
        print("2. Check your internet connection")
        print("3. Try with a single profile first")
        print("4. Check Bright Data service status")

    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
