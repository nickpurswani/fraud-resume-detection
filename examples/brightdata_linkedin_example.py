#!/usr/bin/env python3
"""
Bright Data LinkedIn Service Example Usage

This script demonstrates how to use the Bright Data LinkedIn service
for extracting and verifying LinkedIn profile data in the fraud detection tool.

Requirements:
- Set BRIGHTDATA_API_KEY environment variable
- Install required dependencies: requests

Usage:
    python examples/brightdata_linkedin_example.py
"""

import os
import sys
import json
from datetime import datetime

# Add src to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from brightdata_linkedin import BrightDataLinkedInService, LinkedInProfileData
    from config import Config
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def main():
    """Main example function"""
    print("🔗 Bright Data LinkedIn Service Example")
    print("=" * 50)

    # Check if API key is configured
    if not Config.BRIGHTDATA_API_KEY:
        print("❌ Error: BRIGHTDATA_API_KEY environment variable not set")
        print("\nTo use this example:")
        print("1. Set your Bright Data API key:")
        print("   export BRIGHTDATA_API_KEY='your_api_key_here'")
        print("2. Run the script again")
        return

    try:
        # Initialize the service
        print("🚀 Initializing Bright Data LinkedIn Service...")
        service = BrightDataLinkedInService()

        # Show service status
        status = service.get_service_status()
        print("✅ Service Status:")
        for key, value in status.items():
            print(f"   {key}: {value}")

        print("\n" + "=" * 50)

        # Example 1: Async Snapshot Workflow Demo
        print("📋 Example 1: Complete Async Workflow Demo")
        print("-" * 30)

        # Using the example URL from the prompt
        single_url = "https://www.linkedin.com/in/elad-moshe-05a90413/"
        print(f"Demonstrating async workflow with: {single_url}")

        print("\n🚀 Step 1: Creating snapshot...")
        snapshot_id = service._trigger_snapshot([single_url])

        if snapshot_id:
            print(f"✅ Snapshot created: {snapshot_id}")

            print(f"\n⏳ Step 2: Polling for completion...")
            snapshot_info = service._poll_snapshot_progress(snapshot_id)

            print(f"📊 Final status: {snapshot_info.status.value}")
            print(f"📈 Progress: {snapshot_info.progress_percent:.1f}%")

            if snapshot_info.status.value == 'completed':
                print(f"\n📥 Step 3: Retrieving data...")
                try:
                    snapshot_data = service._get_snapshot_data(snapshot_id)
                    print("✅ Data retrieved successfully!")

                    # Parse and display
                    response = service._parse_snapshot_response(
                        snapshot_data, [single_url], 0, snapshot_id, snapshot_info
                    )

                    if response.profiles and response.profiles[0].success:
                        print("\n📋 Profile Summary:")
                        print_profile_summary(response.profiles[0])
                    else:
                        print("❌ No successful profile extraction")

                except Exception as e:
                    print(f"❌ Failed to retrieve data: {e}")
            else:
                print(f"❌ Snapshot not completed: {snapshot_info.error_message}")
        else:
            print("❌ Failed to create snapshot")

        print("\n" + "=" * 50)

        # Example 2: Batch Profile Extraction (Full Async Workflow)
        print("📋 Example 2: Batch Async Extraction")
        print("-" * 30)

        # Using the example URLs from the prompt
        batch_urls = [
            "https://www.linkedin.com/in/elad-moshe-05a90413/",
            "https://www.linkedin.com/in/jonathan-myrvik-3baa01109",
            "https://www.linkedin.com/in/aviv-tal-75b81/",
            "https://www.linkedin.com/in/bulentakar/"
        ]

        print(f"Processing {len(batch_urls)} profiles via async workflow...")

        # Use the complete async workflow
        response = service.extract_profiles(batch_urls)

        print(f"✅ Complete workflow finished!")
        print(f"   Snapshot ID: {response.snapshot_id or 'N/A'}")
        print(f"   Request ID: {response.request_id}")
        print(f"   Status: {response.status.value}")
        print(f"   Total profiles: {response.total_profiles}")
        print(f"   Successful: {response.successful_extractions}")
        print(f"   Failed: {response.failed_extractions}")
        print(f"   Total execution time: {response.execution_time:.2f}s")

        if response.snapshot_info:
            print(f"\n📸 Snapshot Details:")
            print(f"   Progress: {response.snapshot_info.progress_percent:.1f}%")
            print(f"   Records: {response.snapshot_info.completed_records or 0}/{response.snapshot_info.total_records or 0}")
            if response.snapshot_info.completed_at:
                print(f"   Completed: {response.snapshot_info.completed_at}")

        if response.errors:
            print(f"\n❌ Errors: {len(response.errors)}")
            for error in response.errors[:3]:  # Show first 3 errors
                print(f"     - {error}")

        # Show successful profiles
        successful_profiles = [p for p in response.profiles if p.success]
        if successful_profiles:
            print(f"\n📊 Successfully extracted profiles:")
            for i, profile in enumerate(successful_profiles[:2], 1):  # Show first 2
                print(f"\n   Profile {i}:")
                print_profile_summary(profile, indent=6)

        print("\n" + "=" * 50)

        # Example 3: Manual Snapshot Operations
        print("📋 Example 3: Manual Snapshot Operations")
        print("-" * 30)

        if successful_profiles:
            # Demonstrate individual snapshot operations
            demo_url = batch_urls[0]
            print(f"Demonstrating manual snapshot ops with: {demo_url}")

            # Step 1: Trigger only
            print(f"\n🚀 Manual Step 1: Trigger snapshot...")
            snapshot_id = service._trigger_snapshot([demo_url])

            if snapshot_id:
                print(f"✅ Snapshot ID: {snapshot_id}")

                # Step 2: Manual polling with updates
                print(f"\n⏳ Manual Step 2: Check status...")
                current_status = service.get_snapshot_status(snapshot_id)
                print(f"   Status: {current_status.status.value}")
                print(f"   Progress: {current_status.progress_percent:.1f}%")

                # If not complete, we could poll more, but for demo just show the concept
                print(f"\n💡 In real usage, you would poll until status == 'completed'")
                print(f"   Then use _get_snapshot_data(snapshot_id) to retrieve results")
            else:
                print("❌ Failed to create snapshot")

        print("\n" + "=" * 50)

        # Example 4: Profile Verification Against Resume
        if successful_profiles:
            print("📋 Example 4: Profile Verification Against Resume")
            print("-" * 30)

            # Sample resume data for verification
            sample_resume_data = {
                'name': 'Elad Moshe',
                'location': 'Israel',
                'experience': [
                    {
                        'company': 'Microsoft',
                        'title': 'Software Engineer',
                        'duration': '2020-2023'
                    },
                    {
                        'company': 'Google',
                        'title': 'Senior Developer',
                        'duration': '2018-2020'
                    }
                ],
                'education': [
                    {
                        'school': 'Tel Aviv University',
                        'degree': 'Computer Science',
                        'year': '2018'
                    }
                ],
                'skills': [
                    'Python', 'JavaScript', 'React', 'Node.js',
                    'Machine Learning', 'Docker', 'AWS'
                ]
            }

            profile_to_verify = successful_profiles[0]
            print(f"Verifying profile: {profile_to_verify.url}")
            print(f"Against resume for: {sample_resume_data['name']}")

            verification_results = service.verify_profile_against_resume(
                profile_to_verify,
                sample_resume_data
            )

            print("\n📈 Verification Results:")
            print(f"   Overall Match Score: {verification_results['overall_match_score']:.2%}")
            print(f"   Confidence Score: {verification_results['confidence_score']:.2%}")

            print("\n   Detailed Matches:")
            for category, result in verification_results.items():
                if category.endswith('_match') and isinstance(result, dict):
                    match_icon = "✅" if result['match'] else "❌"
                    score_text = f"{result['score']:.2%}" if result['score'] is not None else "N/A"
                    print(f"     {match_icon} {category.replace('_match', '').title()}: {score_text}")
                    if result['details']:
                        print(f"        {result['details']}")

            if verification_results['fraud_indicators']:
                print(f"\n⚠️  Fraud Indicators:")
                for indicator in verification_results['fraud_indicators']:
                    print(f"     - {indicator}")
            else:
                print(f"\n✅ No fraud indicators detected")

        print("\n" + "=" * 50)

        # Example 5: Export Results to JSON
        print("📋 Example 5: Export Results to JSON")
        print("-" * 30)

        if response.profiles:
            export_data = {
                'extraction_timestamp': datetime.now().isoformat(),
                'service_info': status,
                'batch_results': {
                    'request_id': response.request_id,
                    'snapshot_id': response.snapshot_id,
                    'status': response.status.value,
                    'total_profiles': response.total_profiles,
                    'successful_extractions': response.successful_extractions,
                    'failed_extractions': response.failed_extractions,
                    'execution_time': response.execution_time
                },
                'snapshot_info': {
                    'progress_percent': response.snapshot_info.progress_percent if response.snapshot_info else None,
                    'total_records': response.snapshot_info.total_records if response.snapshot_info else None,
                    'completed_records': response.snapshot_info.completed_records if response.snapshot_info else None
                },
                'profiles': []
            }

            for profile in response.profiles:
                profile_data = {
                    'url': profile.url,
                    'success': profile.success,
                    'full_name': profile.full_name,
                    'headline': profile.headline,
                    'location': profile.location,
                    'connections_count': profile.connections_count,
                    'experience_count': len(profile.experience) if profile.experience else 0,
                    'education_count': len(profile.education) if profile.education else 0,
                    'skills_count': len(profile.skills) if profile.skills else 0,
                    'error_message': profile.error_message
                }
                export_data['profiles'].append(profile_data)

            # Save to file
            output_file = 'linkedin_extraction_results.json'
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

            print(f"✅ Results exported to: {output_file}")
            print(f"   File size: {os.path.getsize(output_file)} bytes")

        print("\n" + "=" * 50)

        # Example 6: Understanding the Async Workflow
        print("📋 Example 6: Async Workflow Summary")
        print("-" * 30)

        print("🔄 The complete Bright Data workflow involves:")
        print("   1. 🚀 Trigger: Create snapshot with LinkedIn URLs")
        print("   2. ⏳ Poll: Check progress until completion")
        print("   3. 📥 Retrieve: Get final data from completed snapshot")
        print("\n📡 This matches these curl commands:")
        print("   1. POST to /trigger → returns snapshot_id")
        print("   2. GET from /progress/{snapshot_id} → check status")
        print("   3. GET from /snapshot/{snapshot_id} → get data")
        print("\n⚡ The service handles all this automatically in extract_profiles()")

    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()

    print("\n🎉 Async workflow examples completed!")


def print_profile_summary(profile: LinkedInProfileData, indent: int = 3):
    """Print a summary of profile data"""
    indent_str = " " * indent

    print(f"{indent_str}Name: {profile.full_name or 'N/A'}")
    print(f"{indent_str}Headline: {profile.headline or 'N/A'}")
    print(f"{indent_str}Location: {profile.location or 'N/A'}")
    print(f"{indent_str}Connections: {profile.connections_count or 'N/A'}")

    if profile.experience:
        print(f"{indent_str}Experience: {len(profile.experience)} positions")
        if profile.current_position:
            current = profile.current_position
            company = current.get('company', 'N/A')
            title = current.get('title', 'N/A')
            print(f"{indent_str}Current: {title} at {company}")
    else:
        print(f"{indent_str}Experience: No data")

    if profile.education:
        print(f"{indent_str}Education: {len(profile.education)} entries")
    else:
        print(f"{indent_str}Education: No data")

    if profile.skills:
        print(f"{indent_str}Skills: {len(profile.skills)} skills")
        # Show first few skills
        skills_preview = ', '.join(profile.skills[:5])
        if len(profile.skills) > 5:
            skills_preview += f" ... (+{len(profile.skills) - 5} more)"
        print(f"{indent_str}Top skills: {skills_preview}")
    else:
        print(f"{indent_str}Skills: No data")

    print(f"{indent_str}Extraction time: {profile.extraction_timestamp}")


if __name__ == "__main__":
    main()
