#!/usr/bin/env python3
"""
Bright Data LinkedIn API - Curl Command Equivalent Demo

This script demonstrates how to replicate the exact curl command functionality
for Bright Data LinkedIn API using Python requests.

Original curl command:
curl -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     -d '[{"url":"https://www.linkedin.com/in/elad-moshe-05a90413/"},
          {"url":"https://www.linkedin.com/in/jonathan-myrvik-3baa01109"},
          {"url":"https://www.linkedin.com/in/aviv-tal-75b81/"},
          {"url":"https://www.linkedin.com/in/bulentakar/"}]' \
     "https://api.brightdata.com/datasets/v3/trigger?dataset_id=gd_l1viktl72bvl7bjuj0&include_errors=true"

Requirements:
- Set BRIGHTDATA_API_KEY environment variable
- Install requests: pip install requests

Usage:
    python examples/brightdata_curl_demo.py
"""

import os
import json
import requests
import time
from datetime import datetime
from typing import Dict, List, Any, Optional


class BrightDataCurlDemo:
    """Direct API demonstration matching curl functionality with async workflow"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize with API key from environment or parameter"""
        self.api_key = api_key or os.getenv('BRIGHTDATA_API_KEY')

        if not self.api_key:
            raise ValueError(
                "API key is required. Set BRIGHTDATA_API_KEY environment variable "
                "or pass api_key parameter"
            )

        # API configuration - matching the curl command exactly
        self.base_url = "https://api.brightdata.com/datasets/v3"
        self.dataset_id = "gd_l1viktl72bvl7bjuj0"
        self.include_errors = True

        # Async workflow configuration
        self.polling_interval = 5  # seconds
        self.max_polling_time = 300  # 5 minutes

        # Headers exactly as in curl command
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

    def execute_curl_equivalent(self, urls: List[str]) -> Dict[str, Any]:
        """
        Execute the complete async workflow: trigger -> poll -> retrieve

        Args:
            urls: List of LinkedIn profile URLs

        Returns:
            Complete workflow results
        """
        print("🚀 Executing Bright Data Async Workflow")
        print("=" * 60)

        total_start_time = time.time()

        try:
            # Step 1: Trigger snapshot creation (original curl equivalent)
            print("Step 1: Creating snapshot...")
            snapshot_result = self._trigger_snapshot(urls)

            if not snapshot_result['success']:
                return snapshot_result

            snapshot_id = snapshot_result['snapshot_id']
            print(f"✅ Snapshot created: {snapshot_id}")

            # Step 2: Poll for completion
            print(f"\nStep 2: Polling for completion...")
            poll_result = self._poll_snapshot_progress(snapshot_id)

            if not poll_result['success']:
                return poll_result

            # Step 3: Retrieve data
            print(f"\nStep 3: Retrieving snapshot data...")
            data_result = self._get_snapshot_data(snapshot_id)

            total_execution_time = time.time() - total_start_time

            if data_result['success']:
                return {
                    'success': True,
                    'snapshot_id': snapshot_id,
                    'total_execution_time': total_execution_time,
                    'trigger_result': snapshot_result,
                    'poll_result': poll_result,
                    'data_result': data_result,
                    'final_data': data_result['data']
                }
            else:
                return data_result

        except Exception as e:
            total_execution_time = time.time() - total_start_time
            return {
                'success': False,
                'error': str(e),
                'total_execution_time': total_execution_time
            }

    def _trigger_snapshot(self, urls: List[str]) -> Dict[str, Any]:
        """Step 1: Trigger snapshot creation (original curl command)"""
        payload = [{"url": url} for url in urls]
        endpoint = f"{self.base_url}/trigger"
        params = {
            'dataset_id': self.dataset_id,
            'include_errors': str(self.include_errors).lower()
        }

        print(f"📍 Trigger Endpoint: {endpoint}")
        print(f"📋 Parameters: {params}")
        print(f"📦 Payload: {len(payload)} URLs")

        start_time = time.time()

        try:
            response = requests.post(
                endpoint,
                headers=self.headers,
                json=payload,
                params=params,
                timeout=60
            )

            execution_time = time.time() - start_time
            print(f"⏱️  Trigger time: {execution_time:.2f} seconds")
            print(f"📊 Status code: {response.status_code}")

            if response.status_code == 200:
                response_data = response.json()
                snapshot_id = response_data.get('snapshot_id') or response_data.get('id') or response_data.get('request_id')

                if snapshot_id:
                    return {
                        'success': True,
                        'snapshot_id': snapshot_id,
                        'execution_time': execution_time,
                        'data': response_data
                    }
                else:
                    return {
                        'success': False,
                        'error': 'No snapshot ID in response',
                        'execution_time': execution_time,
                        'raw_response': response.text
                    }
            else:
                return {
                    'success': False,
                    'status_code': response.status_code,
                    'execution_time': execution_time,
                    'error': response.text
                }

        except requests.exceptions.RequestException as e:
            execution_time = time.time() - start_time
            return {
                'success': False,
                'execution_time': execution_time,
                'error': str(e)
            }

    def _poll_snapshot_progress(self, snapshot_id: str) -> Dict[str, Any]:
        """Step 2: Poll snapshot progress until completion"""
        print(f"📊 Polling snapshot: {snapshot_id}")

        start_time = time.time()
        attempts = 0
        max_attempts = self.max_polling_time // self.polling_interval

        while attempts < max_attempts:
            try:
                # Progress endpoint - second curl command equivalent
                endpoint = f"{self.base_url}/progress/{snapshot_id}"

                response = requests.get(
                    endpoint,
                    headers={'Authorization': self.headers['Authorization']},  # Only auth header needed
                    timeout=30
                )

                if response.status_code == 200:
                    progress_data = response.json()
                    status = progress_data.get('status', 'unknown').lower()
                    progress_percent = float(progress_data.get('progress', progress_data.get('progress_percent', 0)))

                    elapsed = time.time() - start_time
                    print(f"   Status: {status} ({progress_percent:.1f}%) - {elapsed:.0f}s elapsed")

                    if status in ['completed', 'done', 'finished']:
                        return {
                            'success': True,
                            'final_status': status,
                            'progress_percent': progress_percent,
                            'polling_time': elapsed,
                            'attempts': attempts + 1,
                            'progress_data': progress_data
                        }
                    elif status in ['failed', 'error', 'canceled', 'cancelled']:
                        return {
                            'success': False,
                            'error': f"Snapshot {status}",
                            'final_status': status,
                            'polling_time': elapsed,
                            'progress_data': progress_data
                        }

                    # Continue polling
                    time.sleep(self.polling_interval)
                    attempts += 1

                else:
                    print(f"   Progress check failed: {response.status_code}")
                    time.sleep(self.polling_interval)
                    attempts += 1

            except Exception as e:
                print(f"   Progress check error: {e}")
                time.sleep(self.polling_interval)
                attempts += 1

        # Timeout
        return {
            'success': False,
            'error': f'Polling timeout after {time.time() - start_time:.0f} seconds',
            'polling_time': time.time() - start_time,
            'attempts': attempts
        }

    def _get_snapshot_data(self, snapshot_id: str) -> Dict[str, Any]:
        """Step 3: Retrieve final snapshot data (third curl command equivalent)"""
        endpoint = f"{self.base_url}/snapshot/{snapshot_id}"
        params = {'format': 'json'}

        print(f"📥 Data Endpoint: {endpoint}")
        print(f"📋 Params: {params}")

        start_time = time.time()

        try:
            response = requests.get(
                endpoint,
                headers={'Authorization': self.headers['Authorization']},
                params=params,
                timeout=60
            )

            execution_time = time.time() - start_time
            print(f"⏱️  Data retrieval time: {execution_time:.2f} seconds")
            print(f"📊 Status code: {response.status_code}")

            if response.status_code == 200:
                response_data = response.json()
                return {
                    'success': True,
                    'execution_time': execution_time,
                    'data': response_data,
                    'data_size': len(response.text)
                }
            else:
                return {
                    'success': False,
                    'status_code': response.status_code,
                    'execution_time': execution_time,
                    'error': response.text
                }

        except requests.exceptions.RequestException as e:
            execution_time = time.time() - start_time
            return {
                'success': False,
                'execution_time': execution_time,
                'error': str(e)
            }

    def print_response_analysis(self, result: Dict[str, Any]) -> None:
        """Print detailed analysis of the complete workflow"""
        print("\n" + "=" * 60)
        print("📊 WORKFLOW ANALYSIS")
        print("=" * 60)

        if result['success']:
            print("✅ Status: SUCCESS")
            print(f"📸 Snapshot ID: {result['snapshot_id']}")
            print(f"⏱️  Total time: {result['total_execution_time']:.2f} seconds")

            # Trigger phase
            trigger = result['trigger_result']
            print(f"\n🚀 Step 1 - Trigger: {trigger['execution_time']:.2f}s")

            # Polling phase
            poll = result['poll_result']
            print(f"📊 Step 2 - Polling: {poll['polling_time']:.2f}s ({poll['attempts']} attempts)")
            print(f"   Final status: {poll['final_status']}")
            print(f"   Progress: {poll.get('progress_percent', 0):.1f}%")

            # Data retrieval phase
            data_result = result['data_result']
            print(f"📥 Step 3 - Data retrieval: {data_result['execution_time']:.2f}s")
            print(f"   Data size: {data_result.get('data_size', 0):,} bytes")

            # Analyze final data
            final_data = result['final_data']
            if isinstance(final_data, list):
                print(f"\n📋 Final data: {len(final_data)} profiles")
                successful_profiles = [item for item in final_data if not item.get('error')]
                print(f"   Successful: {len(successful_profiles)}")
                print(f"   Failed: {len(final_data) - len(successful_profiles)}")

                if successful_profiles:
                    print("\n🔍 Sample profile data:")
                    sample = successful_profiles[0]
                    for key, value in list(sample.items())[:8]:  # First 8 keys
                        value_preview = str(value)[:50] if value else 'None'
                        if len(str(value)) > 50:
                            value_preview += "..."
                        print(f"   • {key}: {value_preview}")

            elif isinstance(final_data, dict):
                print(f"\n📦 Final data structure: {list(final_data.keys())}")

        else:
            print("❌ Status: FAILED")
            print(f"🔥 Error: {result['error']}")
            if 'total_execution_time' in result:
                print(f"⏱️  Total time before failure: {result['total_execution_time']:.2f} seconds")

    def save_response(self, result: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Save response to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"brightdata_response_{timestamp}.json"

        # Prepare data for JSON serialization
        save_data = {
            'timestamp': datetime.now().isoformat(),
            'success': result['success'],
            'status_code': result['status_code'],
            'execution_time': result['execution_time']
        }

        if result['success']:
            save_data['data'] = result['data']
        else:
            save_data['error'] = result['error']

        save_data['raw_response'] = result['raw_response']

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        print(f"💾 Response saved to: {filename}")
        return filename


def main():
    """Main demonstration function"""
    print("🔗 BRIGHT DATA LINKEDIN API - CURL EQUIVALENT DEMO")
    print("=" * 60)

    # The exact URLs from the original curl command
    linkedin_urls = [
        "https://www.linkedin.com/in/elad-moshe-05a90413/",
        "https://www.linkedin.com/in/jonathan-myrvik-3baa01109",
        "https://www.linkedin.com/in/aviv-tal-75b81/",
        "https://www.linkedin.com/in/bulentakar/"
    ]

    print("📋 LinkedIn URLs to process:")
    for i, url in enumerate(linkedin_urls, 1):
        print(f"   {i}. {url}")

    # Check API key
    api_key = os.getenv('BRIGHTDATA_API_KEY')
    if not api_key:
        print("\n❌ ERROR: BRIGHTDATA_API_KEY environment variable not set!")
        print("\nTo use this demo:")
        print("1. Get your Bright Data API key")
        print("2. Set the environment variable:")
        print("   export BRIGHTDATA_API_KEY='your_api_key_here'")
        print("3. Run this script again")
        return 1

    try:
        # Initialize demo
        demo = BrightDataCurlDemo(api_key)

        print(f"\n✅ API key configured: {api_key[:10]}...{api_key[-4:]}")

        # Execute the curl equivalent
        result = demo.execute_curl_equivalent(linkedin_urls)

        # Analyze response
        demo.print_response_analysis(result)

        # Save response to file
        saved_file = demo.save_response(result)

        print("\n" + "=" * 60)
        print("🎯 CURL COMMANDS COMPARISON")
        print("=" * 60)

        print("Complete workflow - 3 curl commands:")

        print("\n1️⃣ Trigger snapshot:")
        print(f"""curl -H "Authorization: Bearer {api_key}" \\
     -H "Content-Type: application/json" \\
     -d '{json.dumps([{"url": url} for url in linkedin_urls])}' \\
     "https://api.brightdata.com/datasets/v3/trigger?dataset_id=gd_l1viktl72bvl7bjuj0&include_errors=true"
""")

        if result['success']:
            snapshot_id = result['snapshot_id']
            print(f"\n2️⃣ Check progress:")
            print(f"""curl -H "Authorization: Bearer {api_key}" \\
     "https://api.brightdata.com/datasets/v3/progress/{snapshot_id}"
""")

            print(f"\n3️⃣ Get final data:")
            print(f"""curl -H "Authorization: Bearer {api_key}" \\
     "https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}?format=json"
""")

        print("\nPython equivalent workflow just executed ✅")

        # Show practical usage tips
        print("\n" + "=" * 60)
        print("💡 PRACTICAL USAGE TIPS")
        print("=" * 60)

        print("1. 🔄 For polling results (if async):")
        print("   - Check if response contains a job/request ID")
        print("   - Use status endpoint to check completion")
        print("   - Retrieve final results when ready")

        print("\n2. 📊 For batch processing:")
        print("   - Process URLs in batches of 10-50")
        print("   - Implement rate limiting")
        print("   - Handle partial failures gracefully")

        print("\n3. 🛡️  For error handling:")
        print("   - Always check response status codes")
        print("   - Implement retry logic for transient failures")
        print("   - Log all requests for debugging")

        if result['success']:
            print("\n🎉 Demo completed successfully!")
            return 0
        else:
            print(f"\n⚠️  Demo completed with errors: {result['error']}")
            return 1

    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
