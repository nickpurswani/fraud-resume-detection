"""
Bright Data LinkedIn Service - Fixed Implementation
A robust implementation of the Bright Data snapshot workflow for LinkedIn profile extraction.

This implementation handles the complete async workflow:
1. Trigger snapshot creation
2. Poll for completion with proper error handling
3. Retrieve and parse final data

Author: AI Assistant
Date: December 2024
"""

import os
import json
import time
import requests
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)

class SnapshotStatus(Enum):
    """Status enumeration for snapshots"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    UNKNOWN = "unknown"

@dataclass
class ProfileData:
    """LinkedIn profile data structure"""
    url: str
    name: Optional[str] = None
    headline: Optional[str] = None
    title: Optional[str] = None  # Current job title/position
    current_company: Optional[str] = None  # Current company name
    location: Optional[str] = None
    summary: Optional[str] = None
    experience: List[Dict] = None
    education: List[Dict] = None
    skills: List[str] = None
    connections: Optional[int] = None
    success: bool = False
    error: Optional[str] = None
    raw_data: Optional[Dict] = None

@dataclass
class SnapshotResult:
    """Complete snapshot workflow result"""
    snapshot_id: str
    status: SnapshotStatus
    profiles: List[ProfileData]
    total_time: float
    success: bool = False
    error: Optional[str] = None
    metadata: Dict = None

class BrightDataLinkedInFixed:
    """Fixed implementation of Bright Data LinkedIn service"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the service"""
        self.api_key = api_key or os.getenv('BRIGHTDATA_API_KEY')
        if not self.api_key:
            raise ValueError("BRIGHTDATA_API_KEY is required")

        # API configuration
        self.base_url = "https://api.brightdata.com/datasets/v3"
        self.dataset_id = "gd_l1viktl72bvl7bjuj0"

        # Headers for all requests
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

        # Polling configuration
        self.poll_interval = 10  # seconds
        self.max_poll_time = 300  # 5 minutes
        self.request_timeout = 30  # 30 seconds per request

        logger.info(f"BrightData service initialized with dataset {self.dataset_id}")

    def extract_profiles(self, linkedin_urls: List[str],
                        progress_callback=None) -> SnapshotResult:
        """
        Extract LinkedIn profiles using complete snapshot workflow

        Args:
            linkedin_urls: List of LinkedIn profile URLs
            progress_callback: Optional callback function for progress updates

        Returns:
            SnapshotResult with extracted profiles
        """
        start_time = time.time()

        try:
            logger.info(f"Starting profile extraction for {len(linkedin_urls)} URLs")

            # Step 1: Validate URLs
            valid_urls = self._validate_urls(linkedin_urls)
            if not valid_urls:
                return SnapshotResult(
                    snapshot_id="",
                    status=SnapshotStatus.FAILED,
                    profiles=[],
                    total_time=time.time() - start_time,
                    error="No valid LinkedIn URLs provided"
                )

            # Step 2: Create snapshot
            if progress_callback:
                progress_callback("Creating snapshot...", 10)

            snapshot_id = self._create_snapshot(valid_urls)
            if not snapshot_id:
                return SnapshotResult(
                    snapshot_id="",
                    status=SnapshotStatus.FAILED,
                    profiles=[],
                    total_time=time.time() - start_time,
                    error="Failed to create snapshot"
                )

            logger.info(f"Snapshot created: {snapshot_id}")

            # Step 3: Try to retrieve data immediately with retries (don't wait for completion)
            if progress_callback:
                progress_callback(f"Attempting to retrieve data for snapshot {snapshot_id}...", 30)

            # Give the snapshot a moment to start processing
            time.sleep(5)

            snapshot_data = self._get_snapshot_data_with_retries(snapshot_id, progress_callback, max_retries=8)
            if not snapshot_data:
                return SnapshotResult(
                    snapshot_id=snapshot_id,
                    status=SnapshotStatus.FAILED,
                    profiles=[],
                    total_time=time.time() - start_time,
                    error="Failed to retrieve snapshot data after 8 attempts"
                )

            # Step 4: Parse profiles
            profiles = self._parse_profiles(snapshot_data, valid_urls)

            total_time = time.time() - start_time
            logger.info(f"Profile extraction completed in {total_time:.2f}s")

            if progress_callback:
                progress_callback("Complete!", 100)

            return SnapshotResult(
                snapshot_id=snapshot_id,
                status=SnapshotStatus.COMPLETED,
                profiles=profiles,
                total_time=total_time,
                success=True,
                metadata={
                    'total_urls': len(linkedin_urls),
                    'valid_urls': len(valid_urls),
                    'successful_profiles': len([p for p in profiles if p.success]),
                    'failed_profiles': len([p for p in profiles if not p.success])
                }
            )

        except Exception as e:
            logger.error(f"Profile extraction failed: {e}", exc_info=True)
            return SnapshotResult(
                snapshot_id="",
                status=SnapshotStatus.FAILED,
                profiles=[],
                total_time=time.time() - start_time,
                error=str(e)
            )

    def extract_single_profile(self, linkedin_url: str,
                              progress_callback=None) -> ProfileData:
        """Extract a single LinkedIn profile"""
        result = self.extract_profiles([linkedin_url], progress_callback)

        if result.success and result.profiles:
            return result.profiles[0]
        else:
            return ProfileData(
                url=linkedin_url,
                success=False,
                error=result.error or "Unknown extraction error"
            )

    def _validate_urls(self, urls: List[str]) -> List[str]:
        """Validate LinkedIn URLs"""
        valid_urls = []

        for url in urls:
            try:
                # Basic validation
                if not url or not isinstance(url, str):
                    continue

                # Ensure it's a URL
                if not url.startswith('http'):
                    url = 'https://' + url

                # Check if it contains linkedin.com
                if 'linkedin.com' not in url.lower():
                    logger.warning(f"Not a LinkedIn URL: {url}")
                    continue

                # Check if it's a profile URL
                if '/in/' not in url and '/pub/' not in url:
                    logger.warning(f"Not a LinkedIn profile URL: {url}")
                    continue

                valid_urls.append(url)

            except Exception as e:
                logger.warning(f"URL validation error for {url}: {e}")
                continue

        logger.info(f"Validated {len(valid_urls)} out of {len(urls)} URLs")
        return valid_urls

    def _create_snapshot(self, urls: List[str]) -> Optional[str]:
        """Create snapshot and return snapshot ID"""
        try:
            # Prepare payload
            payload = [{"url": url} for url in urls]

            # API endpoint
            endpoint = f"{self.base_url}/trigger"
            params = {
                'dataset_id': self.dataset_id,
                'include_errors': 'true'
            }

            logger.info(f"Creating snapshot with {len(urls)} URLs")
            logger.debug(f"Trigger endpoint: {endpoint}")
            logger.debug(f"Payload: {json.dumps(payload, indent=2)}")

            # Make request
            response = requests.post(
                endpoint,
                headers=self.headers,
                json=payload,
                params=params,
                timeout=self.request_timeout
            )

            logger.info(f"Trigger response status: {response.status_code}")
            logger.debug(f"Trigger response: {response.text}")

            if response.status_code != 200:
                logger.error(f"Trigger request failed: {response.status_code} - {response.text}")
                return None

            # Parse response
            response_data = response.json()

            # Extract snapshot ID - try multiple possible field names
            snapshot_id = (
                response_data.get('snapshot_id') or
                response_data.get('id') or
                response_data.get('request_id') or
                response_data.get('job_id')
            )

            if snapshot_id:
                logger.info(f"Snapshot created successfully: {snapshot_id}")
                return snapshot_id
            else:
                logger.error(f"No snapshot ID in response: {response_data}")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error creating snapshot: {e}")
            return None
        except Exception as e:
            logger.error(f"Error creating snapshot: {e}")
            return None

    def _poll_until_complete(self, snapshot_id: str,
                           progress_callback=None) -> SnapshotStatus:
        """Poll snapshot until completion or timeout"""
        start_time = time.time()
        attempt = 0
        last_progress = 0

        logger.info(f"Starting to poll snapshot {snapshot_id}")

        while time.time() - start_time < self.max_poll_time:
            attempt += 1
            elapsed = time.time() - start_time

            try:
                # Get current status
                status_data = self._get_snapshot_status(snapshot_id)
                if not status_data:
                    logger.warning(f"Failed to get status for snapshot {snapshot_id} (attempt {attempt})")
                    time.sleep(self.poll_interval)
                    continue

                # Parse status
                status = self._parse_snapshot_status(status_data)
                progress = self._extract_progress(status_data)

                logger.info(f"Snapshot {snapshot_id} - Status: {status.value}, Progress: {progress:.1f}%, Elapsed: {elapsed:.0f}s")

                # Update progress callback
                if progress_callback and progress != last_progress:
                    progress_callback(
                        f"Snapshot {status.value}: {progress:.1f}% complete",
                        20 + int(progress * 0.7)  # Scale to 20-90%
                    )
                    last_progress = progress

                # Check if completed
                if status == SnapshotStatus.COMPLETED:
                    logger.info(f"Snapshot {snapshot_id} completed successfully after {elapsed:.0f}s")
                    return status
                elif status == SnapshotStatus.FAILED:
                    error_msg = status_data.get('error', 'Unknown error')
                    logger.error(f"Snapshot {snapshot_id} failed: {error_msg}")
                    return status
                elif status == SnapshotStatus.CANCELED:
                    logger.error(f"Snapshot {snapshot_id} was canceled")
                    return status

                # Continue polling
                time.sleep(self.poll_interval)

            except Exception as e:
                logger.warning(f"Polling error for snapshot {snapshot_id} (attempt {attempt}): {e}")
                time.sleep(self.poll_interval)

        # Timeout reached
        logger.error(f"Polling timeout for snapshot {snapshot_id} after {time.time() - start_time:.0f}s")
        return SnapshotStatus.FAILED

    def _get_snapshot_status(self, snapshot_id: str) -> Optional[Dict]:
        """Get current snapshot status"""
        try:
            endpoint = f"{self.base_url}/progress/{snapshot_id}"

            response = requests.get(
                endpoint,
                headers={'Authorization': self.headers['Authorization']},
                timeout=self.request_timeout
            )

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                logger.warning(f"Snapshot {snapshot_id} not found (may be too new)")
                return None
            else:
                logger.warning(f"Status request failed: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.warning(f"Error getting snapshot status: {e}")
            return None

    def _parse_snapshot_status(self, status_data: Dict) -> SnapshotStatus:
        """Parse snapshot status from API response"""
        if not status_data:
            return SnapshotStatus.UNKNOWN

        status_str = str(status_data.get('status', '')).lower().strip()

        # Map various status strings
        if status_str in ['completed', 'complete', 'done', 'finished', 'success']:
            return SnapshotStatus.COMPLETED
        elif status_str in ['failed', 'fail', 'error', 'failed_with_errors']:
            return SnapshotStatus.FAILED
        elif status_str in ['canceled', 'cancelled', 'aborted']:
            return SnapshotStatus.CANCELED
        elif status_str in ['running', 'processing', 'in_progress', 'active']:
            return SnapshotStatus.RUNNING
        elif status_str in ['pending', 'queued', 'waiting']:
            return SnapshotStatus.PENDING
        else:
            # Default to running if we don't recognize the status
            return SnapshotStatus.RUNNING

    def _extract_progress(self, status_data: Dict) -> float:
        """Extract progress percentage from status data"""
        if not status_data:
            return 0.0

        # Try multiple field names
        for field in ['progress', 'progress_percent', 'completion', 'percent']:
            if field in status_data:
                try:
                    value = float(status_data[field])
                    # Handle both 0-1 and 0-100 formats
                    if value <= 1.0:
                        return value * 100
                    return min(value, 100.0)
                except (ValueError, TypeError):
                    continue

        # Try to calculate from record counts
        total = status_data.get('total_records') or status_data.get('total')
        completed = status_data.get('completed_records') or status_data.get('completed')

        if total and completed and total > 0:
            return (completed / total) * 100

        return 0.0

    def _get_snapshot_data(self, snapshot_id: str) -> Optional[Any]:
        """Retrieve final snapshot data"""
        try:
            endpoint = f"{self.base_url}/snapshot/{snapshot_id}"
            params = {'format': 'json'}

            logger.info(f"Retrieving data for snapshot {snapshot_id}")

            response = requests.get(
                endpoint,
                headers={'Authorization': self.headers['Authorization']},
                params=params,
                timeout=60  # Longer timeout for data retrieval
            )

            logger.info(f"Data retrieval response status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                logger.info(f"Retrieved snapshot data: {type(data)}")
                if isinstance(data, list):
                    logger.info(f"Data contains {len(data)} items")
                elif isinstance(data, dict):
                    logger.info(f"Data keys: {list(data.keys())}")
                return data
            else:
                logger.error(f"Data retrieval failed: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"Error retrieving snapshot data: {e}")
            return None

    def _get_snapshot_data_with_retries(self, snapshot_id: str, progress_callback=None, max_retries: int = 8) -> Optional[Any]:
        """Retrieve snapshot data with aggressive retry logic and smart backoff"""
        wait_times = [0, 5, 10, 15, 30, 45, 60, 90]  # Progressive backoff in seconds

        for attempt in range(1, max_retries + 1):
            try:
                if progress_callback:
                    progress_pct = 30 + int((attempt / max_retries) * 60)  # 30-90%
                    progress_callback(f"Retrieving data (attempt {attempt}/{max_retries})...", progress_pct)

                logger.info(f"Attempting to retrieve snapshot data (attempt {attempt}/{max_retries})")

                # Wait before attempt (except first one)
                if attempt > 1:
                    wait_time = wait_times[attempt - 1] if attempt <= len(wait_times) else 120
                    logger.info(f"Waiting {wait_time}s before attempt {attempt}...")
                    time.sleep(wait_time)

                snapshot_data = self._get_snapshot_data(snapshot_id)

                if snapshot_data:
                    # Check if we have meaningful data
                    if isinstance(snapshot_data, list):
                        if len(snapshot_data) > 0:
                            logger.info(f"Success! Retrieved {len(snapshot_data)} profiles on attempt {attempt}")
                            return snapshot_data
                        else:
                            logger.info(f"Retrieved empty list on attempt {attempt}")
                    elif isinstance(snapshot_data, dict):
                        if snapshot_data:  # Non-empty dict
                            logger.info(f"Success! Retrieved profile data on attempt {attempt}")
                            return snapshot_data
                        else:
                            logger.info(f"Retrieved empty dict on attempt {attempt}")
                    else:
                        logger.info(f"Retrieved unexpected data type: {type(snapshot_data)} on attempt {attempt}")
                else:
                    logger.info(f"No data retrieved on attempt {attempt}")

                # For early attempts, also check snapshot status to log progress
                if attempt <= 3:
                    try:
                        status_data = self._get_snapshot_status(snapshot_id)
                        if status_data:
                            status = self._parse_snapshot_status(status_data)
                            progress = self._extract_progress(status_data)
                            logger.info(f"Current snapshot status: {status.value}, progress: {progress:.1f}%")
                    except:
                        pass  # Don't let status check fail the data retrieval

            except Exception as e:
                logger.warning(f"Attempt {attempt} failed with error: {e}")

        logger.error(f"Failed to retrieve data after {max_retries} attempts")
        return None

    def _quick_data_check(self, snapshot_id: str) -> bool:
        """Quick check to see if snapshot data is available right now"""
        try:
            logger.info(f"Quick data check for snapshot {snapshot_id}")
            snapshot_data = self._get_snapshot_data(snapshot_id)

            if snapshot_data:
                if isinstance(snapshot_data, list) and len(snapshot_data) > 0:
                    logger.info(f"Quick check: Found {len(snapshot_data)} items")
                    return True
                elif isinstance(snapshot_data, dict) and snapshot_data:
                    logger.info(f"Quick check: Found data dict with keys: {list(snapshot_data.keys())}")
                    return True

            logger.info("Quick check: No meaningful data found yet")
            return False

        except Exception as e:
            logger.warning(f"Quick data check failed: {e}")
            return False

    def _parse_profiles(self, snapshot_data: Any, original_urls: List[str]) -> List[ProfileData]:
        """Parse snapshot data into ProfileData objects"""
        profiles = []

        try:
            # Handle different response formats
            if isinstance(snapshot_data, list):
                data_items = snapshot_data
            elif isinstance(snapshot_data, dict):
                # Try different possible keys
                data_items = (
                    snapshot_data.get('data') or
                    snapshot_data.get('items') or
                    snapshot_data.get('results') or
                    []
                )
                if not isinstance(data_items, list):
                    data_items = [snapshot_data]  # Single item response
            else:
                logger.warning(f"Unexpected snapshot data format: {type(snapshot_data)}")
                data_items = []

            logger.info(f"Parsing {len(data_items)} profile data items")

            # Parse each profile
            for i, item in enumerate(data_items):
                try:
                    profile = self._parse_single_profile(item, original_urls)
                    profiles.append(profile)
                except Exception as e:
                    logger.error(f"Error parsing profile {i}: {e}")
                    # Create failed profile
                    url = original_urls[i] if i < len(original_urls) else "unknown"
                    profiles.append(ProfileData(
                        url=url,
                        success=False,
                        error=f"Parsing error: {str(e)}"
                    ))

            # Ensure we have a profile for each original URL
            while len(profiles) < len(original_urls):
                url = original_urls[len(profiles)]
                profiles.append(ProfileData(
                    url=url,
                    success=False,
                    error="No data returned for this URL"
                ))

            successful = len([p for p in profiles if p.success])
            logger.info(f"Successfully parsed {successful}/{len(profiles)} profiles")

            return profiles

        except Exception as e:
            logger.error(f"Error parsing profiles: {e}")
            # Return failed profiles for all URLs
            return [
                ProfileData(url=url, success=False, error=f"Parsing failed: {str(e)}")
                for url in original_urls
            ]

    def _parse_single_profile(self, item: Dict, original_urls: List[str]) -> ProfileData:
        """Parse a single profile from API data"""
        try:
            # Get URL
            url = item.get('url') or item.get('input_url') or item.get('profile_url')
            if not url and original_urls:
                url = original_urls[0]  # Fallback to first original URL

            # Check for errors
            error_msg = item.get('error') or item.get('error_message')
            if error_msg:
                return ProfileData(url=url or "unknown", success=False, error=error_msg)

            # Extract profile data
            name = (
                item.get('name') or
                item.get('full_name') or
                item.get('firstName', '') + ' ' + item.get('lastName', '')
            ).strip()

            headline = item.get('headline') or item.get('title') or item.get('current_position')
            title = item.get('position') or item.get('title') or item.get('current_position') or headline
            current_company = (
                item.get('current_company') or
                item.get('current_company_name') or
                item.get('company')
            )
            location = item.get('location') or item.get('geo_location')
            summary = item.get('summary') or item.get('about') or item.get('description')

            # Parse experience
            experience = []
            exp_data = item.get('experience') or item.get('positions') or []
            if isinstance(exp_data, list):
                for exp in exp_data:
                    if isinstance(exp, dict):
                        experience.append({
                            'title': exp.get('title') or exp.get('position'),
                            'company': exp.get('company') or exp.get('company_name'),
                            'duration': exp.get('duration') or exp.get('dates'),
                            'location': exp.get('location'),
                            'description': exp.get('description')
                        })

            # Parse education
            education = []
            edu_data = item.get('education') or item.get('schools') or []
            if isinstance(edu_data, list):
                for edu in edu_data:
                    if isinstance(edu, dict):
                        education.append({
                            'school': edu.get('school') or edu.get('institution'),
                            'degree': edu.get('degree') or edu.get('field_of_study'),
                            'field': edu.get('field') or edu.get('field_of_study'),
                            'years': edu.get('years') or edu.get('duration'),
                            'description': edu.get('description')
                        })

            # Parse skills
            skills = []
            skills_data = item.get('skills') or []
            if isinstance(skills_data, list):
                for skill in skills_data:
                    if isinstance(skill, str):
                        skills.append(skill)
                    elif isinstance(skill, dict):
                        skills.append(skill.get('name') or skill.get('skill') or str(skill))

            # Parse connections
            connections = None
            conn_data = item.get('connections') or item.get('connectionCount') or item.get('connections_count')
            if conn_data:
                try:
                    connections = int(conn_data)
                except (ValueError, TypeError):
                    pass

            return ProfileData(
                url=url or "unknown",
                name=name or None,
                headline=headline or None,
                title=title or None,
                current_company=current_company or None,
                location=location or None,
                summary=summary or None,
                experience=experience,
                education=education,
                skills=skills,
                connections=connections,
                success=True,
                raw_data=item
            )

        except Exception as e:
            logger.error(f"Error parsing single profile: {e}")
            return ProfileData(
                url=url or "unknown",
                success=False,
                error=f"Profile parsing failed: {str(e)}",
                raw_data=item
            )

    def verify_against_resume(self, profile: ProfileData,
                            resume_data: Dict) -> Dict[str, Any]:
        """Verify LinkedIn profile against resume data"""
        if not profile.success:
            return {
                'overall_match_score': 0.0,
                'confidence_score': 0.0,
                'fraud_indicators': ['Profile extraction failed'],
                'verification_results': {},
                'error': profile.error
            }

        verification_results = {}
        scores = []

        # Name verification
        if profile.name and resume_data.get('name'):
            name_score = self._calculate_name_similarity(profile.name, resume_data['name'])
            verification_results['name_match'] = {
                'match': name_score > 0.8,
                'score': name_score,
                'linkedin_value': profile.name,
                'resume_value': resume_data['name']
            }
            scores.append(name_score)

        # Experience verification
        if profile.experience and resume_data.get('experience'):
            exp_score = self._verify_experience(profile.experience, resume_data['experience'])
            verification_results['experience_match'] = exp_score
            if exp_score.get('score') is not None:
                scores.append(exp_score['score'])

        # Education verification
        if profile.education and resume_data.get('education'):
            edu_score = self._verify_education(profile.education, resume_data['education'])
            verification_results['education_match'] = edu_score
            if edu_score.get('score') is not None:
                scores.append(edu_score['score'])

        # Skills verification
        if profile.skills and resume_data.get('skills'):
            skills_score = self._verify_skills(profile.skills, resume_data['skills'])
            verification_results['skills_match'] = skills_score
            if skills_score.get('score') is not None:
                scores.append(skills_score['score'])

        # Calculate overall scores
        overall_score = sum(scores) / len(scores) if scores else 0.0
        confidence_score = min(overall_score, 0.95)  # Cap confidence

        # Identify fraud indicators
        fraud_indicators = []
        if overall_score < 0.3:
            fraud_indicators.append('Very low overall match score')
        if verification_results.get('name_match', {}).get('score', 1.0) < 0.5:
            fraud_indicators.append('Name mismatch with LinkedIn profile')
        if verification_results.get('experience_match', {}).get('score', 1.0) < 0.2:
            fraud_indicators.append('Major experience discrepancies')

        return {
            'overall_match_score': overall_score,
            'confidence_score': confidence_score,
            'verification_results': verification_results,
            'fraud_indicators': fraud_indicators,
            'service_used': 'Bright Data Fixed',
            'profile_data': profile
        }

    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two names"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, name1.lower(), name2.lower()).ratio()

    def _verify_experience(self, linkedin_exp: List[Dict], resume_exp: List) -> Dict:
        """Verify experience match - handles both Dict and ExperienceEntry objects"""
        if not linkedin_exp or not resume_exp:
            return {'match': False, 'score': 0.0, 'details': 'Missing experience data'}

        matches = 0
        total_compared = min(len(linkedin_exp), len(resume_exp), 3)  # Compare top 3

        for li_exp in linkedin_exp[:3]:
            li_company = (li_exp.get('company') or '').lower()
            li_title = (li_exp.get('title') or '').lower()

            for res_exp in resume_exp[:3]:
                # Handle both Dict and ExperienceEntry objects
                if hasattr(res_exp, 'get'):  # Dictionary
                    res_company = (res_exp.get('company') or '').lower()
                    res_title = (res_exp.get('title') or res_exp.get('position') or '').lower()
                else:  # ExperienceEntry object
                    res_company = (getattr(res_exp, 'company', '') or '').lower()
                    res_title = (getattr(res_exp, 'title', '') or getattr(res_exp, 'position', '') or '').lower()

                if (li_company and res_company and li_company in res_company) or \
                   (li_title and res_title and li_title in res_title):
                    matches += 1
                    break

        score = matches / total_compared if total_compared > 0 else 0.0
        return {
            'match': score > 0.5,
            'score': score,
            'details': f'{matches}/{total_compared} experiences matched'
        }

    def _verify_education(self, linkedin_edu: List[Dict], resume_edu: List) -> Dict:
        """Verify education match - handles both Dict and EducationEntry objects"""
        if not linkedin_edu or not resume_edu:
            return {'match': False, 'score': 0.5, 'details': 'Missing education data'}

        matches = 0
        for li_edu in linkedin_edu:
            li_school = (li_edu.get('school') or '').lower()
            li_degree = (li_edu.get('degree') or '').lower()

            for res_edu in resume_edu:
                # Handle both Dict and EducationEntry objects
                if hasattr(res_edu, 'get'):  # Dictionary
                    res_school = (res_edu.get('school') or res_edu.get('institution') or '').lower()
                    res_degree = (res_edu.get('degree') or '').lower()
                else:  # EducationEntry object
                    res_school = (getattr(res_edu, 'institution', '') or '').lower()
                    res_degree = (getattr(res_edu, 'degree', '') or '').lower()

                if (li_school and res_school and li_school in res_school) or \
                   (li_degree and res_degree and li_degree in res_degree):
                    matches += 1
                    break

        score = matches / min(len(linkedin_edu), len(resume_edu)) if matches > 0 else 0.0
        return {
            'match': score > 0.5,
            'score': score,
            'details': f'{matches} education entries matched'
        }

    def _verify_skills(self, linkedin_skills: List[str], resume_skills: List[str]) -> Dict:
        """Verify skills match"""
        if not linkedin_skills or not resume_skills:
            return {'match': False, 'score': 0.5, 'details': 'Missing skills data'}

        li_skills_lower = {skill.lower() for skill in linkedin_skills if skill}
        res_skills_lower = {skill.lower() for skill in resume_skills if skill}

        common_skills = li_skills_lower.intersection(res_skills_lower)
        all_skills = li_skills_lower.union(res_skills_lower)

        score = len(common_skills) / len(all_skills) if all_skills else 0.0
        return {
            'match': score > 0.3,
            'score': score,
            'details': f'{len(common_skills)} common skills out of {len(all_skills)} total'
        }

    def verify_with_ai_fallback(self, profile: ProfileData, resume_data: Dict) -> Dict[str, Any]:
        """AI-powered verification fallback when parsing fails"""
        try:
            import google.generativeai as genai
            import os

            # Configure Gemini API
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                return {
                    'overall_match_score': 0.0,
                    'confidence_score': 0.0,
                    'fraud_indicators': ['AI analysis unavailable - no API key'],
                    'verification_results': {},
                    'error': 'Gemini API key not found'
                }

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-pro')

            # Prepare data for AI analysis
            linkedin_json = profile.raw_data if profile.raw_data else {}

            prompt = f"""
            Analyze this LinkedIn profile data against the resume data for potential fraud or inconsistencies.

            LinkedIn Profile JSON:
            {linkedin_json}

            Resume Data:
            {resume_data}

            Please provide a JSON response with:
            1. overall_match_score (0.0-1.0)
            2. confidence_score (0.0-1.0)
            3. fraud_indicators (list of strings)
            4. verification_results containing:
               - name_match (score and details)
               - experience_match (score and details)
               - education_match (score and details)
               - skills_match (score and details)

            Focus on detecting:
            - Name mismatches
            - Job title/company inconsistencies
            - Education discrepancies
            - Timeline inconsistencies
            - Inflated or fabricated experience

            Return only valid JSON.
            """

            response = model.generate_content(prompt)

            # Try to parse AI response as JSON
            import json
            try:
                ai_result = json.loads(response.text)
                ai_result['service_used'] = 'AI Analysis (Gemini)'
                ai_result['profile_data'] = profile
                return ai_result
            except json.JSONDecodeError:
                # If JSON parsing fails, create structured response from text
                return {
                    'overall_match_score': 0.5,
                    'confidence_score': 0.3,
                    'fraud_indicators': ['AI analysis completed but response format unclear'],
                    'verification_results': {
                        'ai_analysis': response.text[:500] + '...' if len(response.text) > 500 else response.text
                    },
                    'service_used': 'AI Analysis (Gemini)',
                    'profile_data': profile
                }

        except ImportError:
            return {
                'overall_match_score': 0.0,
                'confidence_score': 0.0,
                'fraud_indicators': ['AI analysis unavailable - Google Generative AI not installed'],
                'verification_results': {},
                'error': 'google-generativeai package not found'
            }
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return {
                'overall_match_score': 0.0,
                'confidence_score': 0.0,
                'fraud_indicators': [f'AI analysis failed: {str(e)}'],
                'verification_results': {},
                'error': str(e)
            }

    def verify_against_resume_safe(self, profile: ProfileData, resume_data: Dict) -> Dict[str, Any]:
        """Safe wrapper for verification with AI fallback"""
        try:
            # Try normal verification first
            return self.verify_against_resume(profile, resume_data)
        except Exception as e:
            logger.error(f"Standard verification failed: {e}, falling back to AI analysis")
            # Fall back to AI analysis
            result = self.verify_with_ai_fallback(profile, resume_data)
            result['fallback_reason'] = f'Standard parsing failed: {str(e)}'
            return result

# Global instance will be created by the application when needed
# brightdata_service_fixed = None  # Created on demand
