"""
Bright Data LinkedIn Service for Fraudulent Candidate Detection Tool

This module provides LinkedIn profile data extraction using Bright Data's API:
- Batch processing of LinkedIn URLs
- Profile data extraction and parsing
- Employment history and education verification
- Skills and endorsements analysis
- Real-time data collection with error handling
"""

import logging
import json
import time
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import requests
from dataclasses import dataclass, asdict
import os
from enum import Enum
from urllib.parse import urlparse, urljoin

from config import Config

# Configure logging
logger = logging.getLogger(__name__)

class BrightDataStatus(Enum):
    """Status enumeration for Bright Data requests"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

class SnapshotStatus(Enum):
    """Status enumeration for snapshot progress"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"

@dataclass
class LinkedInProfileData:
    """Structure for LinkedIn profile data from Bright Data"""
    url: str
    full_name: Optional[str] = None
    headline: Optional[str] = None
    location: Optional[str] = None
    summary: Optional[str] = None
    current_position: Optional[Dict[str, Any]] = None
    experience: List[Dict[str, Any]] = None
    education: List[Dict[str, Any]] = None
    skills: List[str] = None
    connections_count: Optional[int] = None
    followers_count: Optional[int] = None
    profile_image_url: Optional[str] = None
    industry: Optional[str] = None
    about: Optional[str] = None
    languages: List[str] = None
    certifications: List[Dict[str, Any]] = None
    volunteer_experience: List[Dict[str, Any]] = None
    raw_data: Optional[Dict[str, Any]] = None
    extraction_timestamp: datetime = None
    error_message: Optional[str] = None
    success: bool = False

@dataclass
class SnapshotInfo:
    """Information about a snapshot"""
    snapshot_id: str
    status: SnapshotStatus
    progress_percent: float
    total_records: Optional[int] = None
    completed_records: Optional[int] = None
    failed_records: Optional[int] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

@dataclass
class BrightDataResponse:
    """Response structure from Bright Data API"""
    request_id: str
    snapshot_id: Optional[str]
    status: BrightDataStatus
    profiles: List[LinkedInProfileData]
    total_profiles: int
    successful_extractions: int
    failed_extractions: int
    errors: List[Dict[str, Any]]
    execution_time: float
    snapshot_info: Optional[SnapshotInfo] = None
    cost_estimate: Optional[float] = None

class BrightDataLinkedInService:
    """Service for LinkedIn data extraction using Bright Data"""

    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Bright Data LinkedIn service

        Args:
            api_key: Bright Data API key (from environment if not provided)
            config: Configuration dictionary
        """
        self.api_key = api_key or Config.BRIGHTDATA_API_KEY
        self.config = config or Config.get_brightdata_config()

        if not self.api_key:
            raise ValueError("Bright Data API key is required. Set BRIGHTDATA_API_KEY environment variable.")

        self.base_url = self.config.get('base_url', 'https://api.brightdata.com/datasets/v3')
        self.dataset_id = self.config.get('dataset_id', 'gd_l1viktl72bvl7bjuj0')
        self.timeout = self.config.get('timeout', 60)
        self.include_errors = self.config.get('include_errors', True)

        # Rate limiting
        self.rate_limit = Config.API_RATE_LIMIT.get('brightdata', 50)
        self.request_count = 0
        self.request_window_start = time.time()

        # Polling configuration
        self.polling_interval = 5  # seconds
        self.max_polling_time = 300  # 5 minutes max wait
        self.max_polling_attempts = self.max_polling_time // self.polling_interval

        # Headers for API requests
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

    def _check_rate_limit(self) -> None:
        """Check and enforce rate limiting"""
        current_time = time.time()

        # Reset counter every hour
        if current_time - self.request_window_start >= 3600:
            self.request_count = 0
            self.request_window_start = current_time

        if self.request_count >= self.rate_limit:
            sleep_time = 3600 - (current_time - self.request_window_start)
            logger.warning(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
            self.request_count = 0
            self.request_window_start = time.time()

        self.request_count += 1

    def _validate_linkedin_urls(self, urls: List[str]) -> List[str]:
        """
        Validate and clean LinkedIn URLs

        Args:
            urls: List of LinkedIn profile URLs

        Returns:
            List of validated URLs
        """
        valid_urls = []

        for url in urls:
            # Basic URL validation
            try:
                parsed = urlparse(url)
                if not parsed.scheme:
                    url = 'https://' + url
                    parsed = urlparse(url)

                # Check if it's a LinkedIn URL
                if 'linkedin.com' not in parsed.netloc.lower():
                    logger.warning(f"Invalid LinkedIn URL: {url}")
                    continue

                # Ensure it's a profile URL
                if '/in/' not in parsed.path and '/pub/' not in parsed.path:
                    logger.warning(f"Not a LinkedIn profile URL: {url}")
                    continue

                valid_urls.append(url)

            except Exception as e:
                logger.error(f"Error validating URL {url}: {e}")
                continue

        return valid_urls

    def extract_profiles(self, urls: List[str]) -> BrightDataResponse:
        """
        Extract LinkedIn profile data for multiple URLs

        Args:
            urls: List of LinkedIn profile URLs

        Returns:
            BrightDataResponse with extracted profile data
        """
        start_time = time.time()

        try:
            # Validate URLs
            valid_urls = self._validate_linkedin_urls(urls)
            if not valid_urls:
                raise ValueError("No valid LinkedIn URLs provided")

            logger.info(f"Extracting data for {len(valid_urls)} LinkedIn profiles")

            # Step 1: Trigger the snapshot creation
            logger.info("Step 1: Triggering snapshot creation...")
            snapshot_id = self._trigger_snapshot(valid_urls)

            if not snapshot_id:
                raise ValueError("Failed to create snapshot")

            logger.info(f"Snapshot created with ID: {snapshot_id}")

            # Step 2: Poll for completion
            logger.info("Step 2: Polling for snapshot completion...")
            snapshot_info = self._poll_snapshot_progress(snapshot_id)

            if snapshot_info.status != SnapshotStatus.COMPLETED:
                raise ValueError(f"Snapshot failed or timed out. Status: {snapshot_info.status.value}")

            # Step 3: Retrieve snapshot data
            logger.info("Step 3: Retrieving snapshot data...")
            snapshot_data = self._get_snapshot_data(snapshot_id)

            # Parse response
            return self._parse_snapshot_response(snapshot_data, valid_urls, time.time() - start_time, snapshot_id, snapshot_info)

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return self._create_error_response(urls, str(e), time.time() - start_time)

        except Exception as e:
            logger.error(f"Profile extraction failed: {e}")
            return self._create_error_response(urls, str(e), time.time() - start_time)

    def _trigger_snapshot(self, urls: List[str]) -> Optional[str]:
        """
        Trigger snapshot creation and return snapshot ID

        Args:
            urls: List of LinkedIn profile URLs

        Returns:
            Snapshot ID if successful, None otherwise
        """
        try:
            # Check rate limit
            self._check_rate_limit()

            # Prepare request payload
            payload = [{"url": url} for url in urls]

            # Build API endpoint
            endpoint = f"{self.base_url}/trigger"
            params = {
                'dataset_id': self.dataset_id,
                'include_errors': str(self.include_errors).lower()
            }

            # Make API request
            response = requests.post(
                endpoint,
                headers=self.headers,
                json=payload,
                params=params,
                timeout=self.timeout
            )

            response.raise_for_status()
            response_data = response.json()

            # Extract snapshot ID from response
            snapshot_id = response_data.get('snapshot_id')
            if not snapshot_id:
                # Try alternative response formats
                snapshot_id = response_data.get('id') or response_data.get('request_id')

            return snapshot_id

        except Exception as e:
            logger.error(f"Failed to trigger snapshot: {e}")
            return None

    def _poll_snapshot_progress(self, snapshot_id: str) -> SnapshotInfo:
        """
        Poll snapshot progress until completion or timeout

        Args:
            snapshot_id: Snapshot ID to poll

        Returns:
            SnapshotInfo with final status
        """
        logger.info(f"Polling snapshot {snapshot_id} for completion...")

        start_time = time.time()
        attempts = 0
        last_progress = 0.0

        while attempts < self.max_polling_attempts:
            try:
                # Build progress endpoint
                endpoint = f"{self.base_url}/progress/{snapshot_id}"

                # Make progress request with simplified headers
                response = requests.get(
                    endpoint,
                    headers={'Authorization': self.headers['Authorization']},
                    timeout=30
                )

                elapsed_time = time.time() - start_time

                if response.status_code == 200:
                    try:
                        progress_data = response.json()
                    except:
                        # If JSON parsing fails, try to get status from text
                        logger.warning(f"Failed to parse JSON response: {response.text[:200]}")
                        time.sleep(self.polling_interval)
                        attempts += 1
                        continue

                    # Parse progress response with better error handling
                    snapshot_info = self._parse_progress_response(progress_data, snapshot_id)

                    # Log progress changes
                    if snapshot_info.progress_percent != last_progress:
                        logger.info(f"Snapshot {snapshot_id} status: {snapshot_info.status.value} "
                                   f"({snapshot_info.progress_percent:.1f}% complete) - {elapsed_time:.0f}s elapsed")
                        last_progress = snapshot_info.progress_percent

                    # Check if completed or failed
                    if snapshot_info.status == SnapshotStatus.COMPLETED:
                        logger.info(f"Snapshot {snapshot_id} completed successfully after {elapsed_time:.0f}s!")
                        return snapshot_info
                    elif snapshot_info.status == SnapshotStatus.FAILED:
                        logger.error(f"Snapshot {snapshot_id} failed: {snapshot_info.error_message}")
                        return snapshot_info
                    elif snapshot_info.status == SnapshotStatus.CANCELED:
                        logger.error(f"Snapshot {snapshot_id} was canceled")
                        return snapshot_info

                elif response.status_code == 404:
                    # Snapshot not found - might be too early or invalid ID
                    logger.warning(f"Snapshot {snapshot_id} not found (attempt {attempts + 1})")
                    if attempts < 5:  # Give it a few more tries for new snapshots
                        time.sleep(self.polling_interval * 2)  # Wait longer
                        attempts += 1
                        continue
                    else:
                        return SnapshotInfo(
                            snapshot_id=snapshot_id,
                            status=SnapshotStatus.FAILED,
                            progress_percent=0.0,
                            error_message=f"Snapshot not found after {attempts} attempts"
                        )
                else:
                    logger.warning(f"Progress request failed with status {response.status_code}: {response.text[:200]}")

                # Wait before next poll
                time.sleep(self.polling_interval)
                attempts += 1

            except requests.exceptions.RequestException as e:
                logger.warning(f"Progress polling request failed (attempt {attempts + 1}): {e}")
                time.sleep(self.polling_interval)
                attempts += 1
            except Exception as e:
                logger.warning(f"Progress polling attempt {attempts + 1} failed: {e}")
                time.sleep(self.polling_interval)
                attempts += 1

        # Timeout reached
        elapsed_time = time.time() - start_time
        logger.error(f"Snapshot {snapshot_id} timed out after {elapsed_time:.1f} seconds")

        return SnapshotInfo(
            snapshot_id=snapshot_id,
            status=SnapshotStatus.FAILED,
            progress_percent=last_progress,
            error_message=f"Polling timeout after {elapsed_time:.1f} seconds ({attempts} attempts)"
        )

    def _get_snapshot_data(self, snapshot_id: str) -> Dict[str, Any]:
        """
        Retrieve snapshot data once completed

        Args:
            snapshot_id: Snapshot ID to retrieve

        Returns:
            Snapshot data
        """
        try:
            # Build snapshot endpoint
            endpoint = f"{self.base_url}/snapshot/{snapshot_id}"
            params = {
                'format': 'json'
            }

            # Make snapshot data request
            response = requests.get(
                endpoint,
                headers=self.headers,
                params=params,
                timeout=self.timeout
            )

            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.error(f"Failed to retrieve snapshot data: {e}")
            raise

    def _parse_progress_response(self, progress_data: Dict[str, Any], snapshot_id: str) -> SnapshotInfo:
        """
        Parse progress response from API with improved handling

        Args:
            progress_data: Raw progress response
            snapshot_id: Snapshot ID

        Returns:
            SnapshotInfo object
        """
        try:
            # Debug log the raw response
            logger.debug(f"Raw progress data for {snapshot_id}: {progress_data}")

            # Parse status with multiple possible field names
            status_str = (
                progress_data.get('status', '') or
                progress_data.get('state', '') or
                progress_data.get('snapshot_status', '') or
                'unknown'
            ).lower().strip()

            # Map various status strings to our enum
            if status_str in ['completed', 'complete', 'done', 'finished', 'success', 'successful']:
                status = SnapshotStatus.COMPLETED
            elif status_str in ['failed', 'fail', 'error', 'failed_with_errors']:
                status = SnapshotStatus.FAILED
            elif status_str in ['canceled', 'cancelled', 'aborted']:
                status = SnapshotStatus.CANCELED
            elif status_str in ['running', 'processing', 'in_progress', 'active']:
                status = SnapshotStatus.RUNNING
            elif status_str in ['pending', 'queued', 'waiting']:
                status = SnapshotStatus.PENDING
            else:
                # Default to running if we don't recognize the status
                logger.warning(f"Unknown status '{status_str}' for snapshot {snapshot_id}, assuming RUNNING")
                status = SnapshotStatus.RUNNING

            # Parse progress with multiple possible field names and formats
            progress_percent = 0.0
            for field in ['progress', 'progress_percent', 'completion', 'percent', 'completed_percent']:
                if field in progress_data:
                    try:
                        value = progress_data[field]
                        if isinstance(value, (int, float)):
                            progress_percent = float(value)
                            # Handle both 0-1 and 0-100 formats
                            if progress_percent <= 1.0:
                                progress_percent *= 100
                            break
                        elif isinstance(value, str) and '%' in value:
                            progress_percent = float(value.replace('%', ''))
                            break
                    except (ValueError, TypeError):
                        continue

            # If status is completed but progress is 0, set to 100
            if status == SnapshotStatus.COMPLETED and progress_percent == 0.0:
                progress_percent = 100.0

            # Parse record counts with multiple field names
            total_records = None
            for field in ['total_records', 'total', 'total_items', 'total_urls', 'input_count']:
                if field in progress_data and progress_data[field] is not None:
                    try:
                        total_records = int(progress_data[field])
                        break
                    except (ValueError, TypeError):
                        continue

            completed_records = None
            for field in ['completed_records', 'completed', 'processed_items', 'success_count', 'processed']:
                if field in progress_data and progress_data[field] is not None:
                    try:
                        completed_records = int(progress_data[field])
                        break
                    except (ValueError, TypeError):
                        continue

            failed_records = None
            for field in ['failed_records', 'failed', 'failed_items', 'error_count', 'errors']:
                if field in progress_data and progress_data[field] is not None:
                    try:
                        failed_records = int(progress_data[field])
                        break
                    except (ValueError, TypeError):
                        continue

            # Calculate progress from record counts if not available
            if progress_percent == 0.0 and total_records and completed_records is not None:
                if total_records > 0:
                    progress_percent = (completed_records / total_records) * 100

            # Parse timestamps
            created_at = None
            completed_at = None

            for field in ['created_at', 'created', 'start_time', 'started_at']:
                if field in progress_data and progress_data[field]:
                    try:
                        created_at = datetime.fromisoformat(str(progress_data[field]).replace('Z', '+00:00'))
                        break
                    except:
                        continue

            for field in ['completed_at', 'finished_at', 'end_time', 'completed']:
                if field in progress_data and progress_data[field]:
                    try:
                        completed_at = datetime.fromisoformat(str(progress_data[field]).replace('Z', '+00:00'))
                        break
                    except:
                        continue

            # Parse error message
            error_message = None
            for field in ['error', 'error_message', 'message', 'failure_reason', 'error_description']:
                if field in progress_data and progress_data[field]:
                    error_message = str(progress_data[field])
                    break

            snapshot_info = SnapshotInfo(
                snapshot_id=snapshot_id,
                status=status,
                progress_percent=progress_percent,
                total_records=total_records,
                completed_records=completed_records,
                failed_records=failed_records,
                created_at=created_at,
                completed_at=completed_at,
                error_message=error_message
            )

            logger.debug(f"Parsed snapshot info: {snapshot_info}")
            return snapshot_info

        except Exception as e:
            logger.error(f"Failed to parse progress response for {snapshot_id}: {e}")
            logger.error(f"Raw response data: {progress_data}")
            return SnapshotInfo(
                snapshot_id=snapshot_id,
                status=SnapshotStatus.FAILED,
                progress_percent=0.0,
                error_message=f"Failed to parse progress: {str(e)}"
            )

    def _parse_snapshot_response(self, response_data: Dict[str, Any], urls: List[str], execution_time: float,
                               snapshot_id: str, snapshot_info: SnapshotInfo) -> BrightDataResponse:
        """
        Parse Bright Data snapshot response

        Args:
            response_data: Raw response from snapshot API
            urls: Original URLs requested
            execution_time: Time taken for entire process
            snapshot_id: Snapshot ID
            snapshot_info: Snapshot information

        Returns:
            Parsed BrightDataResponse
        """
        profiles = []
        errors = []
        successful_extractions = 0

        # Handle different response formats
        if isinstance(response_data, dict):
            # Response with request ID and data
            request_id = response_data.get('request_id', response_data.get('id', 'unknown'))
            data_items = response_data.get('data', response_data.get('items', []))
            error_items = response_data.get('errors', [])

            # Process successful extractions
            for item in data_items:
                profile = self._parse_profile_data(item)
                if profile.success:
                    successful_extractions += 1
                profiles.append(profile)

            # Process errors
            for error in error_items:
                errors.append(error)
                # Create failed profile entry
                url = error.get('url', 'unknown')
                failed_profile = LinkedInProfileData(
                    url=url,
                    error_message=error.get('error', 'Unknown error'),
                    success=False,
                    extraction_timestamp=datetime.now()
                )
                profiles.append(failed_profile)

        elif isinstance(response_data, list):
            # Direct list response
            request_id = f"batch_{int(time.time())}"
            for item in response_data:
                profile = self._parse_profile_data(item)
                if profile.success:
                    successful_extractions += 1
                profiles.append(profile)

        else:
            # Unexpected response format
            request_id = 'error'
            errors.append({'error': 'Unexpected response format', 'response': response_data})

        return BrightDataResponse(
            request_id=request_id,
            snapshot_id=snapshot_id,
            status=BrightDataStatus.COMPLETED if successful_extractions > 0 else BrightDataStatus.FAILED,
            profiles=profiles,
            total_profiles=len(urls),
            successful_extractions=successful_extractions,
            failed_extractions=len(urls) - successful_extractions,
            errors=errors,
            execution_time=execution_time,
            snapshot_info=snapshot_info
        )

    def _parse_profile_data(self, data: Dict[str, Any]) -> LinkedInProfileData:
        """
        Parse individual profile data from API response

        Args:
            data: Raw profile data from API

        Returns:
            LinkedInProfileData object
        """
        try:
            # Extract basic information
            url = data.get('url', data.get('profile_url', ''))
            full_name = data.get('name', data.get('full_name', data.get('firstName', '') + ' ' + data.get('lastName', ''))).strip()
            headline = data.get('headline', data.get('title', ''))
            location = data.get('location', data.get('geo_location', ''))
            summary = data.get('summary', data.get('about', ''))

            # Current position
            current_position = None
            if 'currentPosition' in data:
                current_position = data['currentPosition']
            elif 'current_position' in data:
                current_position = data['current_position']
            elif 'experience' in data and data['experience']:
                # Take the first experience as current
                experiences = data['experience']
                if isinstance(experiences, list) and experiences:
                    current_position = experiences[0]

            # Experience history
            experience = []
            if 'experience' in data and isinstance(data['experience'], list):
                experience = data['experience']
            elif 'positions' in data and isinstance(data['positions'], list):
                experience = data['positions']

            # Education
            education = []
            if 'education' in data and isinstance(data['education'], list):
                education = data['education']
            elif 'schools' in data and isinstance(data['schools'], list):
                education = data['schools']

            # Skills
            skills = []
            if 'skills' in data:
                if isinstance(data['skills'], list):
                    skills = [skill if isinstance(skill, str) else skill.get('name', '') for skill in data['skills']]
                elif isinstance(data['skills'], str):
                    skills = [data['skills']]

            # Connection counts
            connections_count = data.get('connections', data.get('connectionCount', data.get('connections_count')))
            followers_count = data.get('followers', data.get('followerCount', data.get('followers_count')))

            # Additional fields
            industry = data.get('industry', '')
            profile_image_url = data.get('profilePicture', data.get('profile_image', data.get('avatar', '')))

            # Languages
            languages = []
            if 'languages' in data and isinstance(data['languages'], list):
                languages = [lang if isinstance(lang, str) else lang.get('name', '') for lang in data['languages']]

            # Certifications
            certifications = []
            if 'certifications' in data and isinstance(data['certifications'], list):
                certifications = data['certifications']

            # Volunteer experience
            volunteer_experience = []
            if 'volunteer' in data and isinstance(data['volunteer'], list):
                volunteer_experience = data['volunteer']
            elif 'volunteerWork' in data and isinstance(data['volunteerWork'], list):
                volunteer_experience = data['volunteerWork']

            return LinkedInProfileData(
                url=url,
                full_name=full_name,
                headline=headline,
                location=location,
                summary=summary,
                current_position=current_position,
                experience=experience,
                education=education,
                skills=skills,
                connections_count=connections_count,
                followers_count=followers_count,
                profile_image_url=profile_image_url,
                industry=industry,
                about=summary,
                languages=languages,
                certifications=certifications,
                volunteer_experience=volunteer_experience,
                raw_data=data,
                extraction_timestamp=datetime.now(),
                success=True
            )

        except Exception as e:
            logger.error(f"Error parsing profile data: {e}")
            return LinkedInProfileData(
                url=data.get('url', 'unknown'),
                error_message=str(e),
                success=False,
                extraction_timestamp=datetime.now(),
                raw_data=data
            )

    def _create_error_response(self, urls: List[str], error_message: str, execution_time: float) -> BrightDataResponse:
        """
        Create error response for failed requests

        Args:
            urls: Original URLs requested
            error_message: Error message
            execution_time: Time taken for request

        Returns:
            BrightDataResponse with error information
        """
        profiles = []
        for url in urls:
            profiles.append(LinkedInProfileData(
                url=url,
                error_message=error_message,
                success=False,
                extraction_timestamp=datetime.now()
            ))

        return BrightDataResponse(
            request_id='error',
            snapshot_id=None,
            status=BrightDataStatus.FAILED,
            profiles=profiles,
            total_profiles=len(urls),
            successful_extractions=0,
            failed_extractions=len(urls),
            errors=[{'error': error_message}],
            execution_time=execution_time,
            snapshot_info=None
        )

    def extract_single_profile(self, url: str) -> LinkedInProfileData:
        """
        Extract data for a single LinkedIn profile

        Args:
            url: LinkedIn profile URL

        Returns:
            LinkedInProfileData object
        """
        response = self.extract_profiles([url])

        if response.profiles:
            return response.profiles[0]
        else:
            return LinkedInProfileData(
                url=url,
                error_message="No data returned from API",
                success=False,
                extraction_timestamp=datetime.now()
            )

    def verify_profile_against_resume(self, profile_data: LinkedInProfileData, resume_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify LinkedIn profile data against resume information

        Args:
            profile_data: LinkedIn profile data
            resume_data: Parsed resume data

        Returns:
            Verification results with discrepancies and match scores
        """
        verification_results = {
            'overall_match_score': 0.0,
            'name_match': self._verify_name(profile_data.full_name, resume_data.get('name', '')),
            'experience_match': self._verify_experience(profile_data.experience or [], resume_data.get('experience', [])),
            'education_match': self._verify_education(profile_data.education or [], resume_data.get('education', [])),
            'skills_match': self._verify_skills(profile_data.skills or [], resume_data.get('skills', [])),
            'location_match': self._verify_location(profile_data.location, resume_data.get('location', '')),
            'discrepancies': [],
            'confidence_score': 0.0,
            'fraud_indicators': [],
            'verification_timestamp': datetime.now().isoformat()
        }

        # Calculate overall match score
        scores = []
        if verification_results['name_match']['score'] is not None:
            scores.append(verification_results['name_match']['score'])
        if verification_results['experience_match']['score'] is not None:
            scores.append(verification_results['experience_match']['score'])
        if verification_results['education_match']['score'] is not None:
            scores.append(verification_results['education_match']['score'])
        if verification_results['skills_match']['score'] is not None:
            scores.append(verification_results['skills_match']['score'])

        if scores:
            verification_results['overall_match_score'] = sum(scores) / len(scores)
            verification_results['confidence_score'] = min(verification_results['overall_match_score'], 0.95)

        # Identify fraud indicators
        if verification_results['overall_match_score'] < 0.5:
            verification_results['fraud_indicators'].append('Low overall match score')

        if not verification_results['name_match']['match']:
            verification_results['fraud_indicators'].append('Name mismatch')

        if verification_results['experience_match']['score'] and verification_results['experience_match']['score'] < 0.3:
            verification_results['fraud_indicators'].append('Significant experience discrepancies')

        return verification_results

    def _verify_name(self, linkedin_name: str, resume_name: str) -> Dict[str, Any]:
        """Verify name match between LinkedIn and resume"""
        if not linkedin_name or not resume_name:
            return {'match': False, 'score': None, 'details': 'Missing name data'}

        # Basic string similarity
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, linkedin_name.lower(), resume_name.lower()).ratio()

        return {
            'match': similarity > 0.8,
            'score': similarity,
            'details': f'LinkedIn: "{linkedin_name}" vs Resume: "{resume_name}"'
        }

    def _verify_experience(self, linkedin_exp: List[Dict], resume_exp: List[Dict]) -> Dict[str, Any]:
        """Verify experience match between LinkedIn and resume"""
        if not linkedin_exp and not resume_exp:
            return {'match': True, 'score': 1.0, 'details': 'No experience data to compare'}

        if not linkedin_exp or not resume_exp:
            return {'match': False, 'score': 0.0, 'details': 'Missing experience data'}

        # Simple matching logic - can be enhanced
        matches = 0
        total_comparisons = max(len(linkedin_exp), len(resume_exp))

        for li_exp in linkedin_exp[:3]:  # Check top 3 experiences
            li_company = li_exp.get('company', '').lower()
            li_title = li_exp.get('title', '').lower()

            for res_exp in resume_exp[:3]:
                res_company = res_exp.get('company', '').lower()
                res_title = res_exp.get('title', '').lower()

                if (li_company and res_company and li_company in res_company) or \
                   (li_title and res_title and li_title in res_title):
                    matches += 1
                    break

        score = matches / min(len(linkedin_exp), len(resume_exp), 3) if matches > 0 else 0.0

        return {
            'match': score > 0.5,
            'score': score,
            'details': f'Found {matches} matching experiences out of {min(len(linkedin_exp), len(resume_exp))} compared'
        }

    def _verify_education(self, linkedin_edu: List[Dict], resume_edu: List[Dict]) -> Dict[str, Any]:
        """Verify education match between LinkedIn and resume"""
        if not linkedin_edu and not resume_edu:
            return {'match': True, 'score': 1.0, 'details': 'No education data to compare'}

        if not linkedin_edu or not resume_edu:
            return {'match': False, 'score': 0.5, 'details': 'Missing education data'}

        matches = 0
        for li_edu in linkedin_edu:
            li_school = li_edu.get('school', '').lower()
            li_degree = li_edu.get('degree', '').lower()

            for res_edu in resume_edu:
                res_school = res_edu.get('school', '').lower()
                res_degree = res_edu.get('degree', '').lower()

                if (li_school and res_school and li_school in res_school) or \
                   (li_degree and res_degree and li_degree in res_degree):
                    matches += 1
                    break

        score = matches / min(len(linkedin_edu), len(resume_edu)) if matches > 0 else 0.0

        return {
            'match': score > 0.5,
            'score': score,
            'details': f'Found {matches} matching education entries'
        }

    def _verify_skills(self, linkedin_skills: List[str], resume_skills: List[str]) -> Dict[str, Any]:
        """Verify skills match between LinkedIn and resume"""
        if not linkedin_skills and not resume_skills:
            return {'match': True, 'score': 1.0, 'details': 'No skills data to compare'}

        if not linkedin_skills or not resume_skills:
            return {'match': False, 'score': 0.5, 'details': 'Missing skills data'}

        li_skills_lower = [skill.lower() for skill in linkedin_skills if skill]
        res_skills_lower = [skill.lower() for skill in resume_skills if skill]

        common_skills = set(li_skills_lower).intersection(set(res_skills_lower))
        all_skills = set(li_skills_lower).union(set(res_skills_lower))

        score = len(common_skills) / len(all_skills) if all_skills else 0.0

        return {
            'match': score > 0.3,
            'score': score,
            'details': f'Found {len(common_skills)} common skills out of {len(all_skills)} total unique skills'
        }

    def _verify_location(self, linkedin_location: str, resume_location: str) -> Dict[str, Any]:
        """Verify location match between LinkedIn and resume"""
        if not linkedin_location and not resume_location:
            return {'match': True, 'score': 1.0, 'details': 'No location data to compare'}

        if not linkedin_location or not resume_location:
            return {'match': False, 'score': 0.5, 'details': 'Missing location data'}

        # Simple location matching
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, linkedin_location.lower(), resume_location.lower()).ratio()

        return {
            'match': similarity > 0.6,
            'score': similarity,
            'details': f'LinkedIn: "{linkedin_location}" vs Resume: "{resume_location}"'
        }

    def get_service_status(self) -> Dict[str, Any]:
        """Get service status and configuration"""
        return {
            'service': 'Bright Data LinkedIn Service (Async Snapshots)',
            'api_key_configured': bool(self.api_key),
            'dataset_id': self.dataset_id,
            'rate_limit': self.rate_limit,
            'requests_made': self.request_count,
            'base_url': self.base_url,
            'timeout': self.timeout,
            'include_errors': self.include_errors,
            'polling_interval': self.polling_interval,
            'max_polling_time': self.max_polling_time,
            'max_polling_attempts': self.max_polling_attempts
        }

    def get_snapshot_status(self, snapshot_id: str) -> SnapshotInfo:
        """
        Get current status of a specific snapshot

        Args:
            snapshot_id: Snapshot ID to check

        Returns:
            SnapshotInfo with current status
        """
        try:
            endpoint = f"{self.base_url}/progress/{snapshot_id}"
            response = requests.get(endpoint, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            progress_data = response.json()
            return self._parse_progress_response(progress_data, snapshot_id)
        except Exception as e:
            logger.error(f"Failed to get snapshot status: {e}")
            return SnapshotInfo(
                snapshot_id=snapshot_id,
                status=SnapshotStatus.FAILED,
                progress_percent=0.0,
                error_message=str(e)
            )

# Create a global instance
brightdata_service = BrightDataLinkedInService()
