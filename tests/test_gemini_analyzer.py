"""
Test suite for GeminiAnalyzer with updated Google GenAI client

This module tests the updated Gemini analyzer implementation using the new
Google GenAI client pattern with environment variables and types configuration.
"""

import pytest
import os
import time
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

# Import the classes we're testing
from src.gemini_analyzer import (
    GeminiAnalyzer,
    AnalysisType,
    RiskLevel,
    GeminiAnalysisResult
)

class TestGeminiAnalyzer:
    """Test cases for GeminiAnalyzer class"""

    def setup_method(self):
        """Set up test fixtures"""
        self.mock_api_key = "test_gemini_api_key_123"
        self.mock_response_text = """
        {
            "risk_level": "medium",
            "confidence": 0.85,
            "findings": ["Inconsistent job titles", "Timeline gaps"],
            "evidence": {"gap_months": 6, "title_inconsistency": true},
            "recommendations": ["Verify employment history", "Request references"]
        }
        """

    def test_initialization_with_api_key_parameter(self):
        """Test initialization when API key is passed as parameter"""
        with patch('src.gemini_analyzer.genai.Client') as mock_client:
            analyzer = GeminiAnalyzer(api_key=self.mock_api_key)

            assert analyzer.api_key == self.mock_api_key
            assert analyzer.model == 'gemini-2.5-flash'  # Updated default model
            mock_client.assert_called_once()

    @patch.dict(os.environ, {'GEMINI_API_KEY': 'env_test_key'})
    def test_initialization_with_env_variable(self):
        """Test initialization when API key is loaded from environment variable"""
        with patch('src.gemini_analyzer.genai.Client') as mock_client:
            analyzer = GeminiAnalyzer()

            assert analyzer.api_key == 'env_test_key'
            mock_client.assert_called_once()

    def test_initialization_without_api_key_raises_error(self):
        """Test that initialization fails when no API key is provided"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Gemini API key is required"):
                GeminiAnalyzer()

    def test_initialization_with_custom_config(self):
        """Test initialization with custom configuration"""
        config = {
            'gemini_model': 'gemini-2.5-pro',
            'gemini_temperature': 0.5,
            'gemini_max_tokens': 3000,
            'cache_enabled': False
        }

        with patch('src.gemini_analyzer.genai.Client') as mock_client:
            analyzer = GeminiAnalyzer(api_key=self.mock_api_key, config=config)

            assert analyzer.model == 'gemini-2.5-pro'
            assert analyzer.temperature == 0.5
            assert analyzer.max_tokens == 3000
            assert analyzer.cache_enabled == False

    @patch('src.gemini_analyzer.genai.Client')
    def test_api_call_with_new_client_format(self, mock_client_class):
        """Test that API calls use the new client format with types"""
        # Setup mock
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = self.mock_response_text
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client

        analyzer = GeminiAnalyzer(api_key=self.mock_api_key)

        # Make API call
        prompt = "Analyze this resume for fraud indicators"
        result = analyzer._make_api_call(prompt, "test_analysis")

        # Verify the call was made with correct parameters
        mock_client.models.generate_content.assert_called_once()
        call_args = mock_client.models.generate_content.call_args

        # Check that the call includes model and contents
        assert call_args.kwargs['model'] == 'gemini-2.5-flash'
        assert call_args.kwargs['contents'] == prompt

        # Check that config uses new types format
        config = call_args.kwargs['config']
        assert hasattr(config, 'temperature')
        assert hasattr(config, 'max_output_tokens')
        assert hasattr(config, 'thinking_config')

        assert result == self.mock_response_text

    @patch('src.gemini_analyzer.genai.Client')
    def test_experience_fraud_analysis(self, mock_client_class):
        """Test experience fraud analysis with new client"""
        # Setup mock
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = self.mock_response_text
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client

        analyzer = GeminiAnalyzer(api_key=self.mock_api_key)

        # Test data
        resume_text = "Software Engineer with 10 years experience at Google..."

        # Call the method
        result = analyzer.analyze_experience_fraud(resume_text)

        # Verify result structure
        assert isinstance(result, GeminiAnalysisResult)
        assert result.analysis_type == AnalysisType.EXPERIENCE_VALIDATION
        assert isinstance(result.processing_time, float)
        assert result.processing_time > 0

    @patch('src.gemini_analyzer.genai.Client')
    def test_rate_limiting(self, mock_client_class):
        """Test rate limiting functionality"""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Create analyzer with low rate limit for testing
        config = {'gemini_rate_limit': 2}  # 2 requests per minute
        analyzer = GeminiAnalyzer(api_key=self.mock_api_key, config=config)

        # Simulate hitting rate limit
        current_time = time.time()
        analyzer.request_times = [current_time - 10, current_time - 5, current_time - 1]

        # Should return False when rate limit is exceeded
        assert analyzer._check_rate_limit() == False

    @patch('src.gemini_analyzer.genai.Client')
    def test_caching_functionality(self, mock_client_class):
        """Test that caching works correctly"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = self.mock_response_text
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client

        analyzer = GeminiAnalyzer(api_key=self.mock_api_key)

        # Make first call
        prompt = "Test prompt"
        result1 = analyzer._make_api_call(prompt, "test")

        # Make second call with same prompt
        result2 = analyzer._make_api_call(prompt, "test")

        # Should only make one actual API call due to caching
        assert mock_client.models.generate_content.call_count == 1
        assert result1 == result2

    @patch('src.gemini_analyzer.genai.Client')
    def test_error_handling_api_failure(self, mock_client_class):
        """Test error handling when API call fails"""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("API Error")
        mock_client_class.return_value = mock_client

        analyzer = GeminiAnalyzer(api_key=self.mock_api_key)

        with pytest.raises(Exception, match="API Error"):
            analyzer._make_api_call("test prompt", "test")

    def test_module_import_error_handling(self):
        """Test handling when google-genai is not available"""
        with patch('src.gemini_analyzer.GEMINI_AVAILABLE', False):
            with pytest.raises(ImportError, match="Google Gemini AI not available"):
                GeminiAnalyzer(api_key=self.mock_api_key)

    @patch('src.gemini_analyzer.genai.Client')
    def test_thinking_config_disabled(self, mock_client_class):
        """Test that thinking config is properly disabled"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = self.mock_response_text
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client

        analyzer = GeminiAnalyzer(api_key=self.mock_api_key)
        analyzer._make_api_call("test prompt", "test")

        # Get the config from the API call
        call_args = mock_client.models.generate_content.call_args
        config = call_args.kwargs['config']

        # Verify thinking is disabled (budget = 0)
        assert config.thinking_config.thinking_budget == 0

    @patch('src.gemini_analyzer.genai.Client')
    def test_json_response_parsing(self, mock_client_class):
        """Test JSON response parsing functionality"""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        analyzer = GeminiAnalyzer(api_key=self.mock_api_key)

        # Test valid JSON parsing
        json_response = '{"risk_level": "high", "confidence": 0.9}'
        result = analyzer._parse_json_response(json_response, AnalysisType.EXPERIENCE_VALIDATION)

        assert result.risk_level == RiskLevel.HIGH
        assert result.confidence == 0.9
        assert isinstance(result, GeminiAnalysisResult)
        assert result.analysis_type == AnalysisType.EXPERIENCE_VALIDATION

    @patch('src.gemini_analyzer.genai.Client')
    def test_cache_key_generation(self, mock_client_class):
        """Test cache key generation"""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        analyzer = GeminiAnalyzer(api_key=self.mock_api_key)

        # Test cache key generation
        prompt = "test prompt"
        analysis_type = "experience"

        key1 = analyzer._get_cache_key(prompt, analysis_type)
        key2 = analyzer._get_cache_key(prompt, analysis_type)
        key3 = analyzer._get_cache_key("different prompt", analysis_type)

        # Same inputs should produce same key
        assert key1 == key2
        # Different inputs should produce different keys
        assert key1 != key3


class TestGeminiAnalysisResult:
    """Test cases for GeminiAnalysisResult dataclass"""

    def test_analysis_result_creation(self):
        """Test creation and serialization of analysis result"""
        result = GeminiAnalysisResult(
            analysis_type=AnalysisType.EXPERIENCE_VALIDATION,
            risk_level=RiskLevel.MEDIUM,
            confidence=0.85,
            findings=["Test finding"],
            evidence={"test": "value"},
            recommendations=["Test recommendation"],
            raw_response="Test response",
            processing_time=1.23
        )

        # Test serialization
        result_dict = asdict(result)
        assert result_dict['analysis_type'] == AnalysisType.EXPERIENCE_VALIDATION
        assert result_dict['risk_level'] == RiskLevel.MEDIUM
        assert result_dict['confidence'] == 0.85


if __name__ == "__main__":
    pytest.main([__file__])
