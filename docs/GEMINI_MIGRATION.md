# Gemini AI Client Migration Guide

This guide explains how to migrate from the old Google Gemini AI client to the new Google GenAI client implementation in the Fraud Resume Detection Tool.

## What Changed

### Old Implementation (Before Migration)
- Used older Google AI client pattern
- Required API key passed directly to the analyzer
- Used `generation_config` dictionary format
- Model: `gemini-1.5-flash`

### New Implementation (After Migration)
- Uses modern `google-genai` client with `types` configuration
- Supports environment variables for API key management
- Uses structured `GenerateContentConfig` and `ThinkingConfig` types
- Model: `gemini-2.5-flash` (upgraded default)
- Disabled thinking mode for faster, more consistent responses

## Required Dependencies

Ensure you have the latest dependencies installed:

```bash
pip install google-genai>=0.5.0
pip install python-dotenv>=1.0.0
```

## Environment Variable Setup

### 1. Create `.env` File (Recommended)

Create a `.env` file in your project root:

```env
# Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Other API keys
LINKEDIN_API_KEY=your_linkedin_api_key_here

# Optional: Logging configuration
LOG_LEVEL=INFO
LOG_FILE=fraud_detection.log
```

### 2. Or Export Environment Variable

```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
```

## Code Migration

### Old Usage Pattern

```python
# OLD WAY - No longer supported
from gemini_analyzer import GeminiAnalyzer

# Had to pass API key directly
analyzer = GeminiAnalyzer(
    api_key="your_api_key_here",
    config={
        'gemini_model': 'gemini-1.5-flash',
        'gemini_temperature': 0.3
    }
)
```

### New Usage Pattern

```python
# NEW WAY - Recommended
import os
from dotenv import load_dotenv  # Optional, for .env file support
from gemini_analyzer import GeminiAnalyzer

# Load environment variables from .env file (optional)
load_dotenv()

# API key is automatically loaded from GEMINI_API_KEY environment variable
analyzer = GeminiAnalyzer(
    config={
        'gemini_model': 'gemini-2.5-flash',  # Updated default model
        'gemini_temperature': 0.3,
        'gemini_max_tokens': 2000,
        'cache_enabled': True
    }
)

# Alternative: Still supports direct API key for backward compatibility
analyzer = GeminiAnalyzer(
    api_key="your_api_key_here",  # Optional if env var is set
    config=config
)
```

### Client Initialization Changes

The internal client initialization has changed, but the public API remains the same:

```python
# Internal changes (you don't need to modify these)
# OLD:
# self.client = genai.Client(api_key=self.api_key)

# NEW:
# self.client = genai.Client()  # Uses GEMINI_API_KEY env var automatically
```

### API Call Configuration Changes

The way API calls are configured has been updated:

```python
# OLD format (internal):
response = self.client.models.generate_content(
    model=self.model,
    contents=prompt,
    generation_config={
        'temperature': self.temperature,
        'max_output_tokens': self.max_tokens,
    }
)

# NEW format (internal):
from google.genai import types

response = self.client.models.generate_content(
    model=self.model,
    contents=prompt,
    config=types.GenerateContentConfig(
        temperature=self.temperature,
        max_output_tokens=self.max_tokens,
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    )
)
```

## Configuration Updates

### Updated Default Model

The default model has been upgraded:

```python
# OLD default
'gemini_model': 'gemini-1.5-flash'

# NEW default
'gemini_model': 'gemini-2.5-flash'
```

### New Configuration Options

```python
config = {
    # Updated model
    'gemini_model': 'gemini-2.5-flash',  # or 'gemini-2.5-pro' for higher quality
    
    # Standard options (unchanged)
    'gemini_temperature': 0.3,           # 0.0-1.0, lower = more consistent
    'gemini_max_tokens': 2000,           # Response length limit
    'cache_enabled': True,               # Enable response caching
    'gemini_rate_limit': 60,             # Requests per minute
    
    # The thinking_config is automatically set to disabled for faster responses
}
```

## Benefits of the Migration

### 1. **Better Security**
- Environment variable support reduces hardcoded API keys
- Follows security best practices

### 2. **Improved Performance**
- Gemini 2.5 Flash is faster and more accurate
- Disabled thinking mode provides consistent response times
- Better caching and rate limiting

### 3. **Enhanced Developer Experience**
- Cleaner API with structured configuration types
- Better error messages and debugging
- More robust client initialization

### 4. **Future-Proof**
- Uses the latest Google GenAI client
- Compatible with future Google AI updates
- Better type safety with structured configs

## Migration Checklist

- [ ] Update dependencies: `pip install google-genai>=0.5.0`
- [ ] Set up `GEMINI_API_KEY` environment variable
- [ ] Update any hardcoded API key references
- [ ] Test your existing analysis workflows
- [ ] Update model references from `1.5-flash` to `2.5-flash` if desired
- [ ] Verify caching and rate limiting still work as expected

## Troubleshooting

### Common Issues

#### 1. **ImportError: No module named 'google.genai'**

```bash
# Solution:
pip install google-genai>=0.5.0
```

#### 2. **ValueError: Gemini API key is required**

```bash
# Solution: Set the environment variable
export GEMINI_API_KEY="your_api_key_here"

# Or add to .env file:
echo "GEMINI_API_KEY=your_api_key_here" >> .env
```

#### 3. **API Authentication Errors**

```bash
# Verify your API key is valid:
curl -H "Authorization: Bearer $GEMINI_API_KEY" \
  "https://generativelanguage.googleapis.com/v1/models"
```

#### 4. **Rate Limit Issues**

```python
# Adjust rate limiting in config:
config = {
    'gemini_rate_limit': 30,  # Reduce requests per minute
}
```

### Testing Your Migration

```python
#!/usr/bin/env python3
"""Test script to verify Gemini migration"""

import os
from gemini_analyzer import GeminiAnalyzer

def test_migration():
    try:
        # Test environment variable loading
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            print("❌ GEMINI_API_KEY not found in environment")
            return False
            
        print(f"✅ API key found: {api_key[:10]}...")
        
        # Test analyzer initialization
        analyzer = GeminiAnalyzer()
        print("✅ Analyzer initialized successfully")
        
        # Test basic analysis
        test_text = "Test resume content for fraud detection"
        result = analyzer.analyze_experience_fraud(test_text)
        
        print(f"✅ Analysis completed: Risk level {result.risk_level.value}")
        print("🎉 Migration successful!")
        return True
        
    except Exception as e:
        print(f"❌ Migration test failed: {e}")
        return False

if __name__ == "__main__":
    test_migration()
```

## Getting Help

If you encounter issues during migration:

1. **Check the logs**: Look for detailed error messages in your application logs
2. **Verify API key**: Ensure your Google AI API key is valid and has proper permissions
3. **Test connectivity**: Verify you can reach Google's API endpoints
4. **Review examples**: Check the `examples/gemini_analyzer_example.py` for working code
5. **Run tests**: Execute the test suite to verify functionality

## Example: Complete Migration

Here's a complete before/after example:

### Before (Old Implementation)

```python
from gemini_analyzer import GeminiAnalyzer

analyzer = GeminiAnalyzer(
    api_key="hardcoded_api_key_here",
    config={'gemini_model': 'gemini-1.5-flash'}
)

result = analyzer.analyze_experience_fraud(resume_text)
print(f"Risk: {result.risk_level}")
```

### After (New Implementation)

```python
import os
from dotenv import load_dotenv
from gemini_analyzer import GeminiAnalyzer

# Load environment variables
load_dotenv()

# Initialize with environment variable
analyzer = GeminiAnalyzer(
    config={
        'gemini_model': 'gemini-2.5-flash',
        'gemini_temperature': 0.3,
        'cache_enabled': True
    }
)

result = analyzer.analyze_experience_fraud(resume_text)
print(f"Risk: {result.risk_level}")
print(f"Confidence: {result.confidence:.2%}")
```

This migration ensures your application uses the latest, most secure, and performant Google AI integration.