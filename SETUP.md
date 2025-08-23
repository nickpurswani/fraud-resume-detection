# Setup Guide - Fraud Resume Detection Tool

## Quick Start

### 1. Environment Variables Setup

The application requires several API keys to function properly. Set these environment variables:

#### Required for Basic Functionality:
```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
```

#### Optional for Enhanced Features:
```bash
export BRIGHTDATA_API_KEY="your_brightdata_api_key_here"
export LINKEDIN_API_KEY="your_linkedin_api_key_here"
```

### 2. Getting API Keys

#### Gemini API Key (Required)
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated key

#### Bright Data API Key (Optional - for LinkedIn verification)
1. Go to [Bright Data](https://brightdata.com)
2. Create a free account
3. Navigate to "Datasets" → Search for "LinkedIn"
4. Subscribe to dataset ID: `gd_l1viktl72bvl7bjuj0`
5. Go to "API & Integrations" → Generate API key
6. Copy the generated key

#### LinkedIn API Key (Optional - alternative to Bright Data)
1. Go to [LinkedIn Developer Portal](https://developer.linkedin.com/)
2. Create an application
3. Generate API credentials
4. Copy the API key

### 3. Setting Environment Variables

#### Option A: Export Commands (Temporary)
```bash
export GEMINI_API_KEY="your_gemini_key_here"
export BRIGHTDATA_API_KEY="your_brightdata_key_here"
export LINKEDIN_API_KEY="your_linkedin_key_here"
```

#### Option B: .env File (Recommended)
Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_gemini_key_here
BRIGHTDATA_API_KEY=your_brightdata_key_here
LINKEDIN_API_KEY=your_linkedin_key_here
```

#### Option C: System Environment (Permanent)

**On macOS/Linux:**
Add to your `~/.bashrc` or `~/.zshrc`:
```bash
export GEMINI_API_KEY="your_gemini_key_here"
export BRIGHTDATA_API_KEY="your_brightdata_key_here"
export LINKEDIN_API_KEY="your_linkedin_key_here"
```

**On Windows:**
```cmd
setx GEMINI_API_KEY "your_gemini_key_here"
setx BRIGHTDATA_API_KEY "your_brightdata_key_here"
setx LINKEDIN_API_KEY "your_linkedin_key_here"
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Test Your Setup

#### Basic Test (Gemini only):
```bash
python -c "
from config import Config
result = Config.validate_config()
print('✅ Valid:', result['valid'])
print('⚠️ Warnings:', result['warnings'])
print('❌ Issues:', result['issues'])
"
```

#### Test Bright Data (if configured):
```bash
python test_snapshot_fixed.py
```

#### Test Complete Application:
```bash
streamlit run app.py
```

### 6. Verify Configuration

Open the Streamlit app and check the sidebar for service status:
- ✅ Green checkmarks = Service configured and working
- ❌ Red X = Service not configured or not working
- ⚠️ Yellow warning = Service configured but may have issues

### 7. Troubleshooting

#### Issue: "BRIGHTDATA_API_KEY is required" Error

**Solution 1: Skip Bright Data (Fastest)**
- In the app, check "⚡ Skip LinkedIn verification (faster analysis)"
- This uses only Gemini for fraud detection

**Solution 2: Set the API Key**
```bash
# Check if it's set
echo $BRIGHTDATA_API_KEY

# Set it if missing
export BRIGHTDATA_API_KEY="your_api_key_here"

# Restart Streamlit
pkill -f streamlit
streamlit run app.py
```

#### Issue: "No module named 'src.brightdata_linkedin_fixed'"

**Solution:**
Make sure you're running from the project root directory:
```bash
cd /path/to/fraud-resume-detection
ls src/  # Should show brightdata_linkedin_fixed.py
streamlit run app.py
```

#### Issue: "API key not configured" warnings

**Solution:**
1. Check which APIs are missing:
```bash
python -c "
import os
print('Gemini API Key:', '✅ Set' if os.getenv('GEMINI_API_KEY') else '❌ Missing')
print('Bright Data API Key:', '✅ Set' if os.getenv('BRIGHTDATA_API_KEY') else '❌ Missing')
print('LinkedIn API Key:', '✅ Set' if os.getenv('LINKEDIN_API_KEY') else '❌ Missing')
"
```

2. Set the missing keys using one of the methods above
3. Restart the application

### 8. Feature Availability Based on Configuration

| Configuration | Available Features |
|--------------|-------------------|
| **Gemini only** | ✅ Basic fraud detection<br/>✅ Resume analysis<br/>✅ Risk scoring |
| **Gemini + Bright Data** | ✅ All basic features<br/>✅ LinkedIn verification<br/>✅ Cross-reference checking |
| **Gemini + LinkedIn API** | ✅ All basic features<br/>✅ Basic LinkedIn lookup<br/>⚠️ Limited data |
| **All APIs** | ✅ Complete functionality<br/>✅ Multiple LinkedIn options<br/>✅ Fallback services |

### 9. Environment Variable Security

#### Best Practices:
1. **Never commit API keys to git**
2. **Use .env files for development**
3. **Use system environment variables for production**
4. **Rotate keys periodically**

#### Add to .gitignore:
```
.env
*.env
.env.local
.env.production
```

### 10. Quick Start Commands

**Minimum setup (basic fraud detection):**
```bash
export GEMINI_API_KEY="your_key_here"
streamlit run app.py
```

**Full setup (all features):**
```bash
export GEMINI_API_KEY="your_gemini_key_here"
export BRIGHTDATA_API_KEY="your_brightdata_key_here"
streamlit run app.py
```

**Test everything:**
```bash
python working_example.py
```

### 11. Getting Help

If you're still having issues:

1. **Check the logs:**
```bash
# Look for error messages
tail -f logs/app.log  # if logging to file
# or check console output
```

2. **Test individual components:**
```bash
python test_brightdata.py
python working_example.py
```

3. **Verify file structure:**
```bash
ls -la src/  # Should show all required .py files
```

4. **Check Python path:**
```bash
python -c "import sys; print('\n'.join(sys.path))"
```

---

**Ready to Start!** 🎉

Once you've set at least the `GEMINI_API_KEY`, you can run:
```bash
streamlit run app.py
```

Navigate to the URL shown in your terminal (usually `http://localhost:8501`) and start detecting fraudulent resumes!