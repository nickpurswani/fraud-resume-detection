# Troubleshooting Guide - Fraud Detection Tool

## Common Issues and Solutions

### 🔄 Issue 1: Snapshot Stuck at 0% Progress

**Symptoms:**
- Bright Data snapshot shows "Status: Running" but progress stays at 0.0%
- Process hangs for several minutes without progress updates
- Eventually times out after 5 minutes

**Root Causes:**
1. Snapshot is still initializing (normal for first 30-60 seconds)
2. API polling interval too frequent
3. LinkedIn URLs are invalid or inaccessible
4. Bright Data service temporarily overloaded

**Solutions:**

#### Quick Fix 1: Wait Longer
- **Normal processing time**: 1-5 minutes for snapshot completion
- **First 60 seconds**: Often shows 0% while data collection starts
- **Don't panic**: 0% progress for first minute is normal

#### Quick Fix 2: Use Skip LinkedIn Option
```
1. In Single Candidate Analysis
2. Check "⚡ Skip LinkedIn verification (faster analysis)"
3. Run analysis without LinkedIn verification
4. Return to LinkedIn verification later when service is stable
```

#### Quick Fix 3: Try Different URLs
```
# Test with known working URLs first
https://www.linkedin.com/in/elad-moshe-05a90413/
https://www.linkedin.com/in/jonathan-myrvik-3baa01109
```

#### Debug Steps:
```bash
# 1. Test your API configuration
python debug_snapshot.py

# 2. Check specific snapshot
python -c "
from src.brightdata_linkedin import BrightDataLinkedInService
service = BrightDataLinkedInService()
info = service.get_snapshot_status('YOUR_SNAPSHOT_ID')
print(f'Status: {info.status.value}, Progress: {info.progress_percent}%')
"

# 3. Check logs
tail -f snapshot_debug.log
```

#### Advanced Fix: Manual Polling
```python
# If automatic polling fails, try manual checking
from src.brightdata_linkedin import BrightDataLinkedInService
import time

service = BrightDataLinkedInService()
snapshot_id = "YOUR_SNAPSHOT_ID"

# Check every 30 seconds
for i in range(20):  # 10 minutes max
    info = service.get_snapshot_status(snapshot_id)
    print(f"Attempt {i+1}: {info.status.value} - {info.progress_percent:.1f}%")
    
    if info.status.value == 'completed':
        print("✅ Ready to retrieve data!")
        break
    elif info.status.value in ['failed', 'canceled']:
        print(f"❌ Failed: {info.error_message}")
        break
    
    time.sleep(30)
```

---

### 🔗 Issue 2: LinkedIn Verification Not Showing in Single Candidate Analysis

**Symptoms:**
- LinkedIn verification option missing in single candidate analysis
- Only showing in dedicated LinkedIn Verification page
- No service selection options

**Root Cause:**
- Integration was initially missing from single candidate analysis flow

**Solution - NOW FIXED:**
```
✅ LinkedIn verification now available in Single Candidate Analysis
✅ Service selection (Traditional API vs Bright Data) added
✅ Skip LinkedIn option added for faster processing
✅ Real-time progress tracking implemented
```

**To Use:**
1. Go to **Single Candidate Analysis**
2. Upload resume or paste text
3. Enter **LinkedIn Profile URL** (optional section)
4. Choose service: **Bright Data (Recommended)** or Traditional API
5. Check **"LinkedIn Verification"** under Analysis Options
6. Or check **"⚡ Skip LinkedIn verification"** for faster processing

---

### 🔧 Issue 3: API Configuration Problems

**Symptoms:**
- "API key is required" errors
- "Service not configured" messages
- 403 Forbidden errors

**Solutions:**

#### Fix Environment Variables:
```bash
# Check current environment
echo $BRIGHTDATA_API_KEY
echo $LINKEDIN_API_KEY
echo $GEMINI_API_KEY

# Set missing variables
export BRIGHTDATA_API_KEY="your_bright_data_key_here"
export LINKEDIN_API_KEY="your_linkedin_key_here"  # Optional
export GEMINI_API_KEY="your_gemini_key_here"

# Or add to .env file
echo "BRIGHTDATA_API_KEY=your_key_here" >> .env
echo "LINKEDIN_API_KEY=your_key_here" >> .env
echo "GEMINI_API_KEY=your_key_here" >> .env
```

#### Test API Configuration:
```bash
# Test all services
python test_brightdata.py

# Test specific configuration
python -c "
from config import Config
result = Config.validate_config()
print('Valid:', result['valid'])
print('Issues:', result['issues'])
print('Warnings:', result['warnings'])
"
```

#### Restart Streamlit After Config Changes:
```bash
# Always restart after changing environment variables
pkill -f streamlit
streamlit run app.py
```

---

### ⏱️ Issue 4: Timeout and Performance Issues

**Symptoms:**
- "Snapshot processing timed out after 5 minutes"
- Very slow LinkedIn verification
- Application becomes unresponsive

**Solutions:**

#### Immediate Workarounds:
1. **Use Skip LinkedIn**: Check "⚡ Skip LinkedIn verification" for instant results
2. **Reduce batch size**: Use 1-3 LinkedIn URLs max
3. **Try later**: Bright Data may be experiencing high load

#### Optimize Processing:
```python
# In config.py - adjust timeouts
BRIGHTDATA_TIMEOUT = 120  # Increase timeout
POLLING_INTERVAL = 10     # Longer polling interval
MAX_POLLING_TIME = 600    # 10 minutes max
```

#### Performance Tips:
- **Single URLs**: Process one LinkedIn profile at a time
- **Valid URLs**: Ensure LinkedIn URLs are accessible
- **Off-peak times**: Use service during off-peak hours
- **Cache results**: Don't re-process same profiles

---

### 🐛 Issue 5: General Debugging

#### Enable Debug Logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run your problematic code
# Check detailed logs for issues
```

#### Check Service Status:
```python
from src.brightdata_linkedin import BrightDataLinkedInService

service = BrightDataLinkedInService()
status = service.get_service_status()
print(json.dumps(status, indent=2))
```

#### Test Individual Components:
```bash
# Test just fraud detection (fastest)
python -c "
from src.fraud_detector import FraudDetector
detector = FraudDetector()
result = detector.analyze_resume('Sample resume text')
print('Fraud detection working:', 'risk_score' in result)
"

# Test just NLP processing
python -c "
from src.nlp_analyzer import NLPAnalyzer
analyzer = NLPAnalyzer()
result = analyzer.analyze_text('Sample text')
print('NLP working:', len(result) > 0)
"
```

---

### 🚑 Emergency Fixes

#### If Nothing Works:
1. **Skip All External Services**: Use only basic fraud detection
   ```
   - Uncheck "LinkedIn Verification"
   - Uncheck "Job Fit Analysis" 
   - Use only core resume analysis
   ```

2. **Reset Configuration**:
   ```bash
   # Clear any cached configuration
   rm -f .env
   unset BRIGHTDATA_API_KEY
   unset LINKEDIN_API_KEY
   
   # Restart clean
   streamlit run app.py
   ```

3. **Use Minimal Analysis**:
   ```
   - Upload resume only
   - Skip all optional features
   - Focus on basic fraud indicators
   ```

#### Quick Health Check:
```bash
# Run this to check if basic functionality works
python -c "
print('Testing imports...')
from src.fraud_detector import FraudDetector
from src.nlp_analyzer import NLPAnalyzer
print('✅ Core modules working')

detector = FraudDetector()
result = detector.analyze_resume('Test resume with John Smith, software engineer experience.')
print('✅ Basic fraud detection working')
print(f'Risk score: {result.get(\"risk_score\", \"N/A\")}')
"
```

---

### 📞 Getting Help

#### Before Reporting Issues:
1. Run `python debug_snapshot.py`
2. Check `snapshot_debug.log` for detailed errors
3. Try with "Skip LinkedIn verification" option
4. Note exact error messages and steps to reproduce

#### Include in Bug Reports:
- Python version: `python --version`
- Streamlit version: `streamlit --version`
- Error messages (full stack trace)
- Configuration status (without API keys)
- Steps to reproduce
- Whether skip LinkedIn option works

#### Quick Test Command:
```bash
# Run this and include output in bug reports
python -c "
import sys
print('Python:', sys.version)

try:
    import streamlit
    print('Streamlit:', streamlit.__version__)
except: pass

from config import Config
result = Config.validate_config()
print('Config valid:', result['valid'])
print('Warnings:', len(result['warnings']))
print('Issues:', len(result['issues']))

try:
    from src.brightdata_linkedin import BrightDataLinkedInService
    service = BrightDataLinkedInService()
    print('Bright Data service: ✅')
except Exception as e:
    print('Bright Data service: ❌', str(e))
"
```

---

### ✅ Known Working Configurations

**Minimal Working Setup:**
```bash
# Only required for basic functionality
export GEMINI_API_KEY="your_gemini_key"
# Skip LinkedIn verification for fastest results
```

**Full LinkedIn Setup:**
```bash
export GEMINI_API_KEY="your_gemini_key"
export BRIGHTDATA_API_KEY="your_brightdata_key"
# Use Bright Data (Recommended) option
# Expect 1-5 minute processing times
```

**Testing URLs (known to work):**
```
https://www.linkedin.com/in/elad-moshe-05a90413/
https://www.linkedin.com/in/jonathan-myrvik-3baa01109
https://www.linkedin.com/in/aviv-tal-75b81/
```

---

### 🔄 Recent Updates

**✅ Fixed in Latest Version:**
- LinkedIn verification now available in Single Candidate Analysis
- Added "Skip LinkedIn verification" option for faster processing  
- Improved snapshot polling with better error handling
- Added real-time progress tracking in Streamlit
- Better timeout handling and user feedback
- Service status indicators for both APIs

**🚀 Workarounds No Longer Needed:**
- ~~Manual LinkedIn verification on separate page~~ → Now integrated
- ~~No progress feedback~~ → Real-time progress added  
- ~~Hard to debug issues~~ → Debug script provided
- ~~All-or-nothing analysis~~ → Skip options added

**⏳ Still Applicable:**
- 1-5 minute processing times for LinkedIn verification
- Need valid API keys for external services
- Some LinkedIn profiles may be inaccessible (private/deleted)

---

**Last Updated**: December 2024
**Version**: 2.0 (Post-async workflow integration)