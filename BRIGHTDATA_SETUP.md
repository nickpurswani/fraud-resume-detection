# Bright Data LinkedIn Integration Setup Guide

## Quick Start

This guide will help you set up and use the Bright Data LinkedIn service for enhanced profile verification in the Fraud Detection Tool. 

**Important**: Bright Data uses an **asynchronous workflow** with snapshots:
1. 🚀 **Trigger**: Create a snapshot with your LinkedIn URLs
2. ⏳ **Poll**: Check progress until completion (may take 1-5 minutes)  
3. 📥 **Retrieve**: Get the final extracted data

The integration handles this complete workflow automatically.

## 1. Get Your Bright Data API Key

### Step 1: Sign Up for Bright Data
1. Go to [brightdata.com](https://brightdata.com)
2. Create a free account
3. Navigate to your dashboard

### Step 2: Subscribe to LinkedIn Dataset
1. In your Bright Data dashboard, go to "Datasets"
2. Search for "LinkedIn" datasets
3. Subscribe to dataset ID: `gd_l1viktl72bvl7bjuj0`
4. Choose your pricing plan (free tier available)

### Step 3: Get Your API Key
1. Go to "API & Integrations" in your dashboard
2. Generate a new API key
3. Copy your API key (it looks like: `73e8f3b632204d462eb885a82f7b9ce7dba604024fabeecfde7b2e9f17ef631a`)

## 2. Configure Environment

### Option A: Set Environment Variable
```bash
# Linux/macOS
export BRIGHTDATA_API_KEY="your_api_key_here"

# Windows Command Prompt
set BRIGHTDATA_API_KEY=your_api_key_here

# Windows PowerShell
$env:BRIGHTDATA_API_KEY="your_api_key_here"
```

### Option B: Create .env File
Create a `.env` file in the project root:
```
BRIGHTDATA_API_KEY=your_api_key_here
```

## 3. Test Your Setup

### Quick Test
```bash
python test_brightdata.py
```

### Manual Test with Example URLs
```bash
python examples/brightdata_linkedin_example.py
```

### Curl Equivalent Test
```bash
python examples/brightdata_curl_demo.py
```

## 4. Complete Async Workflow - Curl Commands

Bright Data uses a **3-step async workflow**. Here are the equivalent curl commands:

**Step 1: Trigger Snapshot Creation**
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     -d '[{"url":"https://www.linkedin.com/in/elad-moshe-05a90413/"},
          {"url":"https://www.linkedin.com/in/jonathan-myrvik-3baa01109"},
          {"url":"https://www.linkedin.com/in/aviv-tal-75b81/"},
          {"url":"https://www.linkedin.com/in/bulentakar/"}]' \
     "https://api.brightdata.com/datasets/v3/trigger?dataset_id=gd_l1viktl72bvl7bjuj0&include_errors=true"

# Returns: {"snapshot_id": "s_meo6xnvq3ttlzxgly", ...}
```

**Step 2: Check Progress (repeat until completed)**
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     "https://api.brightdata.com/datasets/v3/progress/s_meo6xnvq3ttlzxgly"

# Returns: {"status": "completed", "progress": 100, ...}
```

**Step 3: Get Final Data**
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     "https://api.brightdata.com/datasets/v3/snapshot/s_meo6xnvq3ttlzxgly?format=json"

# Returns: [{"name": "...", "headline": "...", ...}, ...]
```

## 5. Using in the Web App

1. **Start the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Navigate to LinkedIn Verification**:
   - Select "LinkedIn Verification" from the sidebar
   - Choose "Bright Data (Recommended)" as your service
   - You should see ✅ "Bright Data: Available"

3. **Verify a Profile**:
   - Enter candidate name or LinkedIn URL
   - Upload resume or paste resume text
   - Click "🔍 Verify LinkedIn Profile"

## 6. Python Code Examples

### Basic Usage (Async Workflow)
```python
from src.brightdata_linkedin import BrightDataLinkedInService

# Initialize
service = BrightDataLinkedInService()

# Single profile (handles complete async workflow automatically)
profile = service.extract_single_profile(
    "https://www.linkedin.com/in/username/"
)

if profile.success:
    print(f"Name: {profile.full_name}")
    print(f"Experience: {len(profile.experience)} positions")
else:
    print(f"Error: {profile.error_message}")

# Note: This may take 1-5 minutes due to async processing
```

### Manual Async Workflow
```python
# Step 1: Trigger snapshot
snapshot_id = service._trigger_snapshot([
    "https://www.linkedin.com/in/username/"
])
print(f"Snapshot created: {snapshot_id}")

# Step 2: Poll until complete
snapshot_info = service._poll_snapshot_progress(snapshot_id)
print(f"Status: {snapshot_info.status.value}")

# Step 3: Get data (if completed)
if snapshot_info.status.value == 'completed':
    data = service._get_snapshot_data(snapshot_id)
    print("Data retrieved!")
```

### Batch Processing (Automatic Async)
```python
urls = [
    "https://www.linkedin.com/in/profile1/",
    "https://www.linkedin.com/in/profile2/"
]

# This handles the complete async workflow:
# 1. Creates snapshot with all URLs
# 2. Polls until completion (may take several minutes)
# 3. Retrieves and parses final data
response = service.extract_profiles(urls)

print(f"Snapshot ID: {response.snapshot_id}")
print(f"Success: {response.successful_extractions}/{response.total_profiles}")
print(f"Total time: {response.execution_time:.2f}s")

# Access snapshot info
if response.snapshot_info:
    print(f"Progress: {response.snapshot_info.progress_percent:.1f}%")
    print(f"Records: {response.snapshot_info.completed_records}")
```

### Resume Verification
```python
# After extracting profile
resume_data = {
    'name': 'John Doe',
    'experience': [{'company': 'Google', 'title': 'Engineer'}],
    'skills': ['Python', 'JavaScript']
}

verification = service.verify_profile_against_resume(profile, resume_data)
print(f"Match Score: {verification['overall_match_score']:.2%}")
```

## 7. Features Available

✅ **Async Snapshot Workflow**: Complete 3-step async processing  
✅ **Real-time Progress Tracking**: Monitor snapshot completion  
✅ **Single Profile Extraction**: Extract one LinkedIn profile  
✅ **Batch Processing**: Process multiple profiles simultaneously  
✅ **Resume Verification**: Compare LinkedIn data with resume  
✅ **Fraud Detection**: Identify discrepancies and red flags  
✅ **Error Handling**: Robust error management  
✅ **Rate Limiting**: Built-in request throttling  
✅ **Data Validation**: URL validation and data quality checks  
✅ **Snapshot Management**: Track and retrieve snapshot data  

## 8. Data Extracted

The service extracts comprehensive LinkedIn data:
- **Basic Info**: Name, headline, location, industry
- **Experience**: Job history, companies, titles, dates
- **Education**: Schools, degrees, graduation years  
- **Skills**: Skills list with potential endorsements
- **Network**: Connection count, follower count
- **Additional**: Certifications, languages, volunteer work

## 9. Troubleshooting

### Common Issues

**❌ "API key is required" Error**
- Solution: Set `BRIGHTDATA_API_KEY` environment variable

**❌ "Service not configured" in Web App**  
- Solution: Restart the Streamlit app after setting API key

**❌ 403 Forbidden Errors**
- Check: API key validity in Bright Data dashboard
- Check: Dataset subscription status
- Check: Account credits/quota

**❌ Profile Not Found**
- Verify: LinkedIn URL is accessible
- Try: Different URL format
- Note: Private profiles cannot be accessed

**❌ "Snapshot processing timed out"**
- Cause: Snapshot took longer than 5 minutes
- Solution: Try with fewer URLs or retry later
- Check: Bright Data service status

**❌ "Snapshot failed" or "Snapshot canceled"**
- Check: LinkedIn URLs are valid and accessible
- Verify: Account has sufficient credits
- Retry: The operation with same or different URLs

### Debug Mode
Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 10. Cost Management

- **Free Tier**: Usually includes limited requests
- **Pay Per Use**: Typical pricing is per successful extraction
- **Batch Discounts**: Process multiple URLs together for better rates
- **Monitor Usage**: Check your Bright Data dashboard regularly

## 11. Best Practices

1. **Validate URLs**: Always validate LinkedIn URLs before processing
2. **Handle Errors**: Implement proper error handling for production use
3. **Batch Processing**: Process multiple profiles together when possible  
4. **Async Awareness**: Expect 1-5 minute processing times
5. **Progress Monitoring**: Use progress callbacks for better UX
6. **Timeout Handling**: Set appropriate timeouts for long operations
7. **Rate Limiting**: Respect API limits to avoid throttling
8. **Data Privacy**: Follow GDPR/privacy regulations
9. **Caching**: Cache results to avoid duplicate requests
10. **Snapshot IDs**: Store snapshot IDs for debugging/tracking

## 12. Files Created

- `src/brightdata_linkedin.py` - Main service implementation
- `examples/brightdata_linkedin_example.py` - Usage examples
- `examples/brightdata_curl_demo.py` - Curl command equivalent
- `test_brightdata.py` - Test suite
- `docs/BRIGHTDATA_INTEGRATION.md` - Detailed documentation

## 13. Processing Times & Expectations

**⏱️ Typical Processing Times:**
- Single profile: 30 seconds - 2 minutes
- 2-5 profiles: 1-3 minutes  
- 5-10 profiles: 2-5 minutes
- 10+ profiles: 3-5+ minutes

**🔄 What Happens During Processing:**
1. Snapshot creation: ~5-10 seconds
2. LinkedIn data collection: 30 seconds - 4 minutes
3. Data parsing and cleanup: ~5-15 seconds

**📱 In the Web App:**
- Real-time progress bar updates
- Live status messages
- Snapshot ID tracking
- Automatic retry on errors

## 14. Support

- **Bright Data Support**: support@brightdata.com
- **API Documentation**: [docs.brightdata.com](https://docs.brightdata.com)
- **Test Your Setup**: Run `python test_brightdata.py`
- **Debug Snapshots**: Check snapshot IDs in Bright Data dashboard

## 15. Next Steps

1. ✅ Set up API key
2. ✅ Test configuration 
3. ✅ Try example scripts (understand async workflow)
4. ✅ Test individual curl commands
5. ✅ Use in Streamlit app (see real-time progress)
6. ✅ Integrate with fraud detection workflow
7. ✅ Monitor processing times and optimize batch sizes

---

**Need Help?** 
- Check the detailed guide: `docs/BRIGHTDATA_INTEGRATION.md`
- Run tests: `python test_brightdata.py`
- Try examples: `python examples/brightdata_linkedin_example.py`

**Ready to Use!** 🎉  
Your Bright Data LinkedIn integration with **async snapshot workflow** is now configured and ready for enhanced fraud detection capabilities.

**Remember**: The async workflow means operations take 1-5 minutes, but provide more reliable and comprehensive data extraction than traditional APIs.