#!/usr/bin/env python3
"""
Environment Checker for Fraud Resume Detection Tool

This script checks your environment setup and API configurations.
Run this before using the application to ensure everything is configured correctly.

Usage:
    python check_env.py
"""

import os
import sys
from pathlib import Path
import importlib.util

def check_python_version():
    """Check Python version compatibility"""
    print("🐍 Checking Python Version...")

    version = sys.version_info
    if version >= (3, 8):
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} (Compatible)")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} (Requires 3.8+)")
        return False

def check_environment_variables():
    """Check required and optional environment variables"""
    print("\n🔑 Checking Environment Variables...")

    env_status = {}

    # Required variables
    required_vars = {
        'GEMINI_API_KEY': 'Required for AI-powered fraud detection'
    }

    # Optional variables
    optional_vars = {
        'BRIGHTDATA_API_KEY': 'Optional for LinkedIn profile verification',
        'LINKEDIN_API_KEY': 'Optional alternative to Bright Data'
    }

    print("   Required Variables:")
    all_required_set = True
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            masked_value = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
            print(f"   ✅ {var}: {masked_value} ({description})")
            env_status[var] = True
        else:
            print(f"   ❌ {var}: Not set ({description})")
            env_status[var] = False
            all_required_set = False

    print("   Optional Variables:")
    for var, description in optional_vars.items():
        value = os.getenv(var)
        if value:
            masked_value = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
            print(f"   ✅ {var}: {masked_value} ({description})")
            env_status[var] = True
        else:
            print(f"   ⚪ {var}: Not set ({description})")
            env_status[var] = False

    return env_status, all_required_set

def check_file_structure():
    """Check if required files and directories exist"""
    print("\n📁 Checking File Structure...")

    required_files = [
        'app.py',
        'config.py',
        'requirements.txt',
        'src/fraud_detector.py',
        'src/nlp_analyzer.py',
        'src/brightdata_linkedin_fixed.py',
        'src/utils.py'
    ]

    optional_files = [
        'src/linkedin_verifier.py',
        'src/fit_scorer.py',
        'src/report_generator.py'
    ]

    required_dirs = [
        'src',
        'templates',
        'data'
    ]

    print("   Required Files:")
    all_files_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} (Missing)")
            all_files_exist = False

    print("   Optional Files:")
    for file_path in optional_files:
        if Path(file_path).exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ⚪ {file_path} (Not found, but optional)")

    print("   Required Directories:")
    for dir_path in required_dirs:
        if Path(dir_path).exists() and Path(dir_path).is_dir():
            print(f"   ✅ {dir_path}/")
        else:
            print(f"   ❌ {dir_path}/ (Missing)")
            all_files_exist = False

    return all_files_exist

def check_dependencies():
    """Check if required Python packages are installed"""
    print("\n📦 Checking Dependencies...")

    required_packages = {
        'streamlit': 'Web framework for the application',
        'requests': 'HTTP requests for API calls',
        'pandas': 'Data manipulation and analysis',
        'numpy': 'Numerical computations',
        'python-dotenv': 'Environment variable loading'
    }

    optional_packages = {
        'google-genai': 'Google Gemini AI integration',
        'nltk': 'Natural language processing',
        'spacy': 'Advanced NLP features',
        'plotly': 'Interactive visualizations'
    }

    print("   Required Packages:")
    all_required_installed = True
    for package, description in required_packages.items():
        try:
            __import__(package.replace('-', '_'))
            print(f"   ✅ {package}: Installed ({description})")
        except ImportError:
            print(f"   ❌ {package}: Not installed ({description})")
            all_required_installed = False

    print("   Optional Packages:")
    for package, description in optional_packages.items():
        try:
            __import__(package.replace('-', '_'))
            print(f"   ✅ {package}: Installed ({description})")
        except ImportError:
            print(f"   ⚪ {package}: Not installed ({description})")

    return all_required_installed

def check_module_imports():
    """Test importing main modules"""
    print("\n🔧 Testing Module Imports...")

    modules_to_test = [
        ('config', 'Configuration module'),
        ('src.fraud_detector', 'Fraud detection engine'),
        ('src.nlp_analyzer', 'NLP analysis module'),
        ('src.utils', 'Utility functions')
    ]

    optional_modules = [
        ('src.brightdata_linkedin_fixed', 'Bright Data LinkedIn service'),
        ('src.linkedin_verifier', 'Traditional LinkedIn verifier'),
        ('src.fit_scorer', 'Job fit scoring'),
        ('src.report_generator', 'Report generation')
    ]

    print("   Core Modules:")
    all_core_imported = True
    for module_name, description in modules_to_test:
        try:
            spec = importlib.util.find_spec(module_name)
            if spec is not None:
                print(f"   ✅ {module_name}: Can import ({description})")
            else:
                print(f"   ❌ {module_name}: Cannot find ({description})")
                all_core_imported = False
        except Exception as e:
            print(f"   ❌ {module_name}: Import error - {e}")
            all_core_imported = False

    print("   Optional Modules:")
    for module_name, description in optional_modules:
        try:
            spec = importlib.util.find_spec(module_name)
            if spec is not None:
                print(f"   ✅ {module_name}: Can import ({description})")
            else:
                print(f"   ⚪ {module_name}: Cannot find ({description})")
        except Exception as e:
            print(f"   ⚪ {module_name}: Import error - {e}")

    return all_core_imported

def test_api_connectivity():
    """Test API connectivity if keys are available"""
    print("\n🌐 Testing API Connectivity...")

    # Test Gemini API
    gemini_key = os.getenv('GEMINI_API_KEY')
    if gemini_key:
        print("   Testing Gemini API...")
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            # Just configure, don't make actual calls in env check
            print("   ✅ Gemini API: Key configured successfully")
        except Exception as e:
            print(f"   ⚠️ Gemini API: Configuration issue - {e}")
    else:
        print("   ⚪ Gemini API: No key provided, skipping test")

    # Test Bright Data API
    brightdata_key = os.getenv('BRIGHTDATA_API_KEY')
    if brightdata_key:
        print("   Testing Bright Data API...")
        try:
            import requests
            headers = {'Authorization': f'Bearer {brightdata_key}'}
            # Don't make actual request in env check, just validate format
            if len(brightdata_key) > 20:
                print("   ✅ Bright Data API: Key format looks valid")
            else:
                print("   ⚠️ Bright Data API: Key format seems short")
        except Exception as e:
            print(f"   ⚠️ Bright Data API: Issue - {e}")
    else:
        print("   ⚪ Bright Data API: No key provided, skipping test")

def provide_setup_recommendations(env_status, files_ok, deps_ok, imports_ok):
    """Provide setup recommendations based on check results"""
    print("\n💡 Setup Recommendations:")

    if not env_status.get('GEMINI_API_KEY', False):
        print("\n   🔴 CRITICAL: Missing Gemini API Key")
        print("      1. Get a free API key from: https://makersuite.google.com/app/apikey")
        print("      2. Set it: export GEMINI_API_KEY='your_key_here'")
        print("      3. Or create .env file with: GEMINI_API_KEY=your_key_here")

    if not env_status.get('BRIGHTDATA_API_KEY', False):
        print("\n   🟡 OPTIONAL: Missing Bright Data API Key")
        print("      For LinkedIn verification:")
        print("      1. Sign up at: https://brightdata.com")
        print("      2. Subscribe to LinkedIn dataset: gd_l1viktl72bvl7bjuj0")
        print("      3. Set it: export BRIGHTDATA_API_KEY='your_key_here'")
        print("      4. Or skip LinkedIn verification in the app")

    if not files_ok:
        print("\n   🔴 CRITICAL: Missing required files")
        print("      Make sure you're running from the project root directory")
        print("      All src/ files should be present")

    if not deps_ok:
        print("\n   🔴 CRITICAL: Missing required dependencies")
        print("      Run: pip install -r requirements.txt")

    if not imports_ok:
        print("\n   🔴 CRITICAL: Module import issues")
        print("      Check file structure and dependencies")

def main():
    """Main environment check function"""
    print("🔍 FRAUD RESUME DETECTION TOOL - ENVIRONMENT CHECK")
    print("=" * 60)

    # Run all checks
    python_ok = check_python_version()
    env_status, required_env_ok = check_environment_variables()
    files_ok = check_file_structure()
    deps_ok = check_dependencies()
    imports_ok = check_module_imports()

    # Test APIs if basic setup is OK
    if required_env_ok and deps_ok:
        test_api_connectivity()

    # Overall status
    print("\n" + "=" * 60)
    print("📊 OVERALL STATUS")
    print("=" * 60)

    checks = {
        "Python Version": python_ok,
        "Required Environment Variables": required_env_ok,
        "File Structure": files_ok,
        "Dependencies": deps_ok,
        "Module Imports": imports_ok
    }

    all_critical_ok = all(checks.values())

    for check_name, status in checks.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {check_name}")

    # Feature availability
    print(f"\n🎯 FEATURE AVAILABILITY:")
    if all_critical_ok:
        print("✅ Core fraud detection: Available")
        if env_status.get('BRIGHTDATA_API_KEY', False):
            print("✅ LinkedIn verification: Available")
        else:
            print("⚠️ LinkedIn verification: Not available (missing API key)")

        if env_status.get('LINKEDIN_API_KEY', False):
            print("✅ Traditional LinkedIn API: Available")
        else:
            print("⚪ Traditional LinkedIn API: Not configured")
    else:
        print("❌ Application may not function properly")

    # Recommendations
    provide_setup_recommendations(env_status, files_ok, deps_ok, imports_ok)

    # Final verdict
    print(f"\n" + "=" * 60)
    if all_critical_ok:
        print("🎉 ENVIRONMENT CHECK PASSED!")
        print("Your setup is ready. You can now run:")
        print("   streamlit run app.py")
    else:
        print("⚠️ ENVIRONMENT CHECK FAILED")
        print("Please fix the issues above before running the application.")
        return 1

    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
