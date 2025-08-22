#!/usr/bin/env python3
"""
Gemini Analyzer Example Script

This script demonstrates how to use the updated GeminiAnalyzer with the new
Google GenAI client pattern and environment variable configuration.

Usage:
    1. Set your GEMINI_API_KEY environment variable
    2. Run: python examples/gemini_analyzer_example.py

Requirements:
    - google-genai>=0.5.0
    - python-dotenv>=1.0.0 (for .env file support)
"""

import os
import sys
import time
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file
except ImportError:
    print("python-dotenv not available. Make sure to set GEMINI_API_KEY environment variable.")

from gemini_analyzer import GeminiAnalyzer, AnalysisType, RiskLevel


def main():
    """Main example function demonstrating Gemini analyzer usage"""

    print("🔍 Gemini AI Fraud Detection Example")
    print("=" * 50)

    # Check if API key is available
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ Error: GEMINI_API_KEY environment variable not set!")
        print("\nTo fix this:")
        print("1. Get your API key from https://ai.google.dev/")
        print("2. Set the environment variable:")
        print("   export GEMINI_API_KEY='your-api-key-here'")
        print("3. Or add it to a .env file in the project root")
        return

    print(f"✅ API key found: {api_key[:10]}...")

    try:
        # Initialize the analyzer with custom configuration
        config = {
            'gemini_model': 'gemini-2.5-flash',  # Updated model
            'gemini_temperature': 0.3,
            'gemini_max_tokens': 2000,
            'cache_enabled': True,
            'gemini_rate_limit': 60
        }

        print("\n🚀 Initializing Gemini Analyzer...")
        analyzer = GeminiAnalyzer(config=config)
        print("✅ Analyzer initialized successfully!")

        # Sample resume text for analysis
        sample_resume = """
        John Smith
        Senior Software Engineer

        Experience:
        • Senior Software Engineer at Google (2020-2023) - Led a team of 50+ engineers
        • Lead Developer at Facebook (2018-2020) - Managed entire infrastructure
        • CEO and Founder at TechStartup Inc (2015-2018) - Built company from ground up
        • Senior Data Scientist at Amazon (2013-2015) - Developed ML algorithms

        Education:
        • PhD in Computer Science from MIT (2010-2013)
        • Masters in AI from Stanford (2008-2010)
        • Bachelor's in Engineering from Harvard (2004-2008)

        Skills:
        • Expert in Python, Java, C++, JavaScript, Go, Rust
        • Machine Learning, Deep Learning, AI, Blockchain
        • Cloud platforms: AWS, GCP, Azure
        • Led teams of 100+ people
        """

        print("\n📄 Analyzing sample resume...")
        print("Resume Preview:", sample_resume[:100] + "...")

        # Collect results for summary
        analysis_results = []

        # Example 1: Experience Fraud Analysis
        print("\n🔍 Running Experience Fraud Analysis...")
        start_time = time.time()

        try:
            experience_result = analyzer.analyze_experience_fraud(sample_resume)

            print(f"✅ Analysis completed in {time.time() - start_time:.2f}s")
            print_analysis_result("Experience Analysis", experience_result)
            analysis_results.append(experience_result)

        except Exception as e:
            print(f"❌ Experience analysis failed: {e}")

        # Example 2: Education Fraud Analysis
        print("\n🎓 Running Education Fraud Analysis...")
        start_time = time.time()

        try:
            education_result = analyzer.analyze_education_fraud(sample_resume)

            print(f"✅ Analysis completed in {time.time() - start_time:.2f}s")
            print_analysis_result("Education Analysis", education_result)
            analysis_results.append(education_result)

        except Exception as e:
            print(f"❌ Education analysis failed: {e}")

        # Example 3: Skills Fraud Analysis
        print("\n💻 Running Skills Fraud Analysis...")
        start_time = time.time()

        try:
            skills_result = analyzer.analyze_skills_fraud(sample_resume)

            print(f"✅ Analysis completed in {time.time() - start_time:.2f}s")
            print_analysis_result("Skills Analysis", skills_result)
            analysis_results.append(skills_result)

        except Exception as e:
            print(f"❌ Skills analysis failed: {e}")

        # Example 4: Comprehensive Analysis
        print("\n🔬 Running Comprehensive Fraud Analysis...")
        start_time = time.time()

        try:
            comprehensive_result = analyzer.comprehensive_fraud_analysis(sample_resume)

            print(f"✅ Analysis completed in {time.time() - start_time:.2f}s")
            print_analysis_result("Comprehensive Analysis", comprehensive_result)
            analysis_results.append(comprehensive_result)

        except Exception as e:
            print(f"❌ Comprehensive analysis failed: {e}")

        # Example 5: Demonstrate caching
        print("\n⚡ Demonstrating Cache Performance...")
        print("Running same analysis again (should use cache)...")

        start_time = time.time()
        cached_result = analyzer.analyze_experience_fraud(sample_resume)
        cache_time = time.time() - start_time

        print(f"✅ Cached analysis completed in {cache_time:.2f}s")
        print(f"📊 Cache speedup demonstrated!")

        # Display analysis summary
        print("\n📊 Analysis Summary")
        print("=" * 30)
        if analysis_results:
            summary = analyzer.get_analysis_summary(analysis_results)
            for key, value in summary.items():
                print(f"{key}: {value}")
        else:
            print("❌ No analysis results available for summary")

    except Exception as e:
        print(f"❌ Error initializing analyzer: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check your GEMINI_API_KEY is valid")
        print("2. Ensure you have internet connectivity")
        print("3. Verify google-genai package is installed: pip install google-genai")
        return

    print("\n🎉 Example completed successfully!")
    print("\nNext steps:")
    print("• Try modifying the sample resume to see different results")
    print("• Experiment with different analysis types")
    print("• Integrate this into your own applications")


def print_analysis_result(title: str, result):
    """Print formatted analysis result"""
    print(f"\n📋 {title} Results:")
    print("-" * 40)
    print(f"🔴 Risk Level: {result.risk_level.value.upper()}")
    print(f"📊 Confidence: {result.confidence:.2%}")
    print(f"⚡ Processing Time: {result.processing_time:.2f}s")

    print("\n🔍 Key Findings:")
    for i, finding in enumerate(result.findings, 1):
        print(f"   {i}. {finding}")

    print("\n💡 Recommendations:")
    for i, recommendation in enumerate(result.recommendations, 1):
        print(f"   {i}. {recommendation}")

    if result.evidence:
        print("\n📋 Evidence Summary:")
        for key, value in result.evidence.items():
            print(f"   • {key}: {value}")


def demonstrate_configuration_options():
    """Show different configuration options"""
    print("\n⚙️ Configuration Examples:")
    print("=" * 30)

    # High precision configuration
    high_precision_config = {
        'gemini_model': 'gemini-2.5-pro',  # More powerful model
        'gemini_temperature': 0.1,         # Lower temperature for consistency
        'gemini_max_tokens': 4000,         # More detailed responses
        'cache_enabled': True,
        'gemini_rate_limit': 30            # Conservative rate limiting
    }

    print("🎯 High Precision Configuration:")
    for key, value in high_precision_config.items():
        print(f"   {key}: {value}")

    # Fast processing configuration
    fast_config = {
        'gemini_model': 'gemini-2.5-flash',  # Fastest model
        'gemini_temperature': 0.5,           # Balanced creativity
        'gemini_max_tokens': 1000,           # Shorter responses
        'cache_enabled': True,
        'gemini_rate_limit': 120             # Higher rate limit
    }

    print("\n⚡ Fast Processing Configuration:")
    for key, value in fast_config.items():
        print(f"   {key}: {value}")


if __name__ == "__main__":
    main()

    # Uncomment to see configuration examples
    # demonstrate_configuration_options()
