"""
Test script to verify PII Redaction feature is working correctly
This tests the DLP API integration and UI feedback
"""

import requests
import json

# Test configuration
BACKEND_URL = "http://localhost:8000"

# Sample contract with PII data
SAMPLE_CONTRACT_WITH_PII = """
EMPLOYMENT AGREEMENT

This Employment Agreement is made on January 15, 2024, between:
Employer: Tech Solutions Inc.
Employee: John Michael Smith
Address: 123 Main Street, San Francisco, CA 94102
Phone: (555) 123-4567
Email: john.smith@email.com
Social Security Number: 123-45-6789
Date of Birth: March 10, 1990

TERMS AND CONDITIONS:
1. Position: Senior Software Engineer
2. Salary: $120,000 per year
3. Start Date: February 1, 2024
4. Credit Card for expenses: 4532-1234-5678-9010

Payment details:
Account: 9876543210
Routing: 123456789

Emergency Contact: Jane Smith at (555) 987-6543
"""

def test_pii_redaction():
    """Test if PII redaction is working through the API"""
    
    print("=" * 70)
    print("TESTING PII REDACTION FEATURE")
    print("=" * 70)
    print()
    
    # Test 1: Check if backend is running
    print("Test 1: Checking if backend is running...")
    try:
        response = requests.get(f"{BACKEND_URL}/")
        if response.status_code == 200:
            print("✅ Backend is running")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Backend returned status code: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to backend: {e}")
        print("   Please start the backend first: cd lexiguard-backend && python main.py")
        return
    
    print()
    
    # Test 2: Test Standard Analysis with PII
    print("Test 2: Testing Standard Analysis (/analyze-file) with PII data...")
    try:
        response = requests.post(
            f"{BACKEND_URL}/analyze-file",
            data={"text": SAMPLE_CONTRACT_WITH_PII}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Standard analysis completed")
            print(f"   File Type: {result.get('file_type', 'N/A')}")
            print(f"   PII Redacted: {result.get('pii_redacted', False)}")
            print(f"   Privacy Notice: {result.get('privacy_notice', 'N/A')}")
            print(f"   Number of Risks: {len(result.get('risks', []))}")
            
            if result.get('pii_redacted'):
                print("\n   🎉 PII REDACTION IS WORKING!")
            else:
                print("\n   ⚠️  PII was not redacted (DLP may not be configured)")
                print("   Check if GOOGLE_CLOUD_PROJECT environment variable is set")
        else:
            print(f"❌ Analysis failed with status code: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Error during standard analysis: {e}")
    
    print()
    
    # Test 3: Test Detailed Clause Analysis with PII
    print("Test 3: Testing Detailed Clause Analysis (/analyze-clauses) with PII data...")
    try:
        response = requests.post(
            f"{BACKEND_URL}/analyze-clauses",
            data={"text": SAMPLE_CONTRACT_WITH_PII}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Detailed clause analysis completed")
            print(f"   File Type: {result.get('file_type', 'N/A')}")
            print(f"   PII Redacted: {result.get('pii_redacted', False)}")
            print(f"   Total Risky Clauses: {result.get('total_risky_clauses', 0)}")
            
            if result.get('pii_redacted'):
                print("\n   🎉 PII REDACTION IS WORKING IN DETAILED ANALYSIS!")
            else:
                print("\n   ⚠️  PII was not redacted (DLP may not be configured)")
        else:
            print(f"❌ Detailed analysis failed with status code: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Error during detailed analysis: {e}")
    
    print()
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print()
    print("Expected Behavior:")
    print("1. ✅ Backend should be running and accessible")
    print("2. ✅ Both endpoints should return pii_redacted: true when DLP is configured")
    print("3. ✅ Frontend should display privacy badge when pii_redacted is true")
    print()
    print("If PII redaction is not working, check:")
    print("- GOOGLE_CLOUD_PROJECT environment variable is set in .env file")
    print("- Google Cloud credentials are configured (Application Default Credentials)")
    print("- DLP API is enabled in your Google Cloud Project")
    print("- google-cloud-dlp package is installed")
    print()

if __name__ == "__main__":
    test_pii_redaction()
