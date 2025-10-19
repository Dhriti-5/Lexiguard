print("🚀 Starting LexiGuard SDK Test...")
print("-" * 50)

# Test 1: Check if we can import the SDK
print("\n✓ Test 1: Importing SDK...")
try:
    from lexiguard_sdk import LexiGuard
    print("   SUCCESS! SDK imported.")
except Exception as e:
    print(f"   ERROR: {e}")
    print("   Make sure you're running from the lexiguard folder!")
    exit()

# Test 2: Check API key
print("\n✓ Test 2: Checking API key...")
import os
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("   ERROR: API key not found!")
    print("   Did you run the 'set' or 'export' command?")
    api_key = input("\n   Paste your API key here: ").strip()
    os.environ["GEMINI_API_KEY"] = api_key

print(f"   API key found: {api_key[:10]}...")

# Test 3: Initialize SDK
print("\n✓ Test 3: Initializing LexiGuard...")
try:
    lg = LexiGuard(api_key=api_key)
    print("   SUCCESS! LexiGuard initialized.")
except Exception as e:
    print(f"   ERROR: {e}")
    exit()

# Test 4: Analyze simple text
print("\n✓ Test 4: Analyzing sample contract...")
sample_text = """
This is a simple employment agreement between ABC Corp and John Doe.
The employee will work full-time for $80,000 per year.
The term is 2 years with 30 days notice for termination.
"""

try:
    result = lg.analyze_text(sample_text)
    
    if result["success"]:
        print("   SUCCESS! Analysis complete!")
        print("\n📄 Results:")
        print(f"   Document Type: {result['data'].get('document_type', 'N/A')}")
        print(f"   Summary: {result['data'].get('summary', 'N/A')[:100]}...")
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"   ERROR: {result.get('error', 'Unknown error')}")
except Exception as e:
    print(f"   ERROR: {e}")
    
print("\n" + "-" * 50)
print("Test complete!")