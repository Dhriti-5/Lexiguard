import requests
import json

print("🧪 Testing FastAPI Server...")
print("-" * 60)

# Test 1: Check if server is running
print("\n✓ Test 1: Server Health Check...")
try:
    response = requests.get("http://localhost:8001")
    if response.status_code == 200:
        print("   ✅ SUCCESS! Server is running!")
        data = response.json()
        print(f"   Message: {data['message']}")
        print(f"   Version: {data['version']}")
    else:
        print(f"   ⚠️  Got status code {response.status_code}")
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    print("   Make sure the FastAPI server is running!")
    exit()

# Test 2: Analyze text
print("\n✓ Test 2: Testing /analyze endpoint...")
contract_data = {
    "text": """
    This Software License Agreement is made on January 15, 2025.
    The Client agrees to pay $15,000 annually for software usage.
    The term is 2 years with automatic renewal unless either party 
    provides 90 days written notice. The software is provided 
    "as-is" without warranty. Liability is limited to fees paid.
    """
}

try:
    print("   Sending contract for analysis...")
    response = requests.post(
        "http://localhost:8001/analyze",
        json=contract_data,
        timeout=30  # Wait up to 30 seconds
    )
    
    if response.status_code == 200:
        result = response.json()
        print("   ✅ SUCCESS! Analysis complete!\n")
        
        print("   📄 RESULTS:")
        print(f"   Document Type: {result['data'].get('document_type', 'N/A')}")
        print(f"   \n   Summary: {result['data'].get('summary', 'N/A')[:200]}...")
        
        key_clauses = result['data'].get('key_clauses', [])
        print(f"\n   📝 Key Clauses ({len(key_clauses)}):")
        for i, clause in enumerate(key_clauses[:3], 1):
            print(f"      {i}. {clause}")
        
        risks = result['data'].get('potential_risks', [])
        print(f"\n   ⚠️  Potential Risks ({len(risks)}):")
        for i, risk in enumerate(risks[:3], 1):
            print(f"      {i}. {risk}")
            
        recommendations = result['data'].get('recommendations', [])
        print(f"\n   💡 Recommendations ({len(recommendations)}):")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"      {i}. {rec}")
    else:
        print(f"   ❌ ERROR: Status code {response.status_code}")
        print(f"   Response: {response.text[:200]}")
except Exception as e:
    print(f"   ❌ ERROR: {e}")

# Test 3: Test clause analysis
print("\n✓ Test 3: Testing /analyze-clauses endpoint...")
clause_data = {
    "text": contract_data["text"],
    "clause_types": ["payment", "termination", "warranty"]
}

try:
    response = requests.post(
        "http://localhost:8001/analyze-clauses",
        json=clause_data,
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ SUCCESS! Found {result.get('total_clauses', 0)} clauses")
    else:
        print(f"   ⚠️  Status code {response.status_code}")
except Exception as e:
    print(f"   ❌ ERROR: {e}")

print("\n" + "-" * 60)
print("🎉 Testing complete!")
print("\n💡 TIP: Visit http://localhost:8001/docs to test all endpoints interactively!")