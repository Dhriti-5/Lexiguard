import requests

print("📁 Testing File Upload...")

file_path = "sample_contract.txt"

try:
    with open(file_path, "rb") as f:
        files = {"file": (file_path, f, "text/plain")}
        response = requests.post(
            "http://localhost:8001/analyze-file",
            files=files,
            timeout=30
        )
    
    if response.status_code == 200:
        result = response.json()
        print("✅ File analyzed successfully!\n")
        print(f"File: {result.get('file_info', {}).get('file_name')}")
        print(f"Type: {result.get('file_info', {}).get('file_type')}")
        print(f"Document: {result['data'].get('document_type')}")
        print(f"\nSummary: {result['data'].get('summary')[:150]}...")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        
except FileNotFoundError:
    print(f"❌ File not found: {file_path}")
    print("Create the sample_contract.txt file first!")
except Exception as e:
    print(f"❌ Error: {e}")