import requests
import json

# Test the API
url = 'http://localhost:5001/api/recommend'
data = {
    'nitrogen': 90,
    'phosphorus': 42,
    'potassium': 43,
    'temperature': 20.8,
    'humidity': 82,
    'ph': 6.5,
    'rainfall': 202.9
}

try:
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")
    print(f"Response text: {response.text if 'response' in locals() else 'No response'}")
