import requests

url = "http://127.0.0.1:5000/translate"
data = {"sentences": ["constitution"]}  # Update the key and structure
response = requests.post(url, json=data)

print("Response:", response.json())
