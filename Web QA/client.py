import requests

url = "http://127.0.0.1:5001/answer_question"
data = {
    "url": "https://www.deccanherald.com/elections/india/lok-sabha-election-2024-remove-fake-content-within-3-hours-ec-tells-political-parties-3010408",
    "question": "Will Modi win"
}
response = requests.post(url, json=data)
print(response.text)
