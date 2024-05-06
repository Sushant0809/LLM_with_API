import requests

url = "http://127.0.0.1:5001/answer_question"
data = {
    "format":  "linkedin post",
    "topic": "Generative AI",
    "emotions" : "Influencing",
    "length" : "250"
}

response = requests.post(url, json=data)
print(response.text)
