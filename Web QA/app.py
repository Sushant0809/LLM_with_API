from flask import Flask, request
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

llm_answer_path = "HuggingFaceH4/zephyr-7b-beta"
torch_device = "cuda:7"
API_KEY = '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'

@app.route('/answer_question', methods=['POST'])
def answer_question():
    req_data = request.get_json()
    url = req_data['url']
    question = req_data['question']
    
    # Scrape webpage content
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    text = ' '.join([p.get_text() for p in soup.find_all('p')])
    text

    query = f"""
    answer the following question,

    based on your knowledge and the provided context. 
    Keep the answer concise.

    question: {question}

    context: {text}
    """
    # headers = {
    #         'Content-Type': 'application/json',
    #         'Authorization': 'Bearer ' + API_KEY,
    #     }
    
    
    tokenizer = AutoTokenizer.from_pretrained(llm_answer_path)
    llm_answer = AutoModelForCausalLM.from_pretrained(llm_answer_path,
                                                  device_map=torch_device,
                                                  torch_dtype=torch.float16)#,
                                                #   headers = headers)

    input_ids = tokenizer.encode(query+"\n\nANSWER:", 
                                return_tensors='pt', return_attention_mask=False).to(torch_device)
    greedy_output = llm_answer.generate(input_ids, 
                                        max_new_tokens=250, do_sample=True,
                                        temperature=0.4,
                                       top_k=100,
                                       top_p=0.9
    )
    answer = tokenizer.decode(greedy_output[0], skip_special_tokens=True)

    # TF-IDF for semantic similarity
    vectorizer = TfidfVectorizer()
    context_vec = vectorizer.fit_transform([text])
    answer_vec = vectorizer.transform([answer[len(query):]])
    # answer_vec = vectorizer.transform([question])

    # Calculate cosine similarity
    similarity = (context_vec * answer_vec.T).toarray()[0][0]


    # Threshold for relevance
    similarity_threshold = 0.65 # Adjust threshold as needed

    print(similarity)

    if similarity >= similarity_threshold:
        return answer[len(query):]
    else:
        return "I don't know the answer"



if __name__ == '__main__':
    app.run(debug=True, port=5001)
