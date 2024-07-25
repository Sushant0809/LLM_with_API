from flask import Flask, request
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


app = Flask(__name__)

llm_answer_path = "HuggingFaceH4/zephyr-7b-beta"
torch_device = "cuda:7"
# API_KEY = 'Use your API key'

@app.route('/answer_question', methods=['POST'])
def answer_question():
    req_data = request.get_json()
    format = req_data['format']
    topic = req_data['topic']
    emotions = req_data['emotions']
    length = int(req_data['length'])
    
    query = f"""
        write a marketing content for {format} 

        on {topic}, 

        emotions should be {emotions},
        
        and it should not exeed by {length} words
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
                                        max_new_tokens=length, do_sample=True,
                                        temperature=0.9,
                            )
    

    answer = tokenizer.decode(greedy_output[0], skip_special_tokens=True)

    
    return answer[len(query):]
   


if __name__ == '__main__':
    app.run(debug=True, port=5001)
