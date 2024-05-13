from flask import Flask, request
import requests
import llama_index
import torch
from llama_index.core import VectorStoreIndex,ServiceContext
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.core import download_loader
from llama_index.core import Settings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core import ServiceContext
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor


app = Flask(__name__)

@app.route('/answer_question', methods=['POST'])
def answer_question():
    req_data = request.get_json()
    url = req_data['url']
    query = req_data['question']

    BeautifulSoupWebReader = download_loader("BeautifulSoupWebReader")
    loader = BeautifulSoupWebReader()
    documents = loader.load_data(urls=[url])

    system_prompt="""
    You are a Q&A assistant. Your goal is to answer questions as
    accurately as possible based on the instructions and context provided.
    """
    query_wrapper_prompt=SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")


    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=128,
        generate_kwargs={"temperature": 0.25, "do_sample": False},
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
        model_name="meta-llama/Llama-2-7b-chat-hf",
        device_map="cuda:1",
        tokenizer_kwargs={"max_length": 2048},
        model_kwargs={"torch_dtype": torch.float16 , 
                    "load_in_8bit":True}
    )

    Settings.chunk_size = 512
    Settings.llm = llm

    embed_model=LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))
    
    service_context=ServiceContext.from_defaults(
        chunk_size=1024,
        llm=llm,
        embed_model=embed_model
    )

   
    index=VectorStoreIndex.from_documents(documents,service_context=service_context, show_progress=True)

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=10,
    )

    response_synthesizer = get_response_synthesizer()

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
    )

    response=query_engine.query(query)

    if response.response == 'Empty Response':
        return 'I don’t know the answer'
    else:
        return response.response
        
   
if __name__ == '__main__':
    app.run(debug=True, port=5001)
