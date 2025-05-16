from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA


def load_db():
    embedding_model = OllamaEmbeddings(model="deepseek-r1")  # Change model as needed

    # Load the existing vector database
    vector_db = Chroma(persist_directory="./vector_context", embedding_function=embedding_model)

    print("Vector database loaded successfully!")
    return vector_db


def query_llm(vector_db, query):
    # Load the LLM
    llm = OllamaLLM(model="deepseek-r1")
    # Create a RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_db.as_retriever())
    response = qa_chain.invoke({'query': query})
    # Remove <think>...</think> if present in the response
    import re
    if isinstance(response, dict) and "result" in response:
        result = response["result"]
    else:
        result = response
    # Remove <think>...</think> tags and their content
    result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()
    print("\nLLM Response in rag_pipeline:")
    print(result)
    return result


vector_db = load_db()

def llm_response_finance(query: str) -> str:
    # Add a financial knowledge prompt to guide the LLM
    system_prompt = (
        "You are a financial expert. Use the information from the provided documents and your financial knowledge to answer the following question as accurately and concisely as possible. "
        "If the answer is not present in the documents, say so.\n\nQuestion: "
    )
    full_prompt = system_prompt + query
    response = query_llm(vector_db, full_prompt)
    return response


def llm_response_sit(query: str) -> str:
    # Add a SIT knowledge prompt to guide the LLM
    system_prompt = (
        "You are an expert on the Singapore Institute of Technology (SIT). Use the information from the provided documents and your knowledge to answer the following question as accurately and concisely as possible. "
        "If the answer is not present in the documents, say so.\n\nQuestion: "
    )
    full_prompt = system_prompt + query
    response = query_llm(vector_db, full_prompt)
    return response

# For medical debate
def llm_response_medical_debate(query: str, debate_side: str = "for", debate_round: int = 1) -> str:
    if "Topic:" in query and "User's opening argument:" in query:
        parts = query.split("User's opening argument:")
        topic_part = parts[0].strip()
        topic = topic_part.replace("Topic:", "").strip()
        user_argument = parts[1].strip()
        
        system_prompt = (
            f"You are participating in a formal debate on medical topics. You are arguing {debate_side.upper()} "
            f"the following medical proposition: '{topic}'. "
            f"Respond to your opponent's opening argument. "
            f"Be persuasive, logical, and cite medical evidence when possible. "
            f"Keep your response concise (100-150 words). Sound like a confident medical professional in a debate.\n\n"
            f"Your opponent just said: {user_argument}\n\n"
            f"Your response: "
        )
        
        full_prompt = system_prompt
    else:
        system_prompt = (
            f"You are participating in a formal debate on medical topics. You are arguing {debate_side.upper()} "
            f"the proposition. This is round {debate_round}. "
            f"Address your opponent's previous arguments while advancing your position. "
            f"Be persuasive, logical, and cite medical evidence when possible. "
            f"Keep your response concise (100-150 words). Sound like a confident medical professional in a debate.\n\n"
            f"Your opponent just said: {query}\n\n"
            f"Your response: "
        )
        
        full_prompt = system_prompt
    
    if vector_db:
        response = query_llm(vector_db, full_prompt)
    else:
        llm = OllamaLLM(model="deepseek-r1")
        response = llm.invoke(full_prompt)
        
    return response
