from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA


def load_db():
    embedding_model = OllamaEmbeddings(model="deepseek-r1")  # Change model as needed

    # Load the existing vector database
    vector_db = Chroma(persist_directory="./vector_context", embedding_function=embedding_model)

    print("Vector database loaded successfully!")
    return vector_db


def query_llm(vector_db,query):
    # Load the LLM
    llm = OllamaLLM(model="deepseek-r1")

    # Create a RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_db.as_retriever())

    response = qa_chain.invoke({'query': query})

    print("\nLLM Response in rag_pipeline:")
    print(response)
    return response


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