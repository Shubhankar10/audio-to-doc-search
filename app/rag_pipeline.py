from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import Ollama


def load_db():
    embedding_model = OllamaEmbeddings(model="deepseek-r1")  # Change model as needed

    # Load the existing vector database
    vector_db = Chroma(persist_directory="./vector_context", embedding_function=embedding_model)

    print("Vector database loaded successfully!")
    return vector_db


def query_llm(vector_db,query):
    # Load the LLM
    llm = Ollama(model="deepseek-r1")

    # Create a RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_db.as_retriever())

    response = qa_chain.run(query)

    print("\nLLM Response in rag_pipeline:")
    print(response)
    return response












vector_db = load_db()

def llm_response(query: str) -> str:
    #Modify the query here 
    response = query_llm(vector_db, query)
    return response
