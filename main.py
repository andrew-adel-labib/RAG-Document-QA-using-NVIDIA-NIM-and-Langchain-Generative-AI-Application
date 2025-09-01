import os
import time 
import streamlit as st 
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS

load_dotenv()

os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")

llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.pdf_loader = PyPDFDirectoryLoader("./us_census")
        st.session_state.docs = st.session_state.pdf_loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

st.title("RAG Document Q&A using NVIDIA NIM and Langchain")

prompt=ChatPromptTemplate.from_template(
"""
Answer the question based on the provided context only.
Please provide the most accurate response.
<context>
{context}
</context>
Question: {input}
"""
)

user_input = st.text_input("Enter your question from the documents.")

if st.button("Document Embedding"):
    vector_embedding()
    st.write("Vector Store DB is Ready.")

if user_input:
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({'input': user_input})
    print("Response Time : ", time.process_time() - start)
    st.write(response["answer"])
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")