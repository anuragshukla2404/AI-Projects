import streamlit as st
import os
from langchain_groq ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_objectbox.vectorstores import ObjectBox
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv['OPENAI_API_KEY']
groq_api_key = os.getenv['GROQ_API_KEY']

st.title("Objectbox VectorstoreDB with Llama3 Demo")

llm = ChatGroq(groq_api_key=groq_api_key,
               model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

def vector_embeddings():

    if "vectors" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./us-census")
        st.session_state.docs = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
        st.session_state.vectors = ObjectBox.from_documents(st.session_state.final_documents,st.session_state.embeddings,vector_embeddings=768)


input_prompt = st.text_input("Enter your question from documents")

if st.button("Documents Embedding"):
    vector_embeddings()
    st.write("ObjectBox Database is ready")

import time

if input_prompt:
    document_chain = create_stuff_documents_chain(llm,prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever,document_chain)
    start = time.process_time()

    response = retrieval_chain.invoke("input":input_prompt)

    print("Response time :", time.process_time()-start)
    st.write(response['answer'])

    with st.expander("Document Similariy Search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("----------------------------")