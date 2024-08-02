import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.agents import create_openai_tools_agent
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.tools.retriever import create_retriever_tool

load_dotenv() 

openai_api_key = os.environ["OPENAI_API_KEY"]

if "vector" not in st.session_state:
    st.session_state.embeddings = OpenAIEmbeddings()
    st.session_state.loader = PyPDFLoader('icc.pdf')
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    st.session_state.final_documents =  st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
    st.session_state.vectors = Chroma.from_documents(st.session_state.final_documents,st.session_state.embeddings)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are very powerful assistant, but don't know current events",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

st.title("Information varification")
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

api_wraper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wraper)
retriever=st.session_state.vectors.as_retriever()
retriever_tool = create_retriever_tool(retriever,"icc_search",
                      "Search for information about ICC. For any information about ICC use this tool")
tools = [wiki,retriever_tool]
agent = create_openai_tools_agent(llm,tools,prompt)
agent_executer = AgentExecutor(agent=agent,tools=tools)
prompt = st.text_input("Input your prompt here")            
                                    
if prompt:
    response = agent_executer.invoke({"input":prompt})
    if response['output']:
        st.success("True")
    else:
        st.success("False")