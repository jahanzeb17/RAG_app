from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')


# load documents
loader = PyPDFLoader("attention.pdf")
docs = loader.load()



# split documents
spliter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks = spliter.split_documents(docs)

# vector store
embedding = OpenAIEmbeddings()
vectordb = FAISS.from_documents(chunks,embedding)


prompt = ChatPromptTemplate.from_template('''
Answer the following questions based only on the provided context.
please think step by step before providing the answer
<context>
{context}
<context>
Question:{input} ''')


llm = ChatGroq(model_name='Gemma-7b-It')
# llm = ChatOpenAI(model='gpt-3.5-turbo')

document_chain = create_stuff_documents_chain(llm,prompt)

retriever = vectordb.as_retriever()
st.spinner("processing")

retrieval_chain = create_retrieval_chain(retriever,document_chain)

input_text = st.text_input("Enter your query")

if input_text:

    response = retrieval_chain.invoke({"input":input_text})
    st.write(response['answer'])