import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
import pypdf
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()
api_key=os.getenv("GROQ_API_KEY")


st.title("📄Document Q&A  🤖 Chatbot")

llm=ChatGroq(api_key=api_key,model="llama-3.1-8b-instant")

prompt=ChatPromptTemplate.from_template("""
Answer the following question based on the context only.
please provide the most accurate response based o the question.
<context>
{context}
</context>
Question:{input}


""")

if "messages" and "printed" not in st.session_state:
    st.session_state.messages=[]
    st.session_state.print=False

for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                     st.write(message["content"])

file=st.file_uploader("Choose a PDF file",type="pdf")


def vector_embedding(file):

    if "vectors" not in st.session_state:

        with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as tmp_file:
            tmp_file.write(file.read())
            pdf_path=tmp_file.name
        st.session_state.embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.loader=PyPDFLoader(pdf_path)
        st.session_state.docs=st.session_state.loader.load()
        os.remove(pdf_path)
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_docs=st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings)
        #print(st.session_state.vectors)

if file:
    st.success("File uploaded successfully!")
    if st.button("Create Vector Store"):
        vector_embedding(file)
        st.write("Vector Store DB is Ready")
        st.session_state.printed=True
        

prompt1=st.chat_input("What you want to ask from the document")
        
            
            

if prompt1:
    if not st.session_state.printed:
        st.write("Vector Store DB is Ready")
    st.session_state.printed=False
    st.session_state.messages.append({"role":"user","content":prompt1})
    with st.chat_message("user"):
        st.write(prompt1)
                
    if "vectors" not in st.session_state:
        st.write("⚠️ Please create the vector store first")
                
    else:
        retriver=st.session_state.vectors.as_retriever(search_type="similarity",search_kwargs={"k":4})
        chain=(
            {
                "context":retriver,
                "input":RunnablePassthrough()
            }
            | prompt
            | llm
            )
        respose=chain.invoke(prompt1)
            
        with st.chat_message("assistant"):
            message_placeholder=st.empty()
            message_placeholder.markdown(respose.content)
            st.session_state.messages.append({"role":"assistant","content":respose.content})
        
            


        
            
        
            




    
           
