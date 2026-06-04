import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
import pypdf
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()
os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

st.set_page_config(page_title="Intelligent Document Assistant",page_icon="🤖")
st.title("📄Document Q&A  🤖 Chatbot")

api_key=st.text_input("Enter your Groq API key:",type="password")

if api_key:
    llm=ChatGroq(api_key=api_key,model="llama-3.1-8b-instant")

    prompt=ChatPromptTemplate.from_template("""
    Answer the following question based on the context only.
    please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Question:{input}


    """)

    if "messages" not in st.session_state:
        st.session_state.messages=[]

    if "printed" not in st.session_state:
        st.session_state.printed=False



    for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                     st.write(message["content"])

    file=st.file_uploader("Choose a PDF file",type="pdf",accept_multiple_files=False,)


    def vector_embedding(file):

       

            with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as tmp_file:
              tmp_file.write(file.read())
              pdf_path=tmp_file.name
            if "path" in st.session_state and st.session_state.path==pdf_path:
                return
            st.session_state.path=pdf_path
            loader=PyPDFLoader(pdf_path)
            docs=loader.load()
            os.remove(pdf_path)
            text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
            final_docs=text_splitter.split_documents(docs)
            st.session_state.vectors=FAISS.from_documents(final_docs,embeddings)
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
                st.warning("⚠️ Please upload a PDF and create the vector store first")
                
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
        
            


        
            
        
            




    
           
