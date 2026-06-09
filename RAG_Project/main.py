import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage,HumanMessage
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

    
    system_prompt=(
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retieved context to answer "
    "the question. if you don't know the answer, say that you "
    "don't know. Use three sentence maimum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
    )
    contextualize_q_system_prompt=(
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt=ChatPromptTemplate.from_messages(
    [
    ("system",contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human","{input}")
    ]
    )
    qa_prompt=ChatPromptTemplate.from_messages(
    [
    ("system",system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human","{input}")
    ]
    )

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
                history_aware_retriever=create_history_aware_retriever(llm,retriver,contextualize_q_prompt)
                question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
                chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)
                respose=chain.invoke({"input":prompt1,"chat_history":st.session_state.messages})
            
                with st.chat_message("assistant"):
                   message_placeholder=st.empty()
                   message_placeholder.markdown(respose["answer"])
                   st.session_state.messages.append({"role":"assistant","content":respose["answer"]})
        
            


        
            
        
            




    
           
