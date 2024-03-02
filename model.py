import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
from llama_index import SimpleDirectoryReader
import openai
import fitz  
import os

st.title("Demo ChatBOT")

#client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
# Set OpenAI API key from app's secrets
openai.api_key = st.secrets.OPENAI_API_KEY

# Set app title
st.header("ChatBOT 💬 📚")

# Initialize chat message history
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Ask me a question about mestural cycle"}]

# Load and index data
@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the  docs – hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert on menstrual cycles and women's health. Your job is to provide accurate and helpful information about menstrual health. Keep your answers factual and supportive, addressing common concerns and providing guidance as needed."))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()

# Create chat engine
chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

# Prompt for user input and display message history
if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Pass query to chat engine and display response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)