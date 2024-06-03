import os
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI
import openai


openai.api_key = "YOUR-API-KEY-HERE"

# Check if vector embedding storage location already exists
STORAGE_DIR = "./embedding_storage"
if not os.path.exists(STORAGE_DIR):
    os.path.dirname(STORAGE_DIR)

st.title("Retrieval-Augmented Generation Chat App")

st.sidebar.markdown("# Configuration")

# Choose foundational models
model_options = ("gpt-3.5-turbo", "gpt-4")
model = st.sidebar.selectbox("foundational model", model_options, index=None, placeholder="Choose a model")
st.sidebar.success(f"{model} loaded successfully.")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assisstant", "content": "Ask me any question, Federika-Palooza!"}
    ]

# Upload files with streamlit
st.sidebar.file_uploader("Upload file(s)", type=["pdf"], accept_multiple_files=True)

# Build an index over your external/private documents
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

# Store the vector embeddings to disk
index.storage_context.persist(persist_dir=STORAGE_DIR)

# Create model
if model != None:
    service_context = ServiceContext.from_defaults(llm=OpenAI(model=model))

# Prompt for user input and save to chat history
if prompt := st.chat_input("Write your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Create a chat engine
chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

# If last message is not from assisstant, generate a new response
if st.session_state.messages[-1]["role"] != "assisstant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            # Add response to message history
            st.session_state.messages.append(message)
