import streamlit as st
from langchain.callbacks import get_openai_callback
from config import *
from utils import (get_text,get_text_chunks, 
                get_vectorstore,
                get_conversation_chain, 
                StreamlitChatMessageHistory)

st.set_page_config(
page_title="주식 gpt",
page_icon=":books:")
st.title("_Private Data :red[QA Chat]_ :books:")

if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = None
if "processComplete" not in st.session_state:
    st.session_state.processComplete = None

with st.sidebar:
    uploaded_files =  st.file_uploader("Upload your file",type=['pdf','docx'],accept_multiple_files=True)
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    process = st.button("Process")

if process:
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    files_text = get_text(uploaded_files)
    text_chunks = get_text_chunks(files_text)
    vetorestore = get_vectorstore(text_chunks)
    st.session_state.conversation = get_conversation_chain(vetorestore,MY_OPENAI_API_KEY) 
    st.session_state.processComplete = True

if 'messages' not in st.session_state:
    st.session_state['messages'] = [{"role": "assistant",
                                    "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
history = StreamlitChatMessageHistory(key="chat_messages") #llm into memory

# Chat logic
if query := st.chat_input("질문을 입력해주세요."):
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        chain = st.session_state.conversation

        with st.spinner("Thinking..."):
            result = chain({"question": query})
            with get_openai_callback() as cb:
                st.session_state.chat_history = result['chat_history']
            response = result['answer']
            source_documents = result['source_documents']

            st.markdown(response)
            with st.expander("참고 문서 확인"):
                st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
                st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)

    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})