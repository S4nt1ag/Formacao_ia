import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("Chatbot com Memória – Aula 2")
user_input = st.text_input("Digite sua pergunta:")

col1, col2 = st.columns([1, 1])
with col1:
    send = st.button("Enviar")
with col2:
    limpar = st.button("Limpar histórico")

if limpar:
    st.session_state.chat_history = []
    st.success("Histórico limpo com sucesso!")

api_key = (
    os.getenv("OPENAI_API_KEY")
    or os.getenv("OPENROUTER_API_KEY")
    or os.getenv("\ufeffOPENROUTER_API_KEY")
)

if api_key:
    llm = ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "gpt-4o-mini"),
        openai_api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
        if os.getenv("OPENROUTER_API_KEY") or os.getenv("\ufeffOPENROUTER_API_KEY")
        else None,
    )
else:
    llm = None

if send and user_input:
    if llm is None:
        st.error("Defina OPENAI_API_KEY ou OPENROUTER_API_KEY no arquivo .env.")
    else:
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        response = llm.invoke(st.session_state.chat_history)
        st.session_state.chat_history.append(AIMessage(content=response.content))
        st.write(response.content)

st.write(f"Interações armazenadas: {len(st.session_state.chat_history)}")
