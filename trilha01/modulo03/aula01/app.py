import os

import streamlit as st
from langchain_openai import ChatOpenAI


st.title("Chat com IA (LangChain + Streamlit)")

pergunta = st.text_area("Digite sua pergunta:")

if st.button("Enviar"):
    if not pergunta.strip():
        st.warning("Por favor, digite uma pergunta.")
        st.stop()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        st.error("Defina OPENROUTER_API_KEY antes de rodar o app.")
        st.stop()

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
    )

    resposta = llm.invoke(pergunta)
    st.write(resposta.content)
