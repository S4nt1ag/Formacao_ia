import streamlit as st
import os
from langchain_openai import ChatOpenAI

st.title("Chat integrado com IA usando Langchain")
prompt = st.text_input("Digite sua pergunta:")

if st.button("Enviar"):
    if prompt:
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("Defina OPENROUTER_API_KEY (ou OPENAI_API_KEY) antes de rodar o app.")
            st.stop()

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
        )
        try:
            resposta = llm.invoke(prompt)
            st.write(resposta.content)
        except Exception as e:
            st.error(str(e))
else:
    st.warning("Por favor, digite uma pergunta")