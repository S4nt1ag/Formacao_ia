import os

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

st.set_page_config(page_title="Meu primeiro sistema RAG", page_icon="💬")
st.title("💬 Meu primeiro sistema RAG")
st.subheader("Sistema de IA com LangChain + OpenAI")

pergunta = st.text_area("Digite sua pergunta:")

if st.button("Gerar resposta"):
    if not pergunta.strip():
        st.warning("Digite uma pergunta antes de enviar.")
    else:
        api_key = (
            os.getenv("OPENROUTER_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("\ufeffOPENROUTER_API_KEY")
        )

        if not api_key:
            st.error("Defina OPENROUTER_API_KEY ou OPENAI_API_KEY no arquivo .env.")
        else:
            model = os.getenv("OPENROUTER_MODEL", "openai/gpt-3.5-turbo")

            with st.spinner("Carregando resposta..."):
                modelo = ChatOpenAI(
                    model=model,
                    openai_api_key=api_key,
                    base_url="https://openrouter.ai/api/v1",
                    temperature=0.3,
                )
                resposta = modelo.invoke(pergunta)

            st.write("### Resposta:")
            st.write(resposta.content)
