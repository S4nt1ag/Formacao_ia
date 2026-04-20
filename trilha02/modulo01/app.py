import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

# ── Configuração ─────────────────────────────────────────────
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    st.error("Defina a variável de ambiente OPENROUTER_API_KEY antes de rodar.")
    st.stop()

model_name = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

llm = ChatOpenAI(
    model=model_name,
    openai_api_key=api_key,
    openai_api_base="https://openrouter.ai/api/v1",
    streaming=True,
    temperature=0,
)

# ── Conteúdo da atividade ─────────────────────────────────────
PERGUNTA = """Explique o impacto da energia solar na matriz elétrica brasileira."""

CONTEXTO = """Em 2023, o Brasil ultrapassou 35 GW de capacidade instalada de energia solar fotovoltaica, tornando-se o 5º maior mercado solar do mundo. A ANEEL registrou mais de 2 milhões de sistemas de micro e minigeração distribuída. A fonte solar já responde por mais de 16% da matriz elétrica nacional, atrás apenas da hidrelétrica. O crescimento foi impulsionado pela queda de preços dos painéis (redução de 90% em uma década) e pelo marco legal da geração distribuída (Lei 14.300/2022)."""

SYSTEM_PROMPT = "Você é um assistente didático. Responda de forma clara e objetiva em português."

# ── Interface Streamlit ───────────────────────────────────────
st.set_page_config(page_title="LLM com e sem contexto", layout="wide")
st.title("Comparação: LLM com e sem contexto")
st.caption(f"Modelo: `{model_name}` via OpenRouter")

st.info(f"**Pergunta:** {PERGUNTA}")

col1, col2 = st.columns(2)

# ── Etapa 1: Sem contexto ─────────────────────────────────────
with col1:
    st.subheader("Etapa 1 — Sem contexto")
    st.caption("Apenas a pergunta é enviada ao modelo.")
    if st.button("Gerar resposta sem contexto", use_container_width=True):
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=PERGUNTA),
        ]
        with st.spinner("Gerando..."):
            with st.chat_message("assistant"):
                st.write_stream(
                    chunk.content
                    for chunk in llm.stream(messages)
                    if chunk.content
                )

# ── Etapa 2: Com contexto (RAG) ───────────────────────────────
with col2:
    st.subheader("Etapa 2 — Com contexto (RAG)")
    st.caption("Pergunta + documento são enviados juntos.")

    with st.expander("Ver contexto enviado"):
        st.text(CONTEXTO)

    if st.button("Gerar resposta com contexto", use_container_width=True):
        prompt_rag = f"""Use o contexto abaixo para responder à pergunta com precisão.

CONTEXTO:
{CONTEXTO}

PERGUNTA:
{PERGUNTA}"""
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt_rag),
        ]
        with st.spinner("Gerando..."):
            with st.chat_message("assistant"):
                st.write_stream(
                    chunk.content
                    for chunk in llm.stream(messages)
                    if chunk.content
                )

# ── Reflexão ──────────────────────────────────────────────────
st.divider()
st.subheader("Reflexão — O que mudou?")
st.caption("O modelo analisa as vantagens do contexto para sistemas RAG.")

if st.button("Gerar análise comparativa", use_container_width=True):
    prompt_reflexao = f"""Responda em 3 tópicos curtos:
1. O que tende a mudar na resposta quando fornecemos contexto a um LLM?
2. A resposta fica mais precisa, detalhada ou atualizada? Por quê?
3. Que vantagem essa abordagem traz em um sistema RAG real?

Tema da atividade: "{PERGUNTA}"
"""
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt_reflexao),
    ]
    with st.spinner("Analisando..."):
        with st.chat_message("assistant"):
            st.write_stream(
                chunk.content
                for chunk in llm.stream(messages)
                if chunk.content
            )