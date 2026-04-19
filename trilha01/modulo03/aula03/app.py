import os
import tempfile
from typing import TypedDict

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Você é um assistente que responde perguntas usando SOMENTE o contexto fornecido.\n"
            "Se a resposta não estiver no contexto, responda exatamente: \"Não sei com base no documento.\"",
        ),
        (
            "human",
            "Contexto (extraído do PDF):\n"
            "{context}\n\n"
            "Pergunta do usuário:\n"
            "{question}",
        ),
    ]
)


class QAState(TypedDict):
    context: str
    question: str
    answer: str


def build_graph(model: ChatOpenAI):
    chain = PROMPT | model

    def generate_answer(state: QAState) -> dict:
        result = chain.invoke({"context": state["context"], "question": state["question"]})
        return {"answer": result.content}

    graph = StateGraph(QAState)
    graph.add_node("generate", generate_answer)
    graph.set_entry_point("generate")
    graph.add_edge("generate", END)
    return graph.compile()


def ensure_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pdf_text" not in st.session_state:
        st.session_state.pdf_text = ""
    if "pdf_name" not in st.session_state:
        st.session_state.pdf_name = ""


def extract_pdf_text(uploaded_file) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        docs = PyPDFLoader(tmp_path).load()
        return "\n\n".join(d.page_content for d in docs if getattr(d, "page_content", None))
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


st.set_page_config(page_title="Chat com PDF (LLM + Streamlit)", page_icon="📄")
st.title("Pergunte ao seu PDF")
st.caption("Faça upload de um PDF e pergunte com base no conteúdo do documento.")

ensure_session_state()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    st.error("Defina a variável de ambiente `OPENROUTER_API_KEY` antes de rodar o app.")
    st.stop()

model_name = os.getenv("OPENROUTER_MODEL", "gpt-4o-mini")
llm = ChatOpenAI(
    model=model_name,
    openai_api_key=api_key,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0,
)
app = build_graph(llm)

with st.sidebar:
    st.subheader("PDF")
    uploaded = st.file_uploader("Envie um arquivo PDF", type=["pdf"])

    if uploaded is not None:
        extracted = extract_pdf_text(uploaded)
        st.session_state.pdf_text = extracted
        st.session_state.pdf_name = uploaded.name
        st.success(f"PDF carregado: {uploaded.name}")
        st.caption(f"Caracteres extraídos: {len(extracted):,}".replace(",", "."))
    elif st.session_state.pdf_name:
        st.info(f"PDF atual: {st.session_state.pdf_name}")

    if st.button("Limpar conversa"):
        st.session_state.chat_history = []

for role, content in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(content)

question = st.chat_input("Digite sua pergunta sobre o PDF...")
if question:
    if not st.session_state.pdf_text.strip():
        st.warning("Primeiro faça upload de um PDF para usar como contexto.")
        st.stop()

    st.session_state.chat_history.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        result = app.invoke({"context": st.session_state.pdf_text, "question": question, "answer": ""})
        answer = result["answer"]
        st.session_state.chat_history.append(("assistant", answer))
        st.markdown(answer)
