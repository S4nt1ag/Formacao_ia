from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI


BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"
load_dotenv(dotenv_path=BASE_DIR.parent.parent / ".env", override=True)


def get_api_key() -> str | None:
    return (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("\ufeffOPENROUTER_API_KEY")
    )


def carregar_vectorstore() -> Chroma:
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        embeddings = FakeEmbeddings(size=384)
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name="aula04_rag",
    )


def montar_llm(api_key: str) -> ChatOpenAI:
    usar_openrouter = bool(os.getenv("OPENROUTER_API_KEY") or os.getenv("\ufeffOPENROUTER_API_KEY"))
    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "openai/gpt-3.5-turbo"),
        openai_api_key=api_key,
        base_url="https://openrouter.ai/api/v1" if usar_openrouter else None,
        temperature=0.2,
    )


def responder(pergunta: str, top_k: int = 3, score_threshold: float = 0.5) -> tuple[str, list[str]]:
    vectorstore = carregar_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": top_k, "score_threshold": score_threshold},
    )
    docs = retriever.invoke(pergunta)
    # Em alguns cenarios offline (FakeEmbeddings), scores podem sair fora de 0..1.
    # Faz fallback para similaridade simples para manter a demonstracao utilizavel.
    if not docs:
        docs = vectorstore.as_retriever(search_kwargs={"k": top_k}).invoke(pergunta)
    fontes = sorted({d.metadata.get("fonte", "desconhecida") for d in docs})

    if not docs:
        return (
            "Nao encontrei contexto suficiente na base para responder com confianca. "
            "Tente reformular a pergunta ou reduzir o score_threshold.",
            [],
        )

    contexto = "\n\n".join([f"[Fonte: {d.metadata.get('fonte', 'desconhecida')}] {d.page_content}" for d in docs])
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Voce e um assistente RAG. Responda somente com base no contexto recuperado. "
                "Se faltar informacao, diga explicitamente que nao sabe com base na base.",
            ),
            ("human", "Pergunta: {pergunta}\n\nContexto:\n{contexto}"),
        ]
    )

    api_key = get_api_key()
    if not api_key:
        return (
            "Sem chave de API no .env. Baseado apenas no contexto recuperado: "
            + " ".join([d.page_content for d in docs[:2]]),
            fontes,
        )

    llm = montar_llm(api_key)
    chain = prompt | llm
    try:
        resposta = chain.invoke({"pergunta": pergunta, "contexto": contexto})
        return resposta.content, fontes
    except Exception as exc:
        return (
            "Falha ao consultar o modelo online. Resumo local do contexto recuperado: "
            + " ".join([d.page_content for d in docs[:2]])
            + f" (detalhe tecnico: {type(exc).__name__})",
            fontes,
        )


def main() -> None:
    st.title("Aula 4 - RAG completo em execucao")
    st.write("Execute `python ingest.py` antes de usar esta interface.")

    top_k = st.slider("top_k", 1, 10, 3)
    score_threshold = st.slider("score_threshold", 0.0, 1.0, 0.5)
    pergunta = st.text_input("Pergunte algo para o sistema:")

    if st.button("Consultar RAG") and pergunta.strip():
        if not CHROMA_DIR.exists():
            st.error("Base vetorial nao encontrada. Rode `python ingest.py` em `modulo03/aula04`.")
            return

        with st.spinner("Consultando base e gerando resposta..."):
            resposta, fontes = responder(
                pergunta=pergunta,
                top_k=top_k,
                score_threshold=score_threshold,
            )

        st.write("### Resposta")
        st.write(resposta)
        if fontes:
            st.write("### Fontes")
            for fonte in fontes:
                st.write(f"- {fonte}")
        else:
            st.info("Nenhuma fonte retornada para esta pergunta.")


if __name__ == "__main__":
    main()
