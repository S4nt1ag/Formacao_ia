import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    HuggingFaceEmbeddings = None

load_dotenv()


def criar_documentos_exemplo() -> list[Document]:
    return [
        Document(
            page_content=(
                "A história da Inteligência Artificial começou oficialmente em 1956 na "
                "Conferência de Dartmouth. John McCarthy popularizou o termo IA."
            ),
            metadata={"fonte": "Wikipedia", "topico": "historia_ia"},
        ),
        Document(
            page_content=(
                "Deep Blue venceu Garry Kasparov em 1997 e foi um marco da IA simbólica "
                "e de sistemas especializados para jogos."
            ),
            metadata={"fonte": "Wikipedia", "topico": "deep_blue"},
        ),
        Document(
            page_content=(
                "A partir de 2012, redes neurais profundas e transformadores como BERT e GPT "
                "aceleraram aplicações de IA generativa."
            ),
            metadata={"fonte": "DocumentoProprio", "topico": "deep_learning"},
        ),
    ]


def criar_embeddings():
    if HuggingFaceEmbeddings is not None:
        try:
            return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        except Exception:
            pass
    st.warning("Usando embeddings de demonstração (offline).")
    return FakeEmbeddings(size=384)


def carregar_vectorstore() -> Chroma:
    embeddings = criar_embeddings()
    candidatos = [
        Path("modulo02/aula05/chroma_db"),
        Path("chroma_db"),
    ]
    for caminho in candidatos:
        if caminho.exists():
            return Chroma(
                persist_directory=str(caminho),
                embedding_function=embeddings,
            )

    caminho_novo = Path("modulo03/aula03/chroma_db")
    return Chroma.from_documents(
        documents=criar_documentos_exemplo(),
        embedding=embeddings,
        persist_directory=str(caminho_novo),
        collection_name="aula03_rag",
    )


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "testes" not in st.session_state:
    st.session_state.testes = []

st.title("Chatbot com Memória + RAG – Aula 3")
st.subheader("Teste de top_k, score_threshold e filtros")

top_k = st.slider("Número de documentos a recuperar (top_k):", 1, 10, 3)
score_threshold = st.slider(
    "Limite mínimo de similaridade (score_threshold):", 0.0, 1.0, 0.5
)
fonte = st.text_input("Filtrar por fonte (opcional):")
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
usar_openrouter = bool(os.getenv("OPENROUTER_API_KEY") or os.getenv("\ufeffOPENROUTER_API_KEY"))

if api_key:
    llm = ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "openai/gpt-3.5-turbo"),
        openai_api_key=api_key,
        base_url="https://openrouter.ai/api/v1" if usar_openrouter else None,
        temperature=0.3,
    )
else:
    llm = None

if send and user_input:
    if llm is None:
        st.error("Defina OPENAI_API_KEY ou OPENROUTER_API_KEY no arquivo .env.")
    else:
        vectorstore = carregar_vectorstore()
        search_kwargs = {"k": top_k, "score_threshold": score_threshold}
        if fonte:
            search_kwargs["filter"] = {"fonte": fonte}

        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs=search_kwargs,
        )
        documentos = retriever.invoke(user_input)

        contexto = "\n\n".join([doc.page_content for doc in documentos]) or "Sem contexto recuperado."
        fontes = sorted({doc.metadata.get("fonte", "desconhecida") for doc in documentos})
        fontes_texto = ", ".join(fontes) if fontes else "Nenhuma"

        st.session_state.chat_history.append(HumanMessage(content=user_input))
        prompt = (
            f"Use o contexto para responder.\n\nContexto:\n{contexto}\n\n"
            f"Pergunta: {user_input}\n\n"
            "Se o contexto for insuficiente, diga isso claramente."
        )
        response = llm.invoke(st.session_state.chat_history + [HumanMessage(content=prompt)])
        st.session_state.chat_history.append(AIMessage(content=response.content))

        st.write("### Resposta")
        st.write(response.content)
        st.caption(f"Fontes retornadas: {fontes_texto}")

        st.session_state.testes.append(
            {
                "Pergunta": user_input,
                "top_k": top_k,
                "score_threshold": score_threshold,
                "Fontes retornadas": fontes_texto,
                "Clareza/relevância": "Alta" if len(documentos) >= 2 else "Média/Baixa",
            }
        )
        st.session_state.testes = st.session_state.testes[:3]

st.write(f"Interações armazenadas: {len(st.session_state.chat_history)}")
st.write("### Tabela de observações (3 testes)")
if st.session_state.testes:
    st.table(st.session_state.testes)
else:
    st.info("Envie perguntas para preencher a tabela de observações.")
