from __future__ import annotations

import os
import shutil
from pathlib import Path
from textwrap import shorten

try:
    from dotenv import load_dotenv
    from langchain_community.embeddings import FakeEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI
except ImportError as exc:
    raise SystemExit(
        "Dependência ausente. Instale com: pip install -r requirements.txt\n"
        f"Detalhe: {exc}"
    ) from exc


def criar_embeddings():
    """Tenta usar embedding real e cai para modo offline em caso de erro."""
    try:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as exc:
        print(
            "Aviso: falha ao carregar modelo Hugging Face. "
            "Usando FakeEmbeddings para demonstração local.\n"
            f"Detalhe: {exc}"
        )
        return FakeEmbeddings(size=384)


def obter_env(nome: str, default: str | None = None) -> str | None:
    """Lê variável de ambiente, inclusive quando há BOM acidental no .env."""
    return os.getenv(nome) or os.getenv(f"\ufeff{nome}") or default


def criar_documentos_exemplo() -> list[Document]:
    """Cria um pequeno conjunto de textos para alimentar a base vetorial."""
    return [
        Document(
            page_content=(
                "A história da Inteligência Artificial começou oficialmente em 1956, "
                "na Conferência de Dartmouth, quando John McCarthy popularizou o termo IA. "
                "Nas décadas seguintes, surgiram os primeiros sistemas especialistas."
            ),
            metadata={"fonte": "Wikipedia", "topico": "historia_ia"},
        ),
        Document(
            page_content=(
                "Nos anos 1990 e 2000, marcos importantes incluíram o Deep Blue vencendo "
                "Kasparov em 1997 e o avanço de métodos estatísticos e aprendizado de máquina "
                "aplicados a visão computacional e linguagem natural."
            ),
            metadata={"fonte": "Wikipedia", "topico": "marcos_modernos"},
        ),
        Document(
            page_content=(
                "A partir de 2012, o deep learning ganhou destaque com redes neurais profundas. "
                "Modelos transformadores, como BERT e GPT, mudaram o estado da arte em NLP, "
                "culminando em aplicações generativas amplamente usadas."
            ),
            metadata={"fonte": "DocumentoProprio", "topico": "deep_learning"},
        ),
    ]


def montar_vectorstore(persist_dir: Path) -> Chroma:
    # Evita duplicar documentos no Chroma ao executar o script várias vezes.
    if persist_dir.exists():
        shutil.rmtree(persist_dir)

    embeddings = criar_embeddings()
    documentos = criar_documentos_exemplo()
    return Chroma.from_documents(
        documents=documentos,
        embedding=embeddings,
        persist_directory=str(persist_dir),
        collection_name="historico_ia",
    )


def imprimir_resultados(titulo: str, docs: list[Document]) -> None:
    print(f"\n{titulo}")
    print("-" * len(titulo))
    if not docs:
        print("Nenhum resultado encontrado.")
        return
    for i, doc in enumerate(docs, start=1):
        fonte = doc.metadata.get("fonte", "desconhecida")
        topico = doc.metadata.get("topico", "n/a")
        trecho = shorten(doc.page_content, width=160, placeholder="...")
        print(f"{i}. fonte={fonte} | topico={topico} | trecho={trecho}")


def testar_k(vectorstore: Chroma, query: str) -> None:
    for k in (1, 3, 5):
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        resultados = retriever.invoke(query)
        imprimir_resultados(f"Busca por similaridade com k={k}", resultados)


def testar_threshold(vectorstore: Chroma, query: str) -> None:
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.7, "k": 5},
    )
    resultados = retriever.invoke(query)
    imprimir_resultados(
        "Busca com similarity_score_threshold=0.7 (k=5)",
        resultados,
    )


def testar_filtro_metadados(vectorstore: Chroma, query: str) -> None:
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5, "filter": {"fonte": "Wikipedia"}}
    )
    resultados = retriever.invoke(query)
    imprimir_resultados("Busca com filtro de metadados fonte=Wikipedia", resultados)


def montar_contexto(docs: list[Document]) -> str:
    if not docs:
        return "Nenhum contexto recuperado."
    return "\n\n".join(f"- {doc.page_content}" for doc in docs)


def gerar_resposta(
    query: str,
    contexto: str,
    instrucoes: str,
    temperature: float,
    model: str,
    api_key: str,
) -> str:
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", instrucoes),
            ("human", "Pergunta: {query}\n\nContexto:\n{contexto}"),
        ]
    )
    chain = prompt | llm
    resposta = chain.invoke({"query": query, "contexto": contexto})
    return resposta.content


def comparar_prompts_e_temperaturas(vectorstore: Chroma, query: str) -> None:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)
    contexto = montar_contexto(docs)

    prompt_1 = "Use apenas o contexto fornecido para responder de forma simples e didática."
    prompt_2 = (
        "Combine o contexto fornecido com seu conhecimento prévio e explique de forma aprofundada."
    )
    model = obter_env("OPENROUTER_MODEL", "openai/gpt-3.5-turbo") or "openai/gpt-3.5-turbo"
    api_key = obter_env("OPENROUTER_API_KEY") or obter_env("OPENAI_API_KEY", "") or ""

    for instrucoes in (prompt_1, prompt_2):
        print(f"\nPrompt: {instrucoes}")
        print("-" * (8 + len(instrucoes)))
        for temperature in (0.2, 0.7, 1.0):
            try:
                resposta = gerar_resposta(
                    query=query,
                    contexto=contexto,
                    instrucoes=instrucoes,
                    temperature=temperature,
                    model=model,
                    api_key=api_key,
                )
            except Exception as exc:
                resposta = (
                    "Falha ao consultar o OpenRouter nesta execução. "
                    "Verifique conexão/proxy e tente novamente.\n"
                    f"Detalhe técnico: {type(exc).__name__}: {exc}"
                )
            print(f"\nTemperatura={temperature}")
            print(shorten(resposta.replace("\n", " "), width=500, placeholder="..."))


def main() -> None:
    raiz_projeto = Path(__file__).resolve().parents[2]
    load_dotenv(dotenv_path=raiz_projeto / ".env", override=True)

    persist_dir = Path(__file__).parent / "chroma_db"
    query = "Quais foram os principais marcos da Inteligência Artificial?"

    print("Criando base vetorial com ChromaDB...")
    vectorstore = montar_vectorstore(persist_dir)

    testar_k(vectorstore, query)
    testar_threshold(vectorstore, query)
    testar_filtro_metadados(vectorstore, query)

    if not (obter_env("OPENROUTER_API_KEY") or obter_env("OPENAI_API_KEY")):
        print(
            "\nOPENROUTER_API_KEY não encontrada. "
            "Pulando comparação de prompts e temperaturas."
        )
        print("Defina a chave no ambiente para executar essa etapa.")
        return

    comparar_prompts_e_temperaturas(vectorstore, query)


if __name__ == "__main__":
    main()
