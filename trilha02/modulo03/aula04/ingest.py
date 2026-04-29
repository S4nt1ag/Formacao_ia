from __future__ import annotations

import shutil
from pathlib import Path

from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"


def documentos_base() -> list[Document]:
    return [
        Document(
            page_content=(
                "Aprendizado por reforco e um paradigma no qual um agente aprende por tentativa e erro, "
                "recebendo recompensas ao interagir com um ambiente. O objetivo e maximizar a recompensa acumulada."
            ),
            metadata={"fonte": "Wikipedia", "tema": "reinforcement_learning"},
        ),
        Document(
            page_content=(
                "No artigo Attention Is All You Need, os autores concluem que a arquitetura Transformer, "
                "baseada apenas em mecanismos de atencao, substitui recorrencia com melhor paralelizacao e "
                "atinge resultados de ponta em traducao automatica."
            ),
            metadata={"fonte": "Attention Is All You Need", "tema": "transformers"},
        ),
        Document(
            page_content=(
                "Entre as principais conclusoes do artigo estao: melhor qualidade em benchmarks de traducao, "
                "reducoes no tempo de treinamento e capacidade de modelar dependencias de longo alcance."
            ),
            metadata={"fonte": "Attention Is All You Need", "tema": "conclusoes"},
        ),
        Document(
            page_content=(
                "Sistemas RAG combinam recuperacao de informacao com geracao de texto para responder "
                "perguntas com base em documentos externos e citar fontes do contexto utilizado."
            ),
            metadata={"fonte": "DocumentoProprio", "tema": "rag"},
        ),
    ]


def main() -> None:
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as exc:
        print(f"Aviso: falha ao carregar embedding remoto ({exc}). Usando FakeEmbeddings.")
        embeddings = FakeEmbeddings(size=384)
    docs = documentos_base()
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name="aula04_rag",
    )

    print("Ingestao concluida com sucesso.")
    print(f"Diretorio persistente: {CHROMA_DIR}")
    print(f"Documentos indexados: {db._collection.count()}")


if __name__ == "__main__":
    main()
