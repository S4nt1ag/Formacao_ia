"""
RAG com Wikipedia, ChromaDB e LangGraph
=======================================
Este script implementa um sistema RAG completo usando:
- WikipediaLoader para carregar artigos
- Sentence Transformers para embeddings
- ChromaDB como banco vetorial
- LangGraph para orquestração
"""

import os
from dotenv import load_dotenv

# Carregar API key
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# ── Etapa 1: Carregar artigos da Wikipedia ─────────────────────
print("=" * 60)
print("ETAPA 1: Carregando artigos da Wikipedia")
print("=" * 60)

from langchain_community.document_loaders import WikipediaLoader

# Temas para carregar
temas = ["Inteligência Artificial", "Internet"]

documentos = []
for tema in temas:
    print(f"\n📥 Carregando: {tema}")
    loader = WikipediaLoader(query=tema, load_max_docs=2)
    docs = loader.load()
    print(f"   → {len(docs)} documento(s) carregado(s)")
    for doc in docs:
        doc.metadata["tema"] = tema
        documentos.append(doc)

print(f"\n✅ Total de documentos: {len(documentos)}")

# ── Etapa 2: Dividir em chunks ─────────────────────────────────
print("\n" + "=" * 60)
print("ETAPA 2: Dividindo em chunks")
print("=" * 60)

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = splitter.split_documents(documentos)
print(f"✅ Total de chunks: {len(chunks)}")

for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i+1} ---")
    print(f"Fonte: {chunk.metadata.get('tema', 'N/A')}")
    print(f"Tamanho: {len(chunk.page_content)} caracteres")
    print(f"Preview: {chunk.page_content[:150]}...")

# ── Etapa 3: Gerar embeddings ──────────────────────────────────
print("\n" + "=" * 60)
print("ETAPA 3: Gerando embeddings")
print("=" * 60)

from langchain_huggingface import HuggingFaceEmbeddings

print("📦 Carregando modelo de embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="mixedbread-ai/mxbai-embed-large-v1"
)
print("✅ Modelo carregado!")

# Teste de embedding
test_embedding = embeddings.embed_query("Teste de embedding")
print(f"✅ Embedding gerado! Dimensão: {len(test_embedding)}")

# ── Etapa 4: Criar banco vetorial com ChromaDB ────────────────
print("\n" + "=" * 60)
print("ETAPA 4: Criando banco vetorial com ChromaDB")
print("=" * 60)

from langchain_community.vectorstores import Chroma

# Caminho para persistência
chroma_path = "./chroma_db"

# Verificar se já existe banco anterior
if os.path.exists(chroma_path):
    print("🗑️ Removendo banco anterior...")
    import shutil
    shutil.rmtree(chroma_path)

print("📚 Criando banco vetorial...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=chroma_path
)
print(f"✅ Banco vetorial criado com {vectorstore._collection.count()} documentos")

# ── Etapa 5: Testar consulta de similaridade ───────────────────
print("\n" + "=" * 60)
print("ETAPA 5: Testando consulta de similaridade")
print("=" * 60)

consulta_teste = "Quando surgiu o termo Inteligência Artificial?"

print(f"\n🔍 Consulta: '{consulta_teste}'")
resultados = vectorstore.similarity_search_with_score(consulta_teste, k=3)

print(f"\n📊 Resultados encontrados: {len(resultados)}")
for i, (doc, score) in enumerate(resultados):
    print(f"\n--- Resultado {i+1} (similaridade: {1-score:.4f}) ---")
    print(f"Fonte: {doc.metadata.get('tema', 'N/A')}")
    print(f"Conteúdo: {doc.page_content[:200]}...")

# ── Etapa 6: Montar grafo LangGraph ───────────────────────────
print("\n" + "=" * 60)
print("ETAPA 6: Montando grafo LangGraph")
print("=" * 60)

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

# Definir estado do grafo
class GraphState(TypedDict):
    pergunta: str
    contexto: List[str]
    resposta: str
    citacoes: List[str]

# Configurar LLM
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    openai_api_key=api_key,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0,
)

# Nó 1: Buscar no banco vetorial
def buscar_contexto(state: GraphState) -> GraphState:
    pergunta = state["pergunta"]
    print(f"\n🔍 [Nó Buscar] Pergunta: {pergunta}")
    
    resultados = vectorstore.similarity_search(pergunta, k=3)
    contexto = [doc.page_content for doc in resultados]
    citacoes = [doc.metadata.get("tema", "Unknown") for doc in resultados]
    
    print(f"   → {len(contexto)} chunks encontrados")
    
    return {
        **state,
        "contexto": contexto,
        "citacoes": citacoes
    }

# Nó 2: Gerar resposta
def gerar_resposta(state: GraphState) -> GraphState:
    pergunta = state["pergunta"]
    contexto = "\n\n".join(state["contexto"])
    citacoes = state["citacoes"]
    
    print(f"📝 [Nó Gerar] Criando resposta...")
    
    prompt = f"""Você é um assistente de pesquisa. Use o contexto abaixo para responder à pergunta.

INSTRUÇÕES:
- Responda em português
- Cite as fontes utilizadas no final da resposta
- Sea preciso e objetivo

CONTEXTO:
{contexto}

PERGUNTA: {pergunta}

RESPOSTA:"""

    resposta = llm.invoke(prompt)
    resposta_texto = resposta.content if hasattr(resposta, 'content') else str(resposta)
    
    # Adicionar citações
    citacoes_texto = "\n\n**Fontes:**\n" + "\n".join([f"- {c}" for c in set(citacoes)])
    resposta_completa = resposta_texto + citacoes_texto
    
    print(f"   → Resposta gerada ({len(resposta_texto)} caracteres)")
    
    return {
        **state,
        "resposta": resposta_completa
    }

# Criar grafo
print("🏗️ Construindo grafo...")
grafo = StateGraph(GraphState)

# Adicionar nós
grafo.add_node("buscar", buscar_contexto)
grafo.add_node("gerar", gerar_resposta)

# Definir fluxo
grafo.set_entry_point("buscar")
grafo.add_edge("buscar", "gerar")
grafo.add_edge("gerar", END)

# Compilar
agente_rag = grafo.compile()
print("✅ Grafo compilado!")

# ── Etapa 7: Executar pergunta de teste ────────────────────────
print("\n" + "=" * 60)
print("ETAPA 7: Executando pergunta de teste")
print("=" * 60)

pergunta_final = "Quando surgiu o termo Inteligência Artificial?"

print(f"\n❓ Pergunta: {pergunta_final}")
print("\n" + "-" * 60)

# Executar grafo
resultado = agente_rag.invoke({"pergunta": pergunta_final})

print("\n📬 RESPOSTA:")
print("=" * 60)
print(resultado["resposta"])
print("=" * 60)

print("\n✅ Pipeline RAG executado com sucesso!")