"""
RAG Completo com ChromaDB e LangChain
=====================================
Este script implementa um sistema RAG completo:
- Recupera documentos do ChromaDB
- Cria retriever e faz busca
- Monta prompt com contexto
- Gera resposta com LLM
"""

import os
from dotenv import load_dotenv

# Carregar API key
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

print("=" * 60)
print("ETAPA 1: Conectando ao ChromaDB")
print("=" * 60)

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI
# Removendo imports não utilizados para simplificar
from langchain_core.prompts import PromptTemplate

# Configurar embedding (same as used in aula02)
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# Usar embeddings locais (HuggingFace) - MESMO MODELO da aula02
embeddings = HuggingFaceEmbeddings(
    model_name="mixedbread-ai/mxbai-embed-large-v1"
)

# Conectar ao ChromaDB existente (MESMO CAMINHO da aula02)
vectorstore = Chroma(
    persist_directory="../../chroma_db",
    embedding_function=embeddings
)

print("ChromaDB conectado!")
print(f"   Total de documentos: {vectorstore._collection.count()}")

print("\n" + "=" * 60)
print("ETAPA 2: Criando o Retriever")
print("=" * 60)

# Criar retriever com busca por similaridade
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print("Retriever criado com k=3")
print("   Método: similaridade (similarity search)")

# ── Etapa 3: Fazer uma pergunta e recuperar trechos ────────────
print("\n" + "=" * 60)
print("ETAPA 3: Recuperando trechos relevantes")
print("=" * 60)

pergunta = "O que e Inteligencia Artificial?"

print(f"Pergunta: {pergunta}")
print("\nTrechos recuperados:")

documentos_recuperados = retriever.invoke(pergunta)

for i, doc in enumerate(documentos_recuperados, 1):
    print(f"\n--- Documento {i} ---")
    print(f"Conteudo (primeiros 300 caracteres):")
    print(doc.page_content[:300])
    print(f"\nMetadados:")
    for chave, valor in doc.metadata.items():
        print(f"   {chave}: {valor}")

print("\n" + "=" * 60)
print("ETAPA 4: Montando o prompt com contexto")
print("=" * 60)

# Criar template de prompt
template = """Você é um assistente de IA. Responda apenas com base nas informações fornecidas.

Informações de contexto:
{contexto}

Pergunta do usuário: {pergunta}

Instrução: Responda apenas com base nas informações fornecidas e cite as fontes no final."""

prompt_template = PromptTemplate(
    template=template,
    input_variables=["contexto", "pergunta"]
)

# Montar contexto com os documentos recuperados
contexto = "\n\n".join([
    f"--- Trecho {i+1} ---\n{doc.page_content}"
    for i, doc in enumerate(documentos_recuperados)
])

prompt = prompt_template.format(contexto=contexto, pergunta=pergunta)

print("Prompt montado:")
print("-" * 40)
print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
print("-" * 40)

# Etapa 5: Enviar para o LLM
print("\n" + "=" * 60)
print("ETAPA 5: Gerando respostas com LLM")
print("=" * 60)

# Configurar LLM (usando OpenAI via OpenRouter)
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="openai/gpt-3.5-turbo",
    openai_api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

# Resposta COM contexto
print("\nRESPOSTA COM CONTEXTO (RAG):")
print("-" * 40)
resposta_com_contexto = llm.invoke(prompt)
print(resposta_com_contexto.content)
print("-" * 40)

# Resposta SEM contexto (pergunta direta)
print("\nRESPOSTA SEM CONTEXTO:")
print("-" * 40)
resposta_sem_contexto = llm.invoke(pergunta)
print(resposta_sem_contexto.content)
print("-" * 40)

print("\n" + "=" * 60)
print("COMPARACAO FINAL")
print("=" * 60)

print("""
Diferencas observadas:

RESPOSTA COM CONTEXTO (RAG):
   - Baseada nos documentos recuperados do ChromaDB
   - Cita as fontes dos trechos utilizados
   - Mais precisa e verificavel

RESPOSTA SEM CONTEXTO:
   - Baseada no conhecimento geral do modelo
   - Pode conter informacoes desatualizadas
   - Nao cita fontes especificas
""")

print("RAG completo executado com sucesso!")