# Formacao IA

Repositorio de estudos praticos da Formacao em IA, com exemplos em Python usando OpenRouter, LangChain, LangGraph e Streamlit.

## Estrutura do projeto

- `Monitoria/`: exercicios de fundamentos de Python e scripts simples.
- `trilha01/`: aulas introdutorias de integracao com LLM e apps Streamlit.
- `trilha02/`: aulas focadas em RAG, ingestao, interface e persistencia vetorial (ChromaDB).

## Requisitos

- Python 3.10 ou superior
- Chave de API do OpenRouter

## Setup rapido

1. Crie e ative um ambiente virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Instale as dependencias conforme a trilha/modulo:

```bash
pip install -r trilha01/modulo03/requirements.txt
pip install -r trilha02/requirements.txt
```

## Variaveis de ambiente

Crie um arquivo `.env` (na pasta do modulo que for executar) com:

```env
OPENROUTER_API_KEY=sua_chave_aqui
OPENROUTER_MODEL=gpt-4o-mini
```

## Como executar exemplos

- Chat simples:

```bash
streamlit run trilha01/modulo03/aula01/app.py
```

- Perguntas sobre PDF:

```bash
streamlit run trilha01/modulo03/aula03/app.py
```

- Exemplo de comparacao com contexto:

```bash
streamlit run trilha02/modulo01/app.py
```

## Observacoes

- Bases locais de vetores (ex.: `chroma_db/`) e arquivos `.env` nao devem ser versionados.
- Use o `.gitignore` na raiz para manter o repositorio limpo.
