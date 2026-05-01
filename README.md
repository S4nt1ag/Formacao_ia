# Formacao IA

Repositorio completo da Formacao em IA, organizado em 3 trilhas com projetos praticos de LLM, RAG, agentes, MCP e automacoes.

## Trilhas e projetos

### Trilha 01 - Fundamentos de LLM e agentes

- `trilha01/modulo02/aula01/api_ia.py`: consumo basico de API de IA.
- `trilha01/modulo02/aula02/main.py`: prompt + parser com LangChain.
- `trilha01/modulo02/aula02/openroute_config.py`: configuracao de cliente OpenRouter/OpenAI.
- `trilha01/modulo02/aula03/codigo.py`: fluxo com `LangGraph` e memoria.
- `trilha01/modulo03/aula01/app.py`: chat Streamlit inicial.
- `trilha01/modulo03/aula01/exemple1.py`: exemplo adicional de chat.
- `trilha01/modulo03/aula2/app.py`: chat com historico de mensagens.
- `trilha01/modulo03/aula03/app.py`: leitura de PDF com `PyPDFLoader`.
- `trilha01/modulo03/aula04/app.py`: leitura de PDF com `PyMuPDFLoader`.

### Trilha 02 - RAG, embeddings e persistencia vetorial

- `trilha02/modulo01/app.py`: comparacao de respostas com contexto.
- `trilha02/modulo02/aula02/app.py`: pipeline RAG com Wikipedia, split e embeddings.
- `trilha02/modulo02/aula03/app.py`: testes de loaders PDF e HTML.
- `trilha02/modulo02/aula04/app.py`: RAG completo com ChromaDB.
- `trilha02/modulo02/aula05/app.py`: utilitarios de arquivos e preparacao de base.
- `trilha02/modulo03/aula01/interface_rag.py`: interface RAG com Streamlit.
- `trilha02/modulo03/aula02/app.py`: chat com historico e contexto.
- `trilha02/modulo03/aula03/app.py`: testes com Chroma + embeddings fake.
- `trilha02/modulo03/aula04/ingest.py`: ingestao de documentos no banco vetorial.
- `trilha02/modulo03/aula04/app.py`: consulta RAG sobre base ingerida.
- `trilha02/PUBLIC/ciclismo.html`: arquivo de apoio para testes de ingestao.

### Trilha 03 - Agentes avancados, MCP e automacoes

- `trilha03/modulo01/aula03/app.py`: agente ReAct com funcoes utilitarias.
- `trilha03/modulo01/aula03/M1A3.ipynb`: notebook da aula 03.
- `trilha03/modulo01/aula04/app.py`: agente com tools e busca Tavily.
- `trilha03/modulo01/aula04/M1A4.ipynb`: notebook da aula 04.
- `trilha03/modulo01/aula07/app.py`: planner de viagem com grafo multi-etapas.
- `trilha03/modulo01/aula07/M1A7.py`: versao completa com feedback humano.
- `trilha03/modulo03/aula03/app.py`: servidor MCP para validacao e consulta JSON.
- `trilha03/modulo03/aula05/app.py`: servidor MCP `poi.find` com OpenTripMap.
- `trilha03/modulo03/aula05/n8n_weather_packing_webhook.json`: fluxo n8n de clima + checklist.
- `trilha03/modulo03/aula05/webhook_payload_exemplo.json`: payload de exemplo para webhook.
- `trilha03/modulo03/aula05/README.md`: guia detalhado da aula 05.
- `trilha03/mcp.json` e `trilha03/mcp.python3.json`: configuracoes MCP da trilha.

## Requisitos

- Python 3.10+
- Chaves de API conforme o projeto (`OPENROUTER_API_KEY`, `TAVILY_API_KEY`, `OPENTRIPMAP_API_KEY`)

## Setup rapido

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r trilha01/requirements.txt
pip install -r trilha02/requirements.txt
pip install -r trilha03/requirements.txt
```

## Variaveis de ambiente

Exemplo de `.env` na raiz do projeto:

```env
OPENROUTER_API_KEY=sua_chave_aqui
OPENROUTER_MODEL=openai/gpt-4o-mini
TAVILY_API_KEY=sua_chave_tavily
OPENTRIPMAP_API_KEY=sua_chave_opentripmap
```

## Exemplos de execucao

```bash
streamlit run trilha01/modulo03/aula01/app.py
python trilha02/modulo02/aula04/app.py
python trilha03/modulo01/aula07/app.py
python trilha03/modulo03/aula03/app.py
```

## Observacoes

- Bases locais (`chroma_db/`), arquivos de ambiente e segredos devem ficar fora do versionamento.
- Use o `.gitignore` da raiz para manter o repositorio limpo em todas as trilhas.
