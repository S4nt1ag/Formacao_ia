# Formação IA

Repositório de estudos e práticas da **Formação em IA**, com exemplos em Python usando **OpenRouter**, **LangChain**, **LangGraph** e **Streamlit**.

## O que tem aqui

- **Monitoria / Monitoria 01**: projeto básico em Python com operações matemáticas, módulos, funções e geração de gráficos com Matplotlib.
- **Trilha 01 / Módulo 02**: primeiros exemplos de chamadas a LLM e composição de prompts/chains.
- **Trilha 01 / Módulo 03**: apps com **Streamlit**:
  - chat simples com LLM
  - chatbot com **memória/histórico**
  - perguntas e respostas sobre **PDF**
  - versão com **streaming** (resposta "ao vivo")
- **Trilha 02 / Módulo 01**: app Streamlit comparando respostas de LLM com e sem contexto sobre energia solar no Brasil.

## Requisitos

- **Python 3.10+** (recomendado)
- Conta/chave no **OpenRouter**

## Instalação

Os arquivos de dependências estão localizados em cada módulo/trilha. Por exemplo:

- `trilha01/modulo03/requirements.txt` para apps Streamlit
- `trilha02/requirements.txt` para o app de comparação
- `Monitoria/monitoria01/bibliotecas.txt` para o projeto básico (lista de bibliotecas)

```bash
python -m venv .venv
```

Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
pip install -r [caminho/do/requirements.txt]
```

## Configuração (variáveis de ambiente)

Os apps usam OpenRouter via a variável abaixo:

- **`OPENROUTER_API_KEY`**: sua API key do OpenRouter (**obrigatória**)

Opcional:

- **`OPENROUTER_MODEL`**: modelo a usar (padrão: `gpt-4o-mini`)

Exemplo (PowerShell):

```powershell
$env:OPENROUTER_API_KEY="SUA_CHAVE_AQUI"
$env:OPENROUTER_MODEL="gpt-4o-mini"
```

> Dica: este repo já possui um `.gitignore` no módulo com regras para não versionar `.env` e secrets do Streamlit.

## Como rodar

### 1) Chat simples (Streamlit + LangChain)

```powershell
streamlit run trilha01\modulo03\aula01\app.py
```

### 2) Exemplo alternativo de chat (input + botão)

```powershell
streamlit run trilha01\modulo03\aula01\exemple1.py
```

### 3) BikeBot (memória + histórico) — LangGraph + Streamlit

```powershell
streamlit run trilha01\modulo03\aula2\app.py
```

### 4) Pergunte ao seu PDF (Q&A usando contexto do documento)

Faz upload do PDF e responde **somente** com base no texto extraído. Se não encontrar no contexto, retorna:
`Não sei com base no documento.`

```powershell
streamlit run trilha01\modulo03\aula03\app.py
```

### 5) PDF com streaming (resposta em tempo real)

```powershell
streamlit run trilha01\modulo03\aula04\app.py
```

### 6) Projeto Monitoria (operações básicas)

```powershell
python Monitoria\monitoria01\projeto-python\main.py
```

### 7) Gráfico Monitoria

```powershell
python Monitoria\monitoria01\projeto-python\grafico.py
```

### 8) Comparação com/sem contexto (Trilha 02)

```powershell
streamlit run trilha02\modulo01\app.py
```

## Estrutura (principal)

```text
Formacao_ia/
├─ README.md
├─ Monitoria/
│  └─ monitoria01/
│     ├─ bibliotecas.txt
│     ├─ ambiente-virtual/
│     └─ projeto-python/
│        ├─ grafico.py
│        ├─ main.py
│        ├─ modulos.py
│        └─ funcoes/
│           └─ multiplicacao.py
├─ trilha01/
│  ├─ modulo02/
│  │  ├─ aula01/
│  │  │  └─ api_ia.py
│  │  ├─ aula02/
│  │  │  ├─ main.py
│  │  │  └─ openroute_config.py
│  │  └─ aula03/
│  │     └─ codigo.py
│  └─ modulo03/
│     ├─ requirements.txt
│     ├─ aula01/
│     │  ├─ app.py
│     │  └─ exemple1.py
│     ├─ aula03/
│     │  └─ app.py
│     ├─ aula04/
│     │  └─ app.py
│     └─ aula2/
│        └─ app.py
└─ trilha02/
   ├─ requirements.txt
   └─ modulo01/
      └─ app.py
```

## Notas importantes

- **Chaves e segredos**: não suba sua API key para o GitHub. Use variáveis de ambiente.
- **Custos**: chamadas ao modelo podem gerar custo conforme o provedor/modelo escolhido.

## Próximos passos (ideias)

- Adicionar `.env.example` com as variáveis suportadas
- Melhorar o Q&A de PDF com chunking + embeddings (RAG) para PDFs maiores
- Adicionar testes e padronizar um `Makefile`/scripts de execução