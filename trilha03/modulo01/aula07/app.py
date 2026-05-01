import math
import os
from pathlib import Path
from typing import Annotated, List, TypedDict

import operator

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, START, StateGraph
    from tavily import TavilyClient
except ImportError as exc:
    raise SystemExit(
        "Dependencias ausentes. Instale com: "
        "pip install langgraph langchain-openai tavily-python"
    ) from exc


PROJECT_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def load_env_file(path: Path) -> dict:
    env_data = {}
    if not path.exists():
        return env_data
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env_data[key.strip()] = value.strip().strip('"').strip("'")
    return env_data


dotenv_data = load_env_file(PROJECT_ENV_PATH)
for key, value in dotenv_data.items():
    os.environ[key] = value


def require_env_value(var_names: List[str], env_map: dict, error_message: str) -> str:
    for name in var_names:
        value = os.getenv(name) or env_map.get(name)
        if value and value.strip():
            return value.strip()
    raise SystemExit(error_message)


openrouter_api_key = require_env_value(
    ["OPENROUTER_API_KEY", "OPENAI_API_KEY"],
    dotenv_data,
    f"Defina OPENROUTER_API_KEY (ou OPENAI_API_KEY) em {PROJECT_ENV_PATH}.",
)
tavily_api_key = require_env_value(
    ["TAVILY_API_KEY", "TAVILY_SEARCH_API"],
    dotenv_data,
    f"Defina TAVILY_API_KEY (ou TAVILY_SEARCH_API) em {PROJECT_ENV_PATH}.",
)

model_name = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
temperature = float(os.getenv("OPENROUTER_TEMPERATURE", "0"))

model = ChatOpenAI(
    model=model_name,
    temperature=temperature,
    openai_api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
)
tavily_client = TavilyClient(api_key=tavily_api_key)


class AgentState(TypedDict):
    task: str
    queries: List[str]
    draft: Annotated[List[str], operator.add]
    tool_plan: List[dict]
    tool_results: Annotated[List[str], operator.add]
    result: str

QUERY_MAKER_PROMPT = """
Voce e um especialista em planejamento de viagens.
Recebera um pedido do usuario e deve gerar de 4 a 6 queries curtas para pesquisar:
- atividades no destino
- melhores regioes/bairros
- custos e dicas praticas
Retorne EXATAMENTE uma query por linha no formato:
QUERY: <texto da query>
""".strip()

TOOL_PLANNER_PROMPT = """
Voce e um orquestrador de ferramentas para um agente de viagens.
Com base na tarefa do usuario, gere chamadas para as tools abaixo.

Tools disponiveis:
- calcular_orcamento: args "destino|dias|diaria"
- ver_temperatura: args "destino"
- sugerir_restaurantes: args "destino"
- preco_passagens: args "destino"

Regras:
- Quando o usuario nao informar dias, use 5.
- Quando o usuario nao informar diaria, use 350.
- Gere de 2 a 4 chamadas.
- Inclua calcular_orcamento e sugerir_restaurantes sempre que fizer sentido em roteiro.
Retorne EXATAMENTE uma chamada por linha no formato:
TOOL: <acao>|<args>
Exemplos:
TOOL: ver_temperatura|Paris
TOOL: calcular_orcamento|Paris|5|350
""".strip()

RESULT_PROMPT = """
Voce e um especialista em planejamento de viagens.
Monte um plano claro e objetivo em portugues do Brasil.

Estruture a resposta:
1) Resumo do destino
2) Roteiro sugerido
3) Orcamento e custos
4) Restaurantes e gastronomia
5) Clima e melhor periodo
6) Passagens e dicas finais

Dados pesquisados:
{draft}

Resultados das ferramentas:
{tool_results}

Pedido do usuario:
{task}
""".strip()


def _first_search_content(query: str, default_text: str) -> str:
    try:
        response = tavily_client.search(query=query, max_results=3)
        results = response.get("results", [])
        if not results:
            return default_text
        top = results[0]
        title = top.get("title", "Fonte web")
        content = top.get("content", "").strip()
        if not content:
            return default_text
        return f"{title}: {content[:350]}"
    except Exception:
        return default_text


def calcular_orcamento(args: str) -> str:
    destino, dias, diaria = [p.strip() for p in args.split("|")]
    dias_int = max(1, int(dias))
    diaria_float = float(diaria)
    hospedagem = dias_int * diaria_float
    alimentacao = dias_int * 120.0
    transporte_local = dias_int * 35.0
    atividades = dias_int * 90.0
    subtotal = hospedagem + alimentacao + transporte_local + atividades
    margem = subtotal * 0.12
    total = subtotal + margem
    return (
        f"Orcamento estimado para {destino} ({dias_int} dias): "
        f"Hospedagem R$ {hospedagem:.2f}, Alimentacao R$ {alimentacao:.2f}, "
        f"Transporte local R$ {transporte_local:.2f}, Atividades R$ {atividades:.2f}, "
        f"Reserva/imprevistos R$ {margem:.2f}. Total aproximado R$ {total:.2f}."
    )


def ver_temperatura(args: str) -> str:
    destino = args.strip()
    return _first_search_content(
        f"temperatura media atual em {destino}",
        f"Nao consegui obter clima em tempo real para {destino}.",
    )


def sugerir_restaurantes(args: str) -> str:
    destino = args.strip()
    return _first_search_content(
        f"melhores restaurantes em {destino}",
        f"Nao consegui listar restaurantes para {destino}.",
    )


def preco_passagens(args: str) -> str:
    destino = args.strip()
    web_text = _first_search_content(
        f"preco medio passagem aerea para {destino} saindo do Brasil",
        "",
    )
    if web_text:
        return f"Estimativa de passagens para {destino}: {web_text}"
    base = 3200 + int(math.fabs(hash(destino)) % 1800)
    return (
        f"Estimativa basica de passagem ida e volta para {destino}: "
        f"entre R$ {base} e R$ {base + 1400}, variando por epoca e antecedencia."
    )


tools = {
    "calcular_orcamento": calcular_orcamento,
    "ver_temperatura": ver_temperatura,
    "sugerir_restaurantes": sugerir_restaurantes,
    "preco_passagens": preco_passagens,
}


def parse_queries(raw_text: str) -> List[str]:
    queries = []
    for line in raw_text.splitlines():
        clean = line.strip()
        if not clean:
            continue
        if clean.lower().startswith("query:"):
            query = clean.split(":", 1)[1].strip()
            if query:
                queries.append(query)
    if queries:
        return queries
    fallback = [line.strip("- ").strip() for line in raw_text.splitlines() if line.strip()]
    return fallback[:6]


def parse_tool_plan(raw_text: str) -> List[dict]:
    plan = []
    for line in raw_text.splitlines():
        clean = line.strip()
        if not clean or not clean.lower().startswith("tool:"):
            continue
        payload = clean.split(":", 1)[1].strip()
        if "|" not in payload:
            continue
        action, args = payload.split("|", 1)
        action = action.strip()
        args = args.strip()
        if action:
            plan.append({"action": action, "args": args})
    return plan


def query_node(state: AgentState):
    response = model.invoke(
        [
            SystemMessage(content=QUERY_MAKER_PROMPT),
            HumanMessage(content=state["task"]),
        ]
    )
    queries = parse_queries(response.content or "")
    if not queries:
        queries = [f"o que fazer em {state['task']}"]
    return {"queries": queries[:6]}


def search_node(state: AgentState):
    draft = []
    for query in state["queries"]:
        draft.append(
            _first_search_content(
                query=query,
                default_text=f"Nenhum resultado relevante para '{query}'.",
            )
        )
    return {"draft": draft}


def tool_planner_node(state: AgentState):
    response = model.invoke(
        [
            SystemMessage(content=TOOL_PLANNER_PROMPT),
            HumanMessage(content=state["task"]),
        ]
    )
    plan = parse_tool_plan(response.content or "")
    if not plan:
        plan = [
            {"action": "ver_temperatura", "args": "Paris"},
            {"action": "calcular_orcamento", "args": "Paris|5|350"},
            {"action": "sugerir_restaurantes", "args": "Paris"},
            {"action": "preco_passagens", "args": "Paris"},
        ]
    return {"tool_plan": plan[:4]}


def tool_runner_node(state: AgentState):
    outputs = []
    for call in state["tool_plan"]:
        action = str(call.get("action", "")).strip()
        args = str(call.get("args", "")).strip()
        if action not in tools:
            outputs.append(f"{action}: ferramenta nao encontrada.")
            continue
        try:
            result = tools[action](args)
            outputs.append(f"{action}({args}) => {result}")
        except Exception as exc:
            outputs.append(f"{action}({args}) => erro: {exc}")
    return {"tool_results": outputs}


def result_node(state: AgentState):
    draft = "\n\n".join(state["draft"] or [])
    tool_results = "\n".join(state["tool_results"] or [])
    response = model.invoke(
        [
            SystemMessage(
                content=RESULT_PROMPT.format(
                    draft=draft,
                    tool_results=tool_results,
                    task=state["task"],
                )
            ),
            HumanMessage(content="Gere o plano final."),
        ]
    )
    return {"result": response.content}


builder = StateGraph(AgentState)
builder.add_node("query", query_node)
builder.add_node("search", search_node)
builder.add_node("tool_planner", tool_planner_node)
builder.add_node("tool_runner", tool_runner_node)
builder.add_node("result", result_node)

builder.add_edge(START, "query")
builder.add_edge("query", "search")
builder.add_edge("search", "tool_planner")
builder.add_edge("tool_planner", "tool_runner")
builder.add_edge("tool_runner", "result")
builder.add_edge("result", END)

graph = builder.compile()


def run_travel_agent(query: str):
    state = {
        "task": query,
        "queries": [],
        "draft": [],
        "tool_plan": [],
        "tool_results": [],
        "result": "",
    }
    result = graph.invoke(state)
    print("\n=== Pedido ===")
    print(query)
    print("\n=== Ferramentas executadas ===")
    for line in result.get("tool_results", []):
        print("-", line)
    print("\n=== Resposta final ===")
    print(result["result"])


if __name__ == "__main__":
    run_travel_agent(
        "Monte um roteiro para Paris, incluindo orcamento total e sugestoes de restaurantes."
    )
