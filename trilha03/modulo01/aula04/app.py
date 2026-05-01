import operator
import os
from pathlib import Path
from typing import Annotated, TypedDict

try:
    from langchain_community.tools.tavily_search import TavilySearchResults
    from langchain_core.messages import (
        AnyMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, StateGraph
except ImportError as exc:
    raise SystemExit(
        "Dependências ausentes. Instale com: "
        "pip install langgraph langchain-openai langchain-community"
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


def require_env_value(var_names: list[str], env_map: dict, error_message: str) -> str:
    for name in var_names:
        value = os.getenv(name) or env_map.get(name)
        if value and value.strip():
            return value.strip()
    raise SystemExit(error_message)


openrouter_api_key = require_env_value(
    ["OPENROUTER_API_KEY", "OPENAI_API_KEY"],
    dotenv_data,
    f"Defina OPENROUTER_API_KEY (ou OPENAI_API_KEY) no arquivo {PROJECT_ENV_PATH}.",
)
tavily_api_key = require_env_value(
    ["TAVILY_API_KEY", "TAVILY_SEARCH_API"],
    dotenv_data,
    (
        "Defina TAVILY_API_KEY (ou TAVILY_SEARCH_API) no arquivo "
        f"{PROJECT_ENV_PATH}."
    ),
)

model_name = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
temperature = float(os.getenv("OPENROUTER_TEMPERATURE", "0"))

llm = ChatOpenAI(
    model=model_name,
    temperature=temperature,
    openai_api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
)
search_tool = TavilySearchResults(max_results=3, tavily_api_key=tavily_api_key)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


class Agent:
    def __init__(self, model: ChatOpenAI, tools: list, system: str = ""):
        self.system = system
        self.tools = {tool.name: tool for tool in tools}
        self.model = model.bind_tools(tools)

        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_llm)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()

    def exists_action(self, state: AgentState) -> bool:
        result = state["messages"][-1]
        return len(result.tool_calls) > 0

    def call_llm(self, state: AgentState) -> dict:
        messages = state["messages"]
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {"messages": [message]}

    def take_action(self, state: AgentState) -> dict:
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for call in tool_calls:
            tool_name = call["name"]
            print(f"Chamando ferramenta: {tool_name} com args={call['args']}")

            if tool_name not in self.tools:
                result = "Nome de ferramenta inválido, tente novamente."
            else:
                result = self.tools[tool_name].invoke(call["args"])

            results.append(
                ToolMessage(
                    tool_call_id=call["id"],
                    name=tool_name,
                    content=str(result),
                )
            )
        print("Retornando observações para o modelo...")
        return {"messages": results}


SYSTEM_PROMPT = """
Você é um assistente de pesquisa inteligente.
Use o motor de busca quando precisar confirmar fatos.
Você pode fazer múltiplas buscas em sequência ou em paralelo.
Se a pergunta for factual, prefira validar na web antes da resposta final.
""".strip()


def run_query(agent: Agent, question: str) -> str:
    messages = [HumanMessage(content=question)]
    result = agent.graph.invoke({"messages": messages})
    return result["messages"][-1].content


if __name__ == "__main__":
    bot = Agent(llm, [search_tool], system=SYSTEM_PROMPT)

    test_questions = [
        "Qual a capital do Canadá?",
        "Quem ganhou a Copa do Mundo de 2014?",
    ]

    for idx, question in enumerate(test_questions, start=1):
        print(f"\nTeste {idx}: {question}")
        answer = run_query(bot, question)
        print("Resposta:", answer)
