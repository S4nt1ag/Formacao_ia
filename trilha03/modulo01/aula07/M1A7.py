from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AnyMessage, SystemMessage, AIMessage, ChatMessage

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from tavily import TavilyClient
import os
from IPython.display import Image, display

import operator
from typing import TypedDict, Annotated, List, Dict
from pydantic import BaseModel

from dotenv import load_dotenv

load_dotenv()

checkpoint = InMemorySaver()

# Get API keys from environment variables
TAVILY_SEARCH_API = os.getenv("TAVILY_API_KEY", "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")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

tavily_client = TavilyClient(api_key=TAVILY_SEARCH_API)

class AgentState(TypedDict):
    task: str
    queries: List[str]
    draft: List[str]
    result: str
    user_feedback: str
    revision_count: int

model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

QUERY_MAKER_PROMPT = """
Você é um especialista em planejamento de viagens.
Você receberá um destino de viagem e deverá montar uma lista de queries para buscar informações 
sobre atividades no destino com base no interesse do usuário.

Gere uma lista de 5-8 queries específicas e relevantes para buscar informações sobre:
- Atividades relacionadas aos interesses do usuário
- Locais específicos para essas atividades
- Melhores épocas para visitar
- Informações práticas (preços, horários, etc.)

Retorne apenas as queries, uma por linha, sem numeração.
"""

RESULT_PROMPT = """
Você é um especialista em planejamento de viagens.

Com base nas informações coletadas, você deve gerar um planejamento de viagem completo e personalizado.
Torne o texto amigável para o usuário, cativante e envolvente.

Estruture o resultado com:
1. INTRODUÇÃO - Resumo do destino e interesses
2. ATIVIDADES PRINCIPAIS - Baseadas nos interesses do usuário
3. LOCAIS RECOMENDADOS - Específicos para as atividades
4. DICAS PRÁTICAS - Informações úteis para a viagem
5. CRONOGRAMA SUGERIDO - Organização da viagem

Informações coletadas:
{draft}

Interesses do usuário: {task}

Gere um planejamento completo e detalhado:
"""

REVISION_PROMPT = """
Você é um especialista em planejamento de viagens.

O usuário forneceu feedback sobre um planejamento de viagem. Você deve revisar e melhorar o planejamento
com base no feedback do usuário, utilizando também as informações coletadas durante a pesquisa.

PLANEJAMENTO ATUAL:
{current_result}

FEEDBACK DO USUÁRIO:
{user_feedback}

INTERESSES ORIGINAIS DO USUÁRIO:
{task}

INFORMAÇÕES COLETADAS DURANTE A PESQUISA:
{draft}

Revise o planejamento considerando o feedback do usuário e as informações disponíveis. 
Utilize as informações da pesquisa para enriquecer o planejamento e atender às solicitações do usuário.
Mantenha a estrutura organizada e faça as modificações necessárias.

Gere um planejamento revisado e melhorado:
"""

class Queries(BaseModel):
    queries: List[str]

def query_node(state: AgentState):
    """Generate search queries based on user interests"""
    messages = [    
        SystemMessage(content=QUERY_MAKER_PROMPT),
        HumanMessage(content=state["task"])
    ]
    response = model.with_structured_output(Queries).invoke(messages)
    return {"queries": response.queries}

def search_node(state: AgentState):
    """Search for information using the generated queries"""
    draft = state['draft'] or []
    for q in state['queries']:
        try:
            response = tavily_client.search(query=q, max_results=3)
            for r in response['results']:
                draft.append(r['content'])
        except Exception as e:
            print(f"Error searching for query '{q}': {e}")
            continue
    return {"draft": draft}

def generate_result_node(state: AgentState):
    """Generate final travel plan based on collected information"""
    draft = "\n\n".join(state['draft'] or [])
    user_message = HumanMessage(
        content=f"Aqui estão o meu destino e interesses: {state['task']}")
    system_message = SystemMessage(
                content=RESULT_PROMPT.format(draft=draft, task=state['task'])
            )
    messages = [system_message, user_message]
    response = model.invoke(messages)
    return {
            "result": response.content
        }

def user_feedback_node(state: AgentState):
    """Get user feedback on the travel plan"""
    print("\n" + "=" * 60)
    print("🎯 SEU PLANO DE VIAGEM:")
    print("=" * 60)
    print(state['result'])
    print("=" * 60)
    
    print("\n💬 Você gostaria de fazer alguma alteração no plano?")
    print("Opções:")
    print("1. Digite suas sugestões de mudança")
    print("2. Digite 'ok' ou 'perfeito' para finalizar")
    print("3. Digite 'sair' para encerrar")
    
    feedback = input("\nSua resposta: ").strip()
    
    if feedback.lower() in ['ok', 'perfeito', 'finalizar', 'done']:
        return {"user_feedback": "Aprovado", "revision_count": state.get('revision_count', 0)}
    elif feedback.lower() in ['sair', 'exit', 'quit']:
        return {"user_feedback": "Sair", "revision_count": state.get('revision_count', 0)}
    else:
        return {"user_feedback": feedback, "revision_count": state.get('revision_count', 0) + 1}

def revision_node(state: AgentState):
    """Revise the travel plan based on user feedback"""
    if state['user_feedback'] in ['Aprovado', 'Sair']:
        return {"result": state['result']}
    
    # Prepare draft content for revision
    draft_content = "\n\n".join(state['draft'] or [])
    
    messages = [
        SystemMessage(content=REVISION_PROMPT.format(
            current_result=state['result'],
            user_feedback=state['user_feedback'],
            task=state['task'],
            draft=draft_content
        )),
        HumanMessage(content="Por favor, revise o planejamento com base no feedback fornecido e nas informações disponíveis.")
    ]
    
    response = model.invoke(messages)
    return {"result": response.content}

def should_continue(state: AgentState):
    """Determine if we should continue the feedback loop"""
    if state['user_feedback'] == 'Sair':
        return "end"
    elif state['user_feedback'] == 'Aprovado':
        return "end"
    elif state.get('revision_count', 0) >= 3:
        print("\n⚠️  Limite de 3 revisões atingido. Finalizando...")
        return "end"
    else:
        return "feedback"

# Build the graph
builder = StateGraph(AgentState)

# Adding nodes
builder.add_node("query", query_node)
builder.add_node("search", search_node)
builder.add_node("result", generate_result_node)
builder.add_node("feedback", user_feedback_node)
builder.add_node("revision", revision_node)

# Adding edges
builder.add_edge(START, "query")
builder.add_edge("query", "search")
builder.add_edge("search", "result")
builder.add_edge("result", "feedback")
builder.add_conditional_edges(
    "feedback",
    should_continue,
    {
        "feedback": "revision",
        "end": END
    }
)
builder.add_edge("revision", "feedback")

# Compile the graph
graph = builder.compile(checkpointer=checkpoint)

# Try to display the graph visualization
def display_graph():
    """Display the graph visualization if possible"""
    try:
        # Try to generate and display the graph
        graph_image = graph.get_graph().draw_mermaid_png()
        display(Image(graph_image))
        print("✅ Graph visualization generated successfully!")
    except Exception as e:
        print(f"⚠️  Could not generate graph visualization: {e}")
        print("This is optional and doesn't affect the functionality of the travel planner.")
        print("To enable graph visualization, install: pip install pygraphviz")

# Display the graph
display_graph()

# Run the travel planner
def run_travel_planner():
    """Run the travel planner with a sample query"""
    print("\n🚀 Starting Travel Planner with Human-in-the-Loop...")
    print("=" * 50)
    
    thread = {"configurable": {"thread_id": "1"}}
    
    initial_state = {
        'task': "Gostaria de viajar para o Japão, gosto de surfar, fazer esportes radicais e adoro cerveja.",
        'draft': [],
        'queries': [],
        'result': "",
        'user_feedback': "",
        'revision_count': 0
    }
    
    print("📋 Processing your travel request...")
    print("\n" + "=" * 50)
    
    for step in graph.stream(initial_state, thread):
        if 'query' in step:
            print(f"🔍 Generated {len(step['query']['queries'])} search queries")
        elif 'search' in step:
            print(f"📚 Collected {len(step['search']['draft'])} information sources")
        elif 'result' in step:
            print("✅ Initial travel plan generated!")
        elif 'feedback' in step:
            if step['feedback']['user_feedback'] == 'Aprovado':
                print("\n🎉 Plano aprovado! Obrigado por usar o Travel Planner!")
            elif step['feedback']['user_feedback'] == 'Sair':
                print("\n👋 Obrigado por usar o Travel Planner!")
            else:
                print(f"🔄 Revisão #{step['feedback']['revision_count']} solicitada...")
        elif 'revision' in step:
            print("✅ Plano revisado com base no seu feedback e informações da pesquisa!")

if __name__ == "__main__":
    run_travel_planner()