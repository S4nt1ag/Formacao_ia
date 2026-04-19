import os
from typing import Annotated, TypedDict

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages


SYSTEM_PROMPT = (
    "Você é um assistente de IA de uma loja de bicicletas. 🚲\n"
    "Ajude o usuário a encontrar informações sobre produtos (bicicletas, capacetes, luzes, "
    "cadeados, manutenção, tamanhos e recomendações).\n"
    "Seja amigável, prestativo, use emojis e trocadilhos leves.\n"
    "Se você não souber algo, diga claramente que não sabe e sugira o que perguntar ou "
    "quais detalhes faltam."
)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_graph(model: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    chain = prompt | model

    def chatbot(state: ChatState) -> dict:
        response = chain.invoke({"messages": state["messages"]})
        return {"messages": [response]}

    graph = StateGraph(ChatState)
    graph.add_node("chatbot", chatbot)
    graph.set_entry_point("chatbot")
    graph.add_edge("chatbot", END)
    return graph.compile()


def ensure_session_messages():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            AIMessage(
                content="Oi! Eu sou o BikeBot 🚲😄 Me diga o que você procura: uma bike, "
                "um acessório, ou dicas de manutenção?"
            )
        ]


def render_history():
    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(msg.content)
        else:
            with st.chat_message("assistant"):
                st.markdown(str(getattr(msg, "content", msg)))


st.set_page_config(page_title="BikeBot (memória + emojis)", page_icon="🚲")
st.title("BikeBot 🚲")
st.caption("Chatbot com memória (histórico) e tom amigável.")

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    st.error("Defina a variável de ambiente `OPENROUTER_API_KEY` antes de rodar o app.")
    st.stop()

model_name = os.getenv("OPENROUTER_MODEL", "gpt-4o-mini")
llm = ChatOpenAI(
    model=model_name,
    openai_api_key=api_key,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.7,
)
app = build_graph(llm)

ensure_session_messages()
render_history()

user_text = st.chat_input("Pergunte sobre bicicletas, acessórios, tamanhos, manutenção...")
if user_text:
    st.session_state.messages.append(HumanMessage(content=user_text))

    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        result = app.invoke({"messages": st.session_state.messages})
        assistant_msg = result["messages"][-1]
        st.session_state.messages.append(assistant_msg)
        st.markdown(assistant_msg.content)
