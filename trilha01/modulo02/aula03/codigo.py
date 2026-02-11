import os
import uuid
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph




model = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)

parser = StrOutputParser()

workflow = StateGraph(state_schema=MessagesState)

def call_model(state: MessagesState):
    response = model.invoke(state['messages'])
    return {'messages': response}

workflow.add_edge(START, 'model')
workflow.add_node('model', call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

thread_id = uuid.uuid4()
config = {'configurable': {'thread_id': thread_id}}

query = "Olá, eu sou o Santiago"

input_messages = [HumanMessage(query)]
output = app.invoke({'messages': input_messages}, config)
output['messages'][-1].pretty_print()

query = "como eu me chamo?"

input_messages = [HumanMessage(query)]
output = app.invoke({'messages': input_messages}, config)
output['messages'][-1].pretty_print()


# O gerenciamento de estado é igualmente importante para aplicações interativas, pois permite que o sistema mantenha dados relevantes entre as interações. 
# Isso inclui informações sobre preferências do usuário, o progresso em tarefas específicas ou mesmo respostas anteriores. 
# Um gerenciamento de estado eficaz minimiza repetições desnecessárias e reduz a frustração do usuário, garantindo que a interação seja mais personalizada e eficiente. 
# Em suma, a combinação de histórico de mensagens e gerenciamento de estado contribui significativamente para a criação de experiências de conversação mais coerentes e intuitivas em aplicativos baseados em LLMs.