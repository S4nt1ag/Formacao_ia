import os
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)

parser = StrOutputParser()


system_template = "Traduza o seguinte texto de Inglês para {idioma}"
template_mensagem = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", "{user_text}")
])

chain = template_mensagem | llm | parser

text1 = chain.invoke({"idioma": "inglês", "user_text": "bom dia"})
text2 = chain.invoke({"idioma": "frances", "user_text": "bom dia"})
text3 = chain.invoke({"idioma": "espanhol", "user_text": "bom dia"})
text4 = chain.invoke({"idioma": "português", "user_text": "bom dia"})
text5 = chain.invoke({"idioma": "japonês", "user_text": "bom dia"})
text6 = chain.invoke({"idioma": "alemão", "user_text": "bom dia"})

print(text1)
print(text2)
print(text3)
print(text4)
print(text5)
print(text6)