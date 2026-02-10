from openai import OpenAI
import os

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),

)

completion = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Você é um expert em história dos LLMs."},
        {"role": "user", "content": "em um paragrafo Conte uma história sobre o desenvolvimento da Inteligência Artificial até a invenção dos LLMs."}
    ],
    temperature=0.1
)

print(completion.choices[0].message.content)
