import os
import re
from datetime import datetime
from pathlib import Path

try:
    from langchain_openai import ChatOpenAI
except ImportError as exc:
    raise SystemExit(
        "Dependência ausente: langchain-openai. Instale com: pip install langchain-openai"
    ) from exc

PROJECT_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def load_env_file(path: Path) -> dict:
    env_data = {}
    if not path.exists():
        return env_data

    # utf-8-sig remove BOM automaticamente, evitando erro em chaves.
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        env_data[key] = value
    return env_data


dotenv_data = load_env_file(PROJECT_ENV_PATH)
for key, value in dotenv_data.items():
    os.environ[key] = value

api_key = (
    os.getenv("OPENROUTER_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or dotenv_data.get("OPENROUTER_API_KEY")
    or dotenv_data.get("OPENAI_API_KEY")
)
if api_key:
    api_key = api_key.strip()
if not api_key:
    raise SystemExit(
        f"Defina OPENROUTER_API_KEY (ou OPENAI_API_KEY) no arquivo {PROJECT_ENV_PATH}."
    )

model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
temperature = float(os.getenv("OPENROUTER_TEMPERATURE", "0"))
llm = ChatOpenAI(
    model=model,
    temperature=temperature,
    openai_api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
)


class Agent:
    def __init__(self, system: str = ""):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message: str) -> str:
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self) -> str:
        response = llm.invoke(self.messages)
        return response.content


def calculate(formula: str):
    return eval(formula)


def preco_prato(nome: str) -> str:
    if nome == "Feijoada":
        return "Uma Feijoada custa R$ 75,90"
    if nome == "Moqueca":
        return "Uma Moqueca custa R$ 89,90"
    if nome == "Picanha":
        return "Uma Picanha custa R$ 129,90"
    return "Prato não encontrado no cardápio"


def calcular_idade(ano_nascimento: str) -> str:
    ano = int(ano_nascimento.strip())
    ano_atual = datetime.now().year
    idade = ano_atual - ano
    return f"Alguém que nasceu em {ano} tem {idade} anos em {ano_atual}."


def converter_moeda(entrada: str) -> str:
    valor_usd, taxa = [parte.strip() for parte in entrada.split(",")]
    valor_usd = float(valor_usd)
    taxa = float(taxa)
    valor_brl = valor_usd * taxa
    return f"US$ {valor_usd:.2f} equivale a R$ {valor_brl:.2f} com taxa {taxa:.2f}."


prompt = """
Você executa em um ciclo de Pensamento, Ação, PAUSA, Observação.
No final do ciclo você fornece uma Resposta.
Use Pensamento para descrever seus pensamentos sobre a pergunta que foi feita.
Use Ação para executar uma das ações disponíveis - então retorne PAUSA.
Observação será o resultado da execução dessas ações.

Suas ações disponíveis são:

calculate:
ex: Ação: calculate: 4 * 7 / 3
Executa um cálculo com sintaxe Python.

preco_prato:
ex: Ação: preco_prato: Feijoada
Retorna o preço do prato quando fornecido o nome.

calcular_idade:
ex: Ação: calcular_idade: 1995
Retorna a idade de quem nasceu no ano informado.

converter_moeda:
ex: Ação: converter_moeda: 10, 5
Entrada deve ser "valor_em_usd, taxa_brl_por_usd".
Retorna a conversão de USD para BRL.

Exemplo de sessão:
Pergunta: Quanto custa uma Moqueca?
Pensamento: Devo verificar o preço da Moqueca usando preco_prato.
Ação: preco_prato: Moqueca
PAUSA

Você será chamado novamente com isto:
Observação: Uma Moqueca custa R$ 89,90

Você então fornece:
Resposta: Uma Moqueca custa R$ 89,90
""".strip()


known_actions = {
    "calculate": calculate,
    "preco_prato": preco_prato,
    "calcular_idade": calcular_idade,
    "converter_moeda": converter_moeda,
}

action_re = re.compile(r"^Ação: (\w+): (.*)$")


def query(question: str, max_turns: int = 5):
    i = 0
    bot = Agent(prompt)
    next_prompt = question
    while i < max_turns:
        i += 1
        result = bot(next_prompt)
        print(result)
        actions = [
            action_re.match(linha)
            for linha in result.split("\n")
            if action_re.match(linha)
        ]
        if actions:
            action, action_input = actions[0].groups()
            if action not in known_actions:
                raise Exception(f"Ação desconhecida: {action}: {action_input}")
            print(f"-- executando {action} {action_input}")
            observation = known_actions[action](action_input)
            print("Observação:", observation)
            next_prompt = f"Observação: {observation}"
        else:
            return


if __name__ == "__main__":
    print("Teste 1: idade")
    query("Quantos anos tem alguém que nasceu em 1995?")

    print("\nTeste 2: conversão de moeda")
    query(
        "Quanto é 10 dólares em reais, considerando que 1 USD = 5,00 BRL?"
    )
