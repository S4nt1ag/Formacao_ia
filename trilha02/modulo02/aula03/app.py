# Teste de Document Loaders do LangChain
# Arquivos: PDF e HTML

from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader
from bs4 import BeautifulSoup

# ============================================
# 1. CARREGAR PDF
# ============================================
print("=" * 60)
print("CARREGANDO PDF: manual-do-ciclista_digital.pdf")
print("=" * 60)

loader_pdf = PyPDFLoader("E:/Formacao_ia/trilha02/PUBLIC/manual-do-ciclista_digital.pdf")
documentos_pdf = loader_pdf.load()

print(f"\n📊 Quantidade de Documents criados: {len(documentos_pdf)}")

print("\n--- Primeiro Documento ---")
print(f"\n📝 Texto (primeiros 500 caracteres):")
print(documentos_pdf[0].page_content[:500])

print(f"\n📋 Metadados do primeiro documento:")
for chave, valor in documentos_pdf[0].metadata.items():
    print(f"  {chave}: {valor}")

# ============================================
# 2. CARREGAR HTML
# ============================================
print("\n" + "=" * 60)
print("CARREGANDO HTML: ciclismo.html")
print("=" * 60)

# Ler o HTML manualmente com encoding UTF-8
with open("E:/Formacao_ia/trilha02/PUBLIC/ciclismo.html", "r", encoding="utf-8") as f:
    html_content = f.read()

# Usar BeautifulSoup para parsear
soup = BeautifulSoup(html_content, "lxml")

# Extrair texto de elementos relevantes
textos = []
for tag in soup.find_all(["h1", "h2", "h3", "p", "div"]):
    if tag.get_text(strip=True):
        textos.append(tag.get_text(strip=True))

# Criar documento único com todo o texto
from langchain_core.documents import Document
documentos_html = [Document(
    page_content="\n\n".join(textos),
    metadata={"source": "E:/Formacao_ia/trilha02/PUBLIC/ciclismo.html", "title": soup.title.string if soup.title else "N/A"}
)]

print(f"\n📊 Quantidade de Documents criados: {len(documentos_html)}")

print("\n--- Primeiro Documento ---")
print(f"\n📝 Texto (primeiros 500 caracteres):")
print(documentos_html[0].page_content[:500])

print(f"\n📋 Metadados do primeiro documento:")
for chave, valor in documentos_html[0].metadata.items():
    print(f"  {chave}: {valor}")

# ============================================
# 3. COMPARAÇÃO DOS RESULTADOS
# ============================================
print("\n" + "=" * 60)
print("COMPARAÇÃO DOS RESULTADOS")
print("=" * 60)

print("""
📌 PDF (PyPDFLoader):
   - Organização: O PDF é dividido por páginas. Cada página = 1 Document.
   - Metadados: source (caminho do arquivo), page (número da página)

📌 HTML (BSHTMLLoader):
   - Organização: O HTML é dividido por sections/divs. Cada section = 1 Document.
   - Metadados: source (caminho do arquivo), title (título da página se disponível)
""")