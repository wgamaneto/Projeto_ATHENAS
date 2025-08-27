from fastapi import FastAPI
from athenas.core import AthenasRAG

app = FastAPI(title="ATHENAS MVP")
rag = AthenasRAG()

# Documentos de exemplo para demonstração
documents = [
    "A ATHENAS é um assistente IA conversacional que unifica o conhecimento interno.",
    "O objetivo inicial é reduzir o tempo médio gasto na busca por informações.",
]

@app.get("/answer")
async def answer(pergunta: str):
    """Endpoint simples que utiliza o pipeline RAG."""
    resposta = rag.answer(pergunta, documents)
    return {"pergunta": pergunta, "resposta": resposta}
