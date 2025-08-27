from fastapi import FastAPI
from athenas.core import AthenasRAG

app = FastAPI(title="ATHENAS MVP")
rag = AthenasRAG()

@app.get("/answer")
async def answer(pergunta: str):
    """Endpoint simples que utiliza o pipeline RAG."""
    resposta = rag.answer(pergunta)
    return {"pergunta": pergunta, "resposta": resposta}
