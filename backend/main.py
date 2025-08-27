from fastapi import FastAPI
from athenas.core import AthenasRAG

app = FastAPI(title="ATHENAS MVP")


@app.get("/answer")
async def answer(pergunta: str):
    """Endpoint simples que utiliza o pipeline RAG completo."""
    rag = AthenasRAG()
    resposta = rag.answer(pergunta)
    return {"pergunta": pergunta, "resposta": resposta}
