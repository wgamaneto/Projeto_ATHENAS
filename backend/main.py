import logging
from time import perf_counter

from fastapi import FastAPI
from athenas.core import AthenasRAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ATHENAS MVP")


@app.get("/answer")
async def answer(pergunta: str):
    """Endpoint simples que utiliza o pipeline RAG completo."""
    start_time = perf_counter()
    logger.info("Pergunta recebida: %s", pergunta)
    try:
        rag = AthenasRAG()
        resposta, fontes, tokens = rag.answer(pergunta)
        elapsed = perf_counter() - start_time
        logger.info("Tempo de resposta: %.2fs", elapsed)
        logger.info("Tokens usados: %s", tokens)
        return {
            "pergunta": pergunta,
            "resposta": resposta,
            "fontes": fontes,
            "tokens": tokens,
            "tempo_resposta": elapsed,
        }
    except Exception as exc:
        elapsed = perf_counter() - start_time
        logger.exception("Erro ao processar pergunta: %s", exc)
        logger.info("Tempo até erro: %.2fs", elapsed)
        raise
