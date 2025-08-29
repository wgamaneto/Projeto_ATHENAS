import json
import logging
import os
from pathlib import Path
from time import perf_counter

import redis.asyncio as redis
from fastapi import FastAPI
from athenas.core import AthenasRAG


LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "backend.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

app = FastAPI(title="ATHENAS MVP")


CACHE_TTL = 300  # segundos
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
redis_client = redis.from_url(REDIS_URL, decode_responses=True)


@app.get("/answer")
async def answer(pergunta: str):
    """Endpoint simples que utiliza o pipeline RAG completo.

    Utiliza um cache Redis para evitar chamadas repetidas à API da OpenAI.
    """
    start_time = perf_counter()
    logger.info("Pergunta recebida: %s", pergunta)

    cached = await redis_client.get(pergunta)
    if cached:
        logger.info("Resposta retornada do cache")
        cached_data = json.loads(cached)
        elapsed = perf_counter() - start_time
        return {"pergunta": pergunta, **cached_data, "tempo_resposta": elapsed}

    try:
        rag = AthenasRAG()
        resposta, fontes, tokens = rag.answer(pergunta)
        elapsed = perf_counter() - start_time
        logger.info("Tempo de resposta: %.2fs", elapsed)
        logger.info("Tokens usados: %s", tokens)
        data = {"resposta": resposta, "fontes": fontes, "tokens": tokens}
        await redis_client.setex(pergunta, CACHE_TTL, json.dumps(data))
        return {"pergunta": pergunta, **data, "tempo_resposta": elapsed}
    except Exception as exc:
        elapsed = perf_counter() - start_time
        logger.exception("Erro ao processar pergunta: %s", exc)
        logger.info("Tempo até erro: %.2fs", elapsed)
        raise
