import json
import logging
import os
import sys
from pathlib import Path
from time import perf_counter

import redis.asyncio as redis
from fastapi import FastAPI, File, UploadFile
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

sys.path.append(str(Path(__file__).resolve().parent.parent))
from worker import process_document

app = FastAPI(title="ATHENAS MVP")


CACHE_TTL = 300  # segundos
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
redis_client = redis.from_url(REDIS_URL, decode_responses=True)


@app.get("/answer")
async def answer(
    pergunta: str,
    historico: str | None = None,
    grupos: str | None = None,
):
    """Endpoint simples que utiliza o pipeline RAG completo.

    Recebe a pergunta do usuário, histórico da conversa e, opcionalmente, os
    grupos aos quais o usuário pertence. As respostas retornadas consideram as
    permissões desses grupos. Utiliza um cache Redis para evitar chamadas
    repetidas à API da OpenAI."""
    start_time = perf_counter()
    logger.info("Pergunta recebida: %s", pergunta)
    cache_key = json.dumps(
        {"pergunta": pergunta, "historico": historico, "grupos": grupos},
        sort_keys=True,
    )

    cached = await redis_client.get(cache_key)
    if cached:
        logger.info("Resposta retornada do cache")
        cached_data = json.loads(cached)
        elapsed = perf_counter() - start_time
        return {"pergunta": pergunta, **cached_data, "tempo_resposta": elapsed}

    try:
        rag = AthenasRAG()
        historico_list = json.loads(historico) if historico else []
        grupos_list = [g.strip() for g in grupos.split(",") if g.strip()] if grupos else None
        resposta, fontes, tokens = rag.answer(pergunta, historico_list, grupos_list)
        elapsed = perf_counter() - start_time
        logger.info("Tempo de resposta: %.2fs", elapsed)
        logger.info("Tokens usados: %s", tokens)
        data = {"resposta": resposta, "fontes": fontes, "tokens": tokens}
        await redis_client.setex(cache_key, CACHE_TTL, json.dumps(data))
        return {"pergunta": pergunta, **data, "tempo_resposta": elapsed}
    except Exception as exc:
        elapsed = perf_counter() - start_time
        logger.exception("Erro ao processar pergunta: %s", exc)
        logger.info("Tempo até erro: %.2fs", elapsed)
        raise


@app.post("/documentos")
async def upload_documento(file: UploadFile = File(...)):
    """Recebe um arquivo e agenda a ingestão assíncrona."""
    dest = Path(__file__).resolve().parent.parent / "knowledge_base" / file.filename
    with open(dest, "wb") as f:
        f.write(await file.read())
    process_document.delay(str(dest))
    return {"status": "queued", "filename": file.filename}

