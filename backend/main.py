import logging
from pathlib import Path
from time import perf_counter, time
from typing import Any, Dict, Tuple

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
_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}


@app.get("/answer")
async def answer(pergunta: str):
    """Endpoint simples que utiliza o pipeline RAG completo.

    Utiliza um cache em memória para evitar chamadas repetidas à API da OpenAI.
    """
    start_time = perf_counter()
    logger.info("Pergunta recebida: %s", pergunta)

    now = time()
    cached = _cache.get(pergunta)
    if cached and now - cached[0] < CACHE_TTL:
        logger.info("Resposta retornada do cache")
        elapsed = perf_counter() - start_time
        return {"pergunta": pergunta, **cached[1], "tempo_resposta": elapsed}
    elif cached:
        del _cache[pergunta]

    try:
        rag = AthenasRAG()
        resposta, fontes, tokens = rag.answer(pergunta)
        elapsed = perf_counter() - start_time
        logger.info("Tempo de resposta: %.2fs", elapsed)
        logger.info("Tokens usados: %s", tokens)
        _cache[pergunta] = (now, {"resposta": resposta, "fontes": fontes, "tokens": tokens})
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
