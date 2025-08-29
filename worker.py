"""Tarefas assíncronas para processamento de documentos."""

import os

from celery import Celery

from ingest import ingest_file


REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

celery_app = Celery("athenas", broker=REDIS_URL, backend=REDIS_URL)


@celery_app.task
def process_document(file_path: str) -> None:
    """Executa a ingestão de um documento no background."""
    ingest_file(file_path)

