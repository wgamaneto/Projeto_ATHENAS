"""Script de ingestão de documentos para o ChromaDB.

Lê todos os arquivos de ``./knowledge_base/``, divide o texto em pequenos
trechos, gera embeddings e armazena cada chunk em uma coleção do ChromaDB.
"""

from pathlib import Path
import os
import uuid

import chromadb
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

from athenas.core import AthenasRAG


def ingest(folder: str = "./knowledge_base", collection_name: str = "documents") -> None:
    """Lê documentos, gera embeddings e persiste no ChromaDB."""

    load_dotenv()

    # Tenta conectar a um servidor HTTP do ChromaDB; em caso de falha, usa
    # um client local persistente para fins de demonstração.
    try:
        client = chromadb.HttpClient(
            host=os.getenv("CHROMA_HOST", "localhost"),
            port=int(os.getenv("CHROMA_PORT", "8001")),
        )
    except Exception:
        client = chromadb.PersistentClient(path="chroma_db")

    collection = client.get_or_create_collection(collection_name)

    rag = AthenasRAG()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for path in Path(folder).glob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            for i, chunk in enumerate(splitter.split_text(text)):
                embedding = rag.embedder(chunk)
                collection.add(
                    ids=[f"{path.stem}-{i}-{uuid.uuid4()}"],
                    documents=[chunk],
                    metadatas=[{"source": path.name}],
                    embeddings=[embedding],
                )

    print(
        f"Ingestão concluída: {collection.count()} chunks armazenados na coleção '{collection_name}'."
    )


if __name__ == "__main__":
    ingest()

