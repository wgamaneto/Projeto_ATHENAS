"""Script de ingestão de documentos para o ChromaDB.

Lê arquivos em ``./knowledge_base/`` ou um caminho específico, divide o texto
em pequenos trechos, gera embeddings e persiste cada chunk em uma coleção do
ChromaDB. Também constrói ou atualiza um índice BM25 para busca lexical.
"""

from pathlib import Path
import os
import uuid
import pickle

import chromadb
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from rank_bm25 import BM25Okapi

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

from athenas.core import AthenasRAG


analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()


def _load_file(path: Path):
    """Carrega um arquivo em um formato suportado usando loaders do LangChain."""
    if path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(path))
    else:
        loader = TextLoader(str(path), encoding="utf-8")
    return loader.load()


def _anonymize(text: str) -> str:
    """Anonimiza PII usando o Microsoft Presidio."""
    results = analyzer.analyze(text=text, language="en")
    return anonymizer.anonymize(text=text, analyzer_results=results).text


def ingest_file(
    file_path: str,
    collection_name: str = "documents",
    allowed_groups: list[str] | None = None,
) -> None:
    """Ingere um único arquivo para o ChromaDB e atualiza o índice BM25.

    Cada chunk recebe metadados com o nome do arquivo de origem e os grupos
    autorizados a acessá-lo. Os grupos podem ser informados via parâmetro ou,
    caso não sejam fornecidos, serão lidos da variável ``DEFAULT_ALLOWED_GROUPS``
    (separados por vírgula)."""

    load_dotenv()

    if allowed_groups is None:
        default = os.getenv("DEFAULT_ALLOWED_GROUPS", "public")
        allowed_groups = [g.strip() for g in default.split(",") if g.strip()]

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

    bm25_docs: list[str] = []
    bm25_metas: list[dict] = []
    bm25_corpus: list[list[str]] = []

    path = Path(file_path)
    documents = _load_file(path)
    for doc in documents:
        sanitized = _anonymize(doc.page_content)
        metadata = {"source": path.name, "allowed_groups": allowed_groups}
        sanitized_metadata = {k: str(v) for k, v in metadata.items()}
        for i, chunk in enumerate(splitter.split_text(sanitized)):
            embedding = rag.embedder(chunk)
            collection.add(
                ids=[f"{path.stem}-{i}-{uuid.uuid4()}"],
                documents=[chunk],
                metadatas=[sanitized_metadata],
                embeddings=[embedding],
            )
            bm25_docs.append(chunk)
            bm25_metas.append(metadata)
            bm25_corpus.append(chunk.split())

    if os.path.exists("bm25_index.pkl"):
        with open("bm25_index.pkl", "rb") as f:
            data = pickle.load(f)
        bm25_docs = data["documents"] + bm25_docs
        bm25_metas = data["metadatas"] + bm25_metas
        bm25_corpus = [doc.split() for doc in data["documents"]] + bm25_corpus

    bm25 = BM25Okapi(bm25_corpus)
    with open("bm25_index.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "documents": bm25_docs, "metadatas": bm25_metas}, f)

    print(f"Ingestão do arquivo {path.name} concluída.")


def ingest(folder: str = "./knowledge_base", collection_name: str = "documents") -> None:
    """Ingere todos os arquivos de uma pasta."""

    for path in Path(folder).glob("*"):
        if path.is_file():
            ingest_file(str(path), collection_name)


if __name__ == "__main__":
    ingest()

