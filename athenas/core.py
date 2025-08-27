"""Prototipo inicial do mecanismo RAG da ATHENAS."""

from typing import Iterable, List


class AthenasRAG:
    """Estrutura simples de Retrieval-Augmented Generation para o MVP da ATHENAS.

    O pipeline segue três etapas: geração de embeddings, busca e síntese da resposta.
    As implementações aqui são placeholders que deverão ser substituídas por
    modelos reais e integrações conforme evolução do projeto.
    """

    def __init__(self, embedder=None, retriever=None, generator=None):
        self.embedder = embedder or self._openai_embedder
        self.retriever = retriever or self._chroma_retriever
        self.generator = generator or self._openai_generator

    def _openai_embedder(self, text: str) -> List[float]:
        """Gera embeddings reais usando a API da OpenAI."""
        from openai import OpenAI
        from dotenv import load_dotenv

        load_dotenv()
        client = OpenAI()
        response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002",
        )
        return response.data[0].embedding

    def _chroma_retriever(
        self, query: str, collection_name: str = "documents", top_k: int = 3
    ) -> Iterable[str]:
        """Consulta o ChromaDB para recuperar documentos relevantes."""
        import os
        from dotenv import load_dotenv
        import chromadb

        load_dotenv()
        client = chromadb.HttpClient(
            host=os.getenv("CHROMA_HOST", "localhost"),
            port=int(os.getenv("CHROMA_PORT", "8001")),
        )

        embedding = self.embedder(query)
        collection = client.get_collection(collection_name)
        results = collection.query(query_embeddings=[embedding], n_results=top_k)
        return results.get("documents", [[]])[0]

    def _openai_generator(self, query: str, context: str) -> str:
        """Gera uma resposta usando um LLM real baseado no contexto."""

        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv()
        client = OpenAI()

        messages = [
            {
                "role": "system",
                "content": (
                    "Responda à pergunta do usuário utilizando apenas o contexto "
                    "fornecido. Se a resposta não estiver presente no contexto, "
                    "informe que não sabe."
                ),
            },
            {
                "role": "user",
                "content": f"Contexto:\n{context}\n\nPergunta: {query}",
            },
        ]

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )

        return completion.choices[0].message.content.strip()

    def answer(self, query: str) -> str:
        """Executa o pipeline completo de pergunta e resposta."""
        relevant = next(iter(self.retriever(query)), "")
        return self.generator(query, relevant)
