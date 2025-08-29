"""Prototipo inicial do mecanismo RAG da ATHENAS."""

from typing import Dict, Iterable, List, Tuple


class AthenasRAG:
    """Estrutura simples de Retrieval-Augmented Generation para o MVP da ATHENAS.

    O pipeline segue três etapas: geração de embeddings, busca e síntese da resposta.
    As implementações aqui são placeholders que deverão ser substituídas por
    modelos reais e integrações conforme evolução do projeto.
    """

    def __init__(
        self,
        embedder=None,
        retriever=None,
        generator=None,
        reranker=None,
        summarizer=None,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.embedder = embedder or self._openai_embedder
        self.retriever = retriever or self._chroma_retriever
        self.generator = generator or self._openai_generator
        self.reranker = reranker or self._cross_encoder_rerank
        self.summarizer = summarizer or self._openai_summarizer
        self._cross_encoder_model_name = cross_encoder_model
        self._cross_encoder = None

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
    ) -> Iterable[Dict[str, str]]:
        """Consulta o ChromaDB para recuperar documentos relevantes.

        Retorna uma lista de dicionários com o texto do *chunk* e a fonte
        original do arquivo.
        """
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
        results = collection.query(
            query_embeddings=[embedding], n_results=top_k, include=["documents", "metadatas"]
        )
        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        if docs and self.reranker:
            ranked_docs = self.reranker(query, docs)
            meta_map = {doc: meta for doc, meta in zip(docs, metadatas)}
            docs = ranked_docs
            metadatas = [meta_map[doc] for doc in docs]

        return [
            {"texto": doc, "fonte": meta.get("source", "")}
            for doc, meta in zip(docs, metadatas)
        ]

    def _cross_encoder_rerank(self, query: str, docs: List[str]) -> List[str]:
        """Reordena documentos usando um modelo cross-encoder."""
        from sentence_transformers import CrossEncoder

        if self._cross_encoder is None:
            self._cross_encoder = CrossEncoder(self._cross_encoder_model_name)

        pairs = [[query, doc] for doc in docs]
        scores = self._cross_encoder.predict(pairs)
        ranked = [doc for _, doc in sorted(zip(scores, docs), reverse=True)]
        return ranked

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

    def _openai_summarizer(self, query: str, document: str) -> str:
        """Resume um documento com foco na pergunta do usuário."""

        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv()
        client = OpenAI()

        messages = [
            {
                "role": "system",
                "content": (
                    "Resuma o texto fornecido destacando apenas as partes que "
                    "ajudem a responder à pergunta do usuário."
                ),
            },
            {
                "role": "user",
                "content": f"Pergunta: {query}\n\nTexto:\n{document}",
            },
        ]

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )

        return completion.choices[0].message.content.strip()

    def answer(self, query: str) -> Tuple[str, List[Dict[str, str]]]:
        """Executa o pipeline de pergunta e resposta retornando também as fontes."""
        relevant_docs = list(self.retriever(query))
        summaries = [self.summarizer(query, doc["texto"]) for doc in relevant_docs]
        context = "\n\n".join(summaries)
        resposta = self.generator(query, context)
        return resposta, relevant_docs
