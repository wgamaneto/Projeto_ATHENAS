"""Prototipo inicial do mecanismo RAG da ATHENAS."""

import os
from typing import Dict, Iterable, List, Tuple
from dotenv import load_dotenv

load_dotenv()


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
        cross_encoder_model: str = os.getenv(
            "CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ),
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

        client = OpenAI()
        response = client.embeddings.create(
            input=text,
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"),
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

    def _openai_generator(self, query: str, context: str) -> Tuple[str, int]:
        """Gera uma resposta usando um LLM real baseado no contexto.

        Retorna a resposta gerada e a quantidade de tokens utilizada."""

        from openai import OpenAI

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
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=messages,
        )

        resposta = completion.choices[0].message.content.strip()
        tokens = getattr(getattr(completion, "usage", None), "total_tokens", 0)
        return resposta, tokens

    def _openai_summarizer(self, query: str, document: str) -> Tuple[str, int]:
        """Resume um documento com foco na pergunta do usuário.

        Retorna o resumo e a quantidade de tokens utilizada."""

        from openai import OpenAI

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
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=messages,
        )

        resumo = completion.choices[0].message.content.strip()
        tokens = getattr(getattr(completion, "usage", None), "total_tokens", 0)
        return resumo, tokens

    def answer(self, query: str) -> Tuple[str, List[Dict[str, str]], int]:
        """Executa o pipeline de pergunta e resposta retornando também as fontes e tokens."""
        relevant_docs = list(self.retriever(query))
        summaries = []
        total_tokens = 0
        for doc in relevant_docs:
            resumo, tokens = self.summarizer(query, doc["texto"])
            summaries.append(resumo)
            total_tokens += tokens
        context = "\n\n".join(summaries)
        resposta, tokens = self.generator(query, context)
        total_tokens += tokens
        return resposta, relevant_docs, total_tokens
