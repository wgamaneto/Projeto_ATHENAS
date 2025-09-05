"""Prototipo inicial do mecanismo RAG da ATHENAS."""

import json
import os
import pickle
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple, Optional
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

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
        question_rewriter=None,
        presentation_generator=None,
        cross_encoder_model: str = os.getenv(
            "CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ),
        bm25_index_path: str = "bm25_index.pkl",
    ):
        self.embedder = embedder or self._local_embedder
        self.retriever = retriever or self._chroma_retriever
        self.generator = generator or self._gemini_generator
        self.reranker = reranker or self._cross_encoder_rerank
        self.summarizer = summarizer or self._openai_summarizer
        self.question_rewriter = question_rewriter or self._openai_question_rewriter
        self.presentation_generator = presentation_generator or self._presentation_generator
        self._cross_encoder_model_name = cross_encoder_model
        self._cross_encoder = None
        self._bm25 = None
        self._bm25_docs: List[str] = []
        self._bm25_metas: List[Dict] = []
        self._bm25_index_path = bm25_index_path
        self._embedding_model_path = os.getenv(
            "LOCAL_EMBEDDING_MODEL", "local_embedding_model"
        )
        self._embedding_model = None

    # ------------------------------------------------------------------
    # Router e ferramentas
    # ------------------------------------------------------------------

    def _router(self, pergunta: str) -> Tuple[str, Dict[str, str]]:
        """Decide qual ferramenta utilizar para responder à pergunta."""
        from openai import OpenAI

        client = OpenAI()
        system_prompt = (
            "Você é um roteador de intenções. Escolha qual ferramenta da ATHENAS "
            "deve ser usada para atender a solicitação do usuário. As opções são:\n"
            "- ferramenta_busca_documentos: responde perguntas utilizando a base de documentos. Parâmetros: {\"query\": texto}\n"
            "- ferramenta_gerar_resumo_executivo: cria um resumo sobre um tópico. Parâmetros: {\"topic\": texto}\n"
            "- ferramenta_analisar_feedback: analisa os feedbacks do sistema. Parâmetros: {}\n"
            "Responda apenas em JSON com os campos 'ferramenta' e 'parametros'."
        )

        completion = client.chat.completions.create(
            model=os.getenv("OPENAI_ROUTER_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": pergunta},
            ],
            temperature=0,
        )

        try:
            decisao = json.loads(completion.choices[0].message.content)
            ferramenta = decisao.get("ferramenta", "ferramenta_busca_documentos")
            params = decisao.get("parametros", {})
        except Exception:
            ferramenta = "ferramenta_busca_documentos"
            params = {"query": pergunta}
        return ferramenta, params

    def executar(self, pergunta: str) -> Dict[str, object]:
        """Executa o roteador e chama a ferramenta adequada."""
        ferramenta, params = self._router(pergunta)
        func = getattr(self, ferramenta, None)
        if not func:
            return {"erro": f"Ferramenta '{ferramenta}' não encontrada."}
        resultado = func(**params)
        return {"ferramenta": ferramenta, **resultado}

    # -------------------- Ferramentas ---------------------------------

    def ferramenta_busca_documentos(self, query: str) -> Dict[str, object]:
        """Wrapper do fluxo RAG tradicional."""
        resposta, fontes, tokens = self.answer(query)
        return {"resposta": resposta, "fontes": fontes, "tokens": tokens}

    def ferramenta_gerar_resumo_executivo(self, topic: str) -> Dict[str, object]:
        """Gera um resumo executivo sobre um tópico."""
        from openai import OpenAI

        client = OpenAI()
        completion = client.chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "system",
                    "content": "Você produz resumos executivos claros e objetivos.",
                },
                {"role": "user", "content": f"Gere um resumo executivo sobre: {topic}"},
            ],
        )
        resumo = completion.choices[0].message.content.strip()
        tokens = getattr(getattr(completion, "usage", None), "total_tokens", 0)
        return {"resumo": resumo, "tokens": tokens}

    def ferramenta_analisar_feedback(self) -> Dict[str, object]:
        """Retorna estatísticas simples de feedback dos usuários."""
        from analyze_feedback import read_feedback_log

        positivos, negativos, _, _ = read_feedback_log()
        total = positivos + negativos
        pct_pos = (positivos / total * 100) if total else 0
        pct_neg = (negativos / total * 100) if total else 0
        resumo = (
            f"Feedbacks positivos: {positivos} ({pct_pos:.2f}%) | "
            f"Feedbacks negativos: {negativos} ({pct_neg:.2f}%)"
        )
        return {"resumo": resumo, "tokens": 0}

    def _local_embedder(self, text: str) -> List[float]:
        """Gera embeddings usando um modelo SentenceTransformer local."""
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(self._embedding_model_path)
        return self._embedding_model.encode(text).tolist()

    def _load_bm25_index(self) -> None:
        """Carrega o índice BM25 persistido durante a ingestão."""
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi

            try:
                with open(self._bm25_index_path, "rb") as f:
                    data = pickle.load(f)
                self._bm25 = data.get("bm25")
                self._bm25_docs = data.get("documents", [])
                self._bm25_metas = data.get("metadatas", [])
                if self._bm25 is None and self._bm25_docs:
                    corpus = [doc.split() for doc in self._bm25_docs]
                    self._bm25 = BM25Okapi(corpus)
            except FileNotFoundError:
                self._bm25 = None

    def _bm25_search(
        self, query: str, top_k: int, user_groups: Optional[List[str]] = None
    ) -> Tuple[List[str], List[Dict]]:
        """Executa busca BM25 usando rank_bm25.

        Caso ``user_groups`` seja fornecido, apenas documentos cujo metadado
        ``allowed_groups`` intersecte com esses grupos serão retornados."""

        self._load_bm25_index()
        if not self._bm25:
            return [], []
        tokens = query.split()
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        docs: List[str] = []
        metas: List[Dict] = []
        for idx, _ in ranked:
            meta = self._bm25_metas[idx]
            allowed = meta.get("allowed_groups", [])
            if user_groups is None or set(allowed).intersection(user_groups):
                docs.append(self._bm25_docs[idx])
                metas.append(meta)
            if len(docs) >= top_k:
                break
        return docs, metas

    def _rrf_fusion(
        self,
        docs_lists: List[List[str]],
        metas_lists: List[List[Dict]],
        top_k: int,
        k: int = 60,
    ) -> Tuple[List[str], List[Dict]]:
        """Combina listas ranqueadas usando Reciprocal Rank Fusion."""
        scores = defaultdict(float)
        meta_map: Dict[str, Dict] = {}
        for docs, metas in zip(docs_lists, metas_lists):
            for rank, (doc, meta) in enumerate(zip(docs, metas), start=1):
                scores[doc] += 1 / (k + rank)
                if doc not in meta_map:
                    meta_map[doc] = meta
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        docs = [doc for doc, _ in ranked[:top_k]]
        metas = [meta_map[doc] for doc in docs]
        return docs, metas

    def _chroma_retriever(
        self,
        query: str,
        collection_name: str = "documents",
        top_k: int = 3,
        user_groups: Optional[List[str]] = None,
    ) -> Iterable[Dict[str, str]]:
        """Consulta o ChromaDB e BM25 para recuperar documentos relevantes.

        Quando ``user_groups`` é fornecido, a consulta ao ChromaDB aplica um
        filtro para retornar apenas documentos cujo campo ``allowed_groups``
        contenha algum desses grupos."""
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
        where = {"allowed_groups": {"$in": user_groups}} if user_groups else {}
        results = collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas"],
        )
        chroma_docs = results.get("documents", [[]])[0]
        chroma_metas = results.get("metadatas", [[]])[0]

        bm25_docs, bm25_metas = self._bm25_search(query, top_k, user_groups)

        docs, metadatas = self._rrf_fusion(
            [chroma_docs, bm25_docs],
            [chroma_metas, bm25_metas],
            top_k,
        )

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

    def _openai_question_rewriter(
        self, history: List[Dict[str, str]], question: str
    ) -> Tuple[str, int]:
        """Condensa o histórico e a nova pergunta em uma pergunta autônoma."""
        from openai import OpenAI

        client = OpenAI()
        conversa = ""
        for turno in history:
            q = turno.get("pergunta") or turno.get("question") or ""
            a = turno.get("resposta") or turno.get("answer") or ""
            conversa += f"Usuário: {q}\nAssistente: {a}\n"

        prompt = (
            "Reescreva a última pergunta do usuário para que ela seja independente "
            "e contenha todo o contexto necessário dado o histórico."\
            f"\n\n{conversa}\nUsuário: {question}\nPergunta reescrita:"
        )

        completion = client.chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "system",
                    "content": "Você reformula perguntas considerando o contexto da conversa.",
                },
                {"role": "user", "content": prompt},
            ],
        )

        pergunta_reescrita = completion.choices[0].message.content.strip()
        tokens = getattr(getattr(completion, "usage", None), "total_tokens", 0)
        return pergunta_reescrita, tokens

    def _select_model(self, query: str) -> str:
        """Seleciona dinamicamente o modelo de chat conforme a complexidade."""

        fast_model = os.getenv("OPENAI_FAST_MODEL", "gpt-3.5-turbo")
        full_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        threshold = int(os.getenv("ROUTER_LENGTH_THRESHOLD", "12"))

        keywords = {
            "resuma",
            "resumo",
            "explique",
            "analise",
            "análise",
            "compare",
            "detalhe",
            "summarize",
            "explain",
            "analyze",
        }

        if len(query.split()) > threshold or any(k in query.lower() for k in keywords):
            return full_model
        return fast_model

    def _gemini_generator(self, query: str, context: str) -> Tuple[str, int]:
        """Gera uma resposta usando a API do Gemini."""
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

        model = genai.GenerativeModel("gemini-1.5-flash-latest")

        prompt = (
            "Responda à pergunta do usuário utilizando apenas o contexto "
            "fornecido. Se a resposta não estiver presente no contexto, "
            f"informe que não sabe.\n\nContexto:\n{context}\n\nPergunta: {query}"
        )

        response = model.generate_content(prompt)
        resposta = response.text.strip()

        tokens = 0

        return resposta, tokens

    def _openai_generator(self, query: str, context: str) -> Tuple[str, int]:
        """Gera uma resposta usando um LLM real baseado no contexto.

        Utiliza um *router* simples para escolher entre modelos mais caros ou
        econômicos, retornando também a quantidade de tokens utilizada."""

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

        model = self._select_model(query)

        completion = client.chat.completions.create(
            model=model,
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

    def answer(
        self,
        query: str,
        historico: Optional[List[Dict[str, str]]] = None,
        user_groups: Optional[List[str]] = None,
    ) -> Tuple[str, List[Dict[str, str]], int]:
        """Executa o pipeline de pergunta e resposta retornando também as fontes e tokens.

        Os documentos recuperados respeitam as permissões indicadas em
        ``user_groups``."""
        total_tokens = 0
        search_query = query
        if historico:
            search_query, tokens = self.question_rewriter(historico, query)
            total_tokens += tokens

        relevant_docs = list(self.retriever(search_query, user_groups=user_groups))
        summaries = []
        for doc in relevant_docs:
            resumo, tokens = self.summarizer(query, doc["texto"])
            summaries.append(resumo)
            total_tokens += tokens
        context = "\n\n".join(summaries)
        resposta, tokens = self.generator(query, context)
        total_tokens += tokens
        return resposta, relevant_docs, total_tokens

    def _presentation_generator(
        self, topic: str, num_slides: int
    ) -> Tuple[List[str], int]:
        """Gera tópicos de apresentação para um dado assunto."""
        total_tokens = 0
        pontos_chave: List[str] = []
        for _ in range(num_slides):
            resp, _, tokens = self.answer(f"Apresente um ponto chave sobre {topic}.")
            pontos_chave.append(resp)
            total_tokens += tokens

        from openai import OpenAI

        client = OpenAI()
        prompt = (
            f"Organize os seguintes pontos em {num_slides} slides sobre '{topic}'. "
            "Responda no formato:\nSlide 1:\n- ponto\nSlide 2:\n- ponto\n..."
        )
        prompt += "\n\nPontos:\n" + "\n".join(f"- {p}" for p in pontos_chave)
        completion = client.chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "system",
                    "content": "Você organiza informações em apresentações.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        texto = completion.choices[0].message.content.strip()
        tokens = getattr(getattr(completion, "usage", None), "total_tokens", 0)
        total_tokens += tokens
        slides = [s.strip() for s in texto.split("Slide") if s.strip()]
        return slides, total_tokens

    def generate_presentation(self, topic: str, num_slides: int) -> Tuple[List[str], int]:
        """Interface pública para geração de apresentações."""
        return self.presentation_generator(topic, num_slides)
