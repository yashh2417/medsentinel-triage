"""LangChain-based Clinical Hybrid RAG module using `EnsembleRetriever` and `BM25Retriever`."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnableSequence
from sentence_transformers import CrossEncoder

from medsentinel_triage.base import BaseInferenceModel, InputValidationError
from medsentinel_triage.preprocessing import validate_non_empty_text


def tokenize_clinical_text(text: str) -> List[str]:
    """Regex tokenizer matching training (`re.findall(r'[a-z0-9]+', text.lower())`)."""
    return re.findall(r"[a-z0-9]+", text.lower())


class ProductionClinicalHybridRAG(BaseInferenceModel[str, str, List[Dict[str, Any]]]):
    """Hybrid Clinical RAG using LangChain `EnsembleRetriever` (`Chroma` + `BM25Retriever`) and `CrossEncoder`."""

    model_name: str = "clinical_hybrid_rag"

    def __init__(
        self,
        corpus: Optional[List[Dict[str, str]]] = None,
        embed_model_name: str = "NeuML/pubmedbert-base-embeddings",
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        collection_name: str = "pubmedqa_clinical_guidelines",
        top_k: int = 3,
        persist_dir: Optional[Union[str, Path]] = None,
        force_reindex: bool = False,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.persist_dir = Path(persist_dir) if persist_dir is not None else None
        self.bm25_cache_path = (
            self.persist_dir / "rag_bm25_corpus.pkl" if self.persist_dir is not None else None
        )

        # 1. Initialize LangChain HuggingFaceEmbeddings (PubMedBERT) and CrossEncoder reranker
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embed_model_name,
            encode_kwargs={"normalize_embeddings": True},
        )
        self.reranker = CrossEncoder(reranker_model_name)

        # 2. Load from persistent Chroma + BM25 cache if available, otherwise index from provided corpus
        has_persisted_store = (
            not force_reindex
            and self.persist_dir is not None
            and self.persist_dir.exists()
            and self.bm25_cache_path is not None
            and self.bm25_cache_path.exists()
        )

        if has_persisted_store:
            assert self.persist_dir is not None and self.bm25_cache_path is not None
            cached_payload = joblib.load(self.bm25_cache_path)
            self.corpus: List[Dict[str, str]] = cached_payload["corpus"]
            self.bm25_retriever: BM25Retriever = cached_payload["bm25_retriever"]
            self.vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_dir),
                collection_metadata={"hnsw:space": "cosine"},
            )
            self.logger.info(
                "Loaded persistent Chroma vector store and BM25Retriever from %s (skipped re-embedding)",
                self.persist_dir,
            )
        else:
            if not corpus:
                raise InputValidationError(
                    "No existing persistent RAG vector store found and no `corpus` was provided. "
                    "Provide a list of document dicts (`doc_id`, `title`, `source`, `text`) to build the store."
                )
            self.corpus = corpus
            documents = [
                Document(
                    page_content=f"{doc['title']}. {doc['text']}",
                    metadata={
                        "doc_id": doc["doc_id"],
                        "title": doc["title"],
                        "source": doc["source"],
                        "text": doc["text"],
                    },
                )
                for doc in self.corpus
            ]

            if self.persist_dir is not None:
                self.persist_dir.mkdir(parents=True, exist_ok=True)

            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                ids=[doc["doc_id"] for doc in self.corpus],
                collection_name=collection_name,
                persist_directory=str(self.persist_dir) if self.persist_dir is not None else None,
                collection_metadata={"hnsw:space": "cosine"},
            )
            self.bm25_retriever = BM25Retriever.from_documents(
                documents,
                preprocess_func=tokenize_clinical_text,
            )
            self.bm25_retriever.k = min(10, len(documents))

            if self.bm25_cache_path is not None:
                joblib.dump(
                    {"corpus": self.corpus, "bm25_retriever": self.bm25_retriever},
                    self.bm25_cache_path,
                )

        # 3. Combine Dense Chroma Retriever + Sparse BM25Retriever via LangChain EnsembleRetriever (RRF)
        self.dense_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": min(10, len(self.corpus))}
        )
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.dense_retriever, self.bm25_retriever],
            weights=[0.5, 0.5],
            c=60,
        )
        self.rag_chain: RunnableSequence = self.ensemble_retriever | RunnableLambda(
            lambda docs: docs[:8]
        )

    def as_langchain_retriever(self) -> EnsembleRetriever:
        """Returns the underlying LangChain `EnsembleRetriever` instance."""
        return self.ensemble_retriever

    def hybrid_search_with_rerank(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieves RRF candidates via `EnsembleRetriever` and reranks them with `CrossEncoder`."""
        cleaned_query = self.preprocess(query)
        candidate_docs: List[Document] = self.rag_chain.invoke(cleaned_query)
        if not candidate_docs:
            return []

        pairs = [[cleaned_query, doc.page_content] for doc in candidate_docs]
        ce_scores = self.reranker.predict(pairs)
        ranked_order = np.argsort(-np.asarray(ce_scores))[:top_k]

        results: List[Dict[str, Any]] = []
        for idx in ranked_order:
            doc = candidate_docs[int(idx)]
            meta = doc.metadata
            raw_text = str(meta.get("text", doc.page_content))
            results.append(
                {
                    "doc_id": str(meta["doc_id"]),
                    "title": str(meta["title"]),
                    "source": str(meta["source"]),
                    "snippet": raw_text[:450] + "...",
                    "cross_encoder_score": round(float(ce_scores[int(idx)]), 4),
                }
            )
        return results

    def validate_input(self, raw_input: str) -> None:
        validate_non_empty_text(raw_input, field_name="clinical_query")

    def preprocess(self, raw_input: str) -> str:
        return validate_non_empty_text(raw_input, field_name="clinical_query")

    def predict(self, preprocessed_input: str) -> List[Dict[str, Any]]:
        return self.hybrid_search_with_rerank(preprocessed_input, top_k=self.top_k)
