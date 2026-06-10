# -*- coding: utf-8 -*-

import os
import time
from dataclasses import asdict, dataclass
from typing import Any

from dotenv import load_dotenv


# CLI、API 和离线脚本都会走 Pipeline，这里统一加载 .env，减少环境差异。
load_dotenv()


@dataclass
class RetrievalConfig:
    bm25_topk: int = int(os.getenv("RAG_BM25_TOPK", "10"))
    milvus_topk: int = int(os.getenv("RAG_MILVUS_TOPK", "10"))
    rerank_topk: int = int(os.getenv("RAG_RERANK_TOPK", "5"))
    warmup_query: str = os.getenv("RAG_WARMUP_QUERY", "这是一条测试数据")
    warmup_topk: int = int(os.getenv("RAG_WARMUP_TOPK", "3"))


class QAPipeline:
    """Reusable RAG pipeline shared by CLI and API entrypoints."""

    def __init__(self, config: RetrievalConfig | None = None):
        self.config = config or RetrievalConfig()

        # Delay imports because these modules connect to MongoDB and load GPU models.
        from src.constant import bge_reranker_tuned_model_path
        from src.retriever.bm25_retriever import BM25
        from src.retriever.milvus_retriever import MilvusRetriever
        from src.reranker.bge_m3_reranker import BGEM3ReRanker
        from src.utils import merge_docs, post_processing

        self._merge_docs = merge_docs
        self._post_processing = post_processing
        self.bm25_retriever = BM25(docs=None, retrieve=True)
        self.milvus_retriever = MilvusRetriever(docs=None, retrieve=True)
        self.reranker = BGEM3ReRanker(model_path=bge_reranker_tuned_model_path)

        if self.config.warmup_query:
            self.milvus_retriever.retrieve_topk(
                self.config.warmup_query, topk=self.config.warmup_topk
            )

    def retrieve(
        self,
        query: str,
        bm25_topk: int | None = None,
        milvus_topk: int | None = None,
        rerank_topk: int | None = None,
    ) -> dict[str, Any]:
        bm25_topk = bm25_topk or self.config.bm25_topk
        milvus_topk = milvus_topk or self.config.milvus_topk
        rerank_topk = rerank_topk or self.config.rerank_topk

        t1 = time.time()
        bm25_docs = self.bm25_retriever.retrieve_topk(query, topk=bm25_topk)
        t2 = time.time()
        milvus_docs = self.milvus_retriever.retrieve_topk(query, topk=milvus_topk)
        t3 = time.time()
        merged_docs = self._merge_docs(bm25_docs, milvus_docs)
        t4 = time.time()
        ranked_docs = self.reranker.rank(query, merged_docs, topk=rerank_topk)
        t5 = time.time()

        return {
            "bm25_docs": bm25_docs,
            "milvus_docs": milvus_docs,
            "merged_docs": merged_docs,
            "ranked_docs": ranked_docs,
            "timings": {
                "bm25": round(t2 - t1, 4),
                "milvus": round(t3 - t2, 4),
                "merge": round(t4 - t3, 4),
                "rerank": round(t5 - t4, 4),
                "total_retrieval": round(t5 - t1, 4),
            },
        }

    def build_context(self, docs) -> str:
        return "\n".join(
            ["【" + str(idx + 1) + "】" + doc.page_content for idx, doc in enumerate(docs)]
        )

    def post_process(self, response: str, docs):
        return self._post_processing(response, docs)

    def answer(
        self,
        query: str,
        bm25_topk: int | None = None,
        milvus_topk: int | None = None,
        rerank_topk: int | None = None,
    ) -> dict[str, Any]:
        from src.client.llm_local_client import request_chat

        retrieval = self.retrieve(
            query,
            bm25_topk=bm25_topk,
            milvus_topk=milvus_topk,
            rerank_topk=rerank_topk,
        )
        ranked_docs = retrieval["ranked_docs"]
        context = self.build_context(ranked_docs)
        response = request_chat(query, context, stream=False)
        answer = self.post_process(response, ranked_docs)

        return {
            "query": query,
            "answer": answer["answer"],
            "cite_pages": answer["cite_pages"],
            "related_images": answer["related_images"],
            "raw_response": response,
            "context": [doc.page_content for doc in ranked_docs],
            "timings": retrieval["timings"],
            "retrieval_config": asdict(self.config),
        }
