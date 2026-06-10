# -*- coding: utf-8 -*-

import os
import threading
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# 先加载 .env，再初始化 Pipeline，避免容器或服务器上的路径/模型配置失效。
load_dotenv()
from src.qa_pipeline import QAPipeline


SERVICE_VERSION = os.getenv("SERVICE_VERSION", "0.2.0")
MAX_INFLIGHT_REQUESTS = int(os.getenv("RAG_MAX_INFLIGHT_REQUESTS", "2"))
ENABLE_METRICS = os.getenv("ENABLE_METRICS", "1") == "1"


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="用户问题")
    bm25_topk: int | None = Field(default=None, ge=1, le=50)
    milvus_topk: int | None = Field(default=None, ge=1, le=50)
    rerank_topk: int | None = Field(default=None, ge=1, le=20)
    return_context: bool = Field(default=False, description="是否返回精排后的上下文")


class ChatResponse(BaseModel):
    query: str
    answer: str
    cite_pages: list[int]
    related_images: list[dict[str, Any]]
    raw_response: str
    timings: dict[str, float]
    retrieval_config: dict[str, Any] | None = None
    context: list[str] | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # RAG 检索、rerank 和本地 LLM 都是重资源操作，用信号量做背压保护，
    # 避免压测或流量突增时把 GPU/显存打满后整体雪崩。
    app.state.chat_semaphore = threading.BoundedSemaphore(MAX_INFLIGHT_REQUESTS)
    app.state.pipeline = QAPipeline()
    yield
    app.state.pipeline = None


app = FastAPI(
    title="Tesla Manual RAG QA Service",
    description="基于 Model 3 用户手册的 RAG 问答服务",
    version=SERVICE_VERSION,
    lifespan=lifespan,
)

if ENABLE_METRICS:
    Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


@app.get("/health")
def health():
    return {"status": "ok", "version": SERVICE_VERSION}


@app.get("/ready")
def ready(request: Request):
    # health 只表示进程活着；ready 表示模型和索引已经加载完成，可以接真实流量。
    if getattr(request.app.state, "pipeline", None) is None:
        raise HTTPException(status_code=503, detail="pipeline is not ready")
    return {"status": "ready", "max_inflight_requests": MAX_INFLIGHT_REQUESTS}


@app.post("/chat", response_model=ChatResponse)
def chat(request: Request, payload: ChatRequest):
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must not be empty")

    semaphore: threading.BoundedSemaphore = request.app.state.chat_semaphore
    if not semaphore.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="service is busy, please retry later",
        )

    pipeline: QAPipeline = request.app.state.pipeline
    try:
        result = pipeline.answer(
            query,
            bm25_topk=payload.bm25_topk,
            milvus_topk=payload.milvus_topk,
            rerank_topk=payload.rerank_topk,
        )
        if not payload.return_context:
            result["context"] = None
        return result
    finally:
        semaphore.release()
