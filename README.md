# Tesla Owner's Manual QA System

A RAG-based question answering system built on the Tesla Model 3 owner's manual. The project covers PDF parsing, document cleaning, semantic chunking, hybrid retrieval, reranking, local LLM generation, and offline evaluation. It is designed as an end-to-end LLM application engineering project.

## Architecture

```text
Tesla_Manual.pdf
  -> PDF parsing / image extraction
  -> LLM-based cleaning / semantic chunking
  -> MongoDB storage for parent-child chunks and image metadata
  -> BM25 + Milvus (BGE-M3 dense/sparse) hybrid retrieval
  -> BGE reranker for final ranking
  -> vLLM OpenAI-compatible API for answer generation
  -> source citation, page number, and related image post-processing
```

## Main Entry Points

- `build_index.py`: Parses the PDF, cleans documents, and builds BM25 and Milvus indexes.
- `infer.py`: Command-line QA entry point.
- `app.py`: FastAPI service exposing `/chat` and `/health`.
- `src/qa_pipeline.py`: Shared RAG pipeline used by both CLI and API entry points.
- `src/gen_qa/run.py`: Generates QA data, expands questions, and creates train/test splits.
- `final_score.py`: Evaluates semantic similarity, keyword coverage, and RAGAS metrics.

## Runtime Requirements

Before running the full pipeline, prepare:

- A Python environment with dependencies from `requirements.txt`.
- MongoDB for storing document chunks, parent-child relationships, and image metadata.
- A Milvus Lite index, or rebuild it with `build_index.py`.
- BGE-M3 embedding and reranker models.
- A vLLM service that loads the local fine-tuned Qwen model and exposes an OpenAI-compatible API.

## Configuration

Copy the environment template and update paths for your machine:

```bash
cp .env.example .env
```

Important variables:

- `RAG_BASE_DIR`: Project root directory. The original server path was `/root/autodl-tmp/RAG`; update it for local execution.
- `LOCAL_LLM_BASE_URL`: Local vLLM endpoint, defaulting to `http://localhost:8000/v1`.
- `LOCAL_LLM_MODEL`: Model path loaded by vLLM, defaulting to `LLaMA-Factory-main/output/qwen3_lora_sft_int4`.
- `MONGO_HOST` / `MONGO_PORT` / `MONGO_DB_NAME`: MongoDB connection settings.
- `DOUBAO_API_KEY` / `DOUBAO_BASE_URL` / `DOUBAO_MODEL_NAME`: Cloud LLM settings for document cleaning, data generation, or a generation fallback.

## Start Services

Start the semantic chunking service and vLLM:

```bash
bash scripts/start_services.sh
```

To let the script start MongoDB as well, set `START_MONGODB=1` in `.env` and verify `MONGODB_BIN`, `MONGODB_DBPATH`, and `MONGODB_LOGPATH`.

You can also start services manually:

```bash
python src/server/semantic_chunk.py
vllm serve LLaMA-Factory-main/output/qwen3_lora_sft_int4 --max-model-len 8192 --gpu-memory-utilization 0.7
```

## Build Indexes

```bash
python build_index.py
```

This step reads the owner's manual PDF, generates cleaned documents and split documents, builds BM25 and Milvus indexes, and writes document metadata to MongoDB.

## Command-Line QA

```bash
python infer.py
```

## API Service

```bash
uvicorn app:app --host 0.0.0.0 --port 8080
```

Example request:

```bash
curl -X POST http://localhost:8080/chat \
  -H 'Content-Type: application/json' \
  -d '{"query":"How does Walk-Away Door Lock work?", "return_context": true}'
```

Available endpoints:

- `/health`: Process health check.
- `/ready`: Readiness check for models, indexes, and the RAG pipeline.
- `/metrics`: Prometheus metrics, enabled by default with `ENABLE_METRICS=1`.

## Docker Deployment

```bash
cp .env.example .env
docker compose up -d --build
curl http://127.0.0.1:8080/ready
```

For 8-GPU distributed vLLM inference:

```bash
bash scripts/deploy_8gpu_vllm.sh
```

See `docs/deployment.md` for deployment details.

Interview preparation notes and bilingual Q&A are available in `docs/interview_qa_bilingual.md`.

## Load Testing

```bash
python -m pip install -r load_tests/requirements.txt
locust -f load_tests/locustfile.py --host http://127.0.0.1:8080
```

Headless load test example:

```bash
mkdir -p reports
locust -f load_tests/locustfile.py --host http://127.0.0.1:8080 --headless -u 20 -r 2 -t 5m --csv reports/tesla_qa
```

## Highlights

- Hybrid retrieval: BM25 handles keyword matching, while BGE-M3 dense/sparse retrieval handles semantic matching.
- Parent-child chunking: Retrieved child chunks are mapped back to parent chunks to reduce fragmented context.
- Reranking: A BGE reranker improves the relevance of final context passages.
- Local model serving: vLLM exposes an OpenAI-compatible API, decoupling inference code from model deployment.
- Production-oriented API: Health checks, readiness checks, Prometheus metrics, and concurrency backpressure are included.
- Engineering delivery: Docker Compose, 8-GPU vLLM tensor parallel deployment, and Locust load testing scripts are provided.
- Data feedback loop: The project includes QA generation, training data construction, reranker data construction, and evaluation scripts.

## Notes

- Before first run, verify `RAG_BASE_DIR`, model paths, and MongoDB paths.
- `app.py` loads retrievers and the reranker during startup, so startup can be slow when GPU models are large.
- If local vLLM is not available, the generation client can be replaced with the cloud LLM call in `src/client/llm_chat_client.py`.
