# 车书问答系统

基于 Tesla Model 3 用户手册构建的 RAG 问答系统。项目覆盖 PDF 解析、文档清洗、语义切分、混合召回、重排序、本地模型生成和离线评估，适合作为 LLM 应用工程项目展示。

## 架构

```text
Tesla_Manual.pdf
  -> PDF 解析 / 图片抽取
  -> LLM 清洗 / 语义切分
  -> MongoDB 存储父子文档块与图片元信息
  -> BM25 + Milvus(BGE-M3 dense/sparse) 混合召回
  -> BGE reranker 精排
  -> vLLM OpenAI-compatible API 生成答案
  -> 引用页码和相关图片后处理
```

## 核心入口

- `build_index.py`：解析 PDF、清洗文档、构建 BM25 和 Milvus 索引。
- `infer.py`：命令行问答入口。
- `app.py`：FastAPI 问答服务，提供 `/chat` 和 `/health`。
- `src/qa_pipeline.py`：CLI 和 API 共享的 RAG pipeline。
- `src/gen_qa/run.py`：生成 QA 数据、扩写问题、划分训练/测试集。
- `final_score.py`：语义相似度、关键词命中和 RAGAS 评估。

## 运行依赖

运行完整链路前需要准备：

- Python 环境和 `requirements.txt` 依赖。
- MongoDB，用于存储文档块、父子块关系和图片元信息。
- Milvus Lite 索引文件，或通过 `build_index.py` 重新构建。
- BGE-M3 embedding 模型和 reranker 模型。
- vLLM 服务，用于加载本地 Qwen 微调模型并提供 OpenAI-compatible API。

## 配置

复制环境变量模板并按机器路径修改：

```bash
cp .env.example .env
```

关键配置：

- `RAG_BASE_DIR`：项目根目录。原始服务器路径是 `/root/autodl-tmp/RAG`，本地运行时需要改成当前项目路径。
- `LOCAL_LLM_BASE_URL`：本地 vLLM 服务地址，默认 `http://localhost:8000/v1`。
- `LOCAL_LLM_MODEL`：vLLM 加载的模型路径，默认 `LLaMA-Factory-main/output/qwen3_lora_sft_int4`。
- `MONGO_HOST` / `MONGO_PORT` / `MONGO_DB_NAME`：MongoDB 连接配置。
- `DOUBAO_API_KEY` / `DOUBAO_BASE_URL` / `DOUBAO_MODEL_NAME`：用于云端 LLM 清洗、数据生成或替代本地生成。

## 启动服务

启动语义切分服务和 vLLM：

```bash
bash scripts/start_services.sh
```

如果需要脚本同时启动 MongoDB，将 `.env` 中的 `START_MONGODB` 改为 `1`，并确认 `MONGODB_BIN`、`MONGODB_DBPATH`、`MONGODB_LOGPATH` 路径正确。

也可以手动启动：

```bash
python src/server/semantic_chunk.py
vllm serve LLaMA-Factory-main/output/qwen3_lora_sft_int4 --max-model-len 8192 --gpu-memory-utilization 0.7
```

## 构建索引

```bash
python build_index.py
```

该步骤会读取手册 PDF，生成清洗文档、切分文档、BM25 索引、Milvus 索引，并把文档元信息写入 MongoDB。

## 命令行问答

```bash
python infer.py
```

## API 服务

```bash
uvicorn app:app --host 0.0.0.0 --port 8080
```

请求示例：

```bash
curl -X POST http://localhost:8080/chat \
  -H 'Content-Type: application/json' \
  -d '{"query":"介绍一下离车后自动上锁功能", "return_context": true}'
```

服务接口：

- `/health`：进程健康检查。
- `/ready`：模型、索引和 Pipeline 就绪检查。
- `/metrics`：Prometheus 指标，默认通过 `ENABLE_METRICS=1` 开启。

## Docker 部署

```bash
cp .env.example .env
docker compose up -d --build
curl http://127.0.0.1:8080/ready
```

八卡 vLLM 分布式推理：

```bash
bash scripts/deploy_8gpu_vllm.sh
```

详细说明见 `docs/deployment.md`。

面试复述和中英文问答见 `docs/interview_qa_bilingual.md`。

## 压力测试

```bash
python -m pip install -r load_tests/requirements.txt
locust -f load_tests/locustfile.py --host http://127.0.0.1:8080
```

无界面压测示例：

```bash
mkdir -p reports
locust -f load_tests/locustfile.py --host http://127.0.0.1:8080 --headless -u 20 -r 2 -t 5m --csv reports/tesla_qa
```

## 项目亮点

- 混合召回：BM25 负责关键词匹配，BGE-M3 dense/sparse 召回负责语义匹配。
- 父子文档块：检索子块命中后回溯父块，降低答案上下文碎片化。
- 重排序：使用 BGE reranker 提升最终上下文相关性。
- 本地模型服务：通过 vLLM 暴露 OpenAI-compatible API，推理端与模型部署解耦。
- 生产化 API：提供健康检查、就绪检查、Prometheus 指标和并发背压。
- 工程化交付：提供 Docker Compose、八卡 vLLM tensor parallel 部署和 Locust 压测脚本。
- 数据闭环：包含 QA 生成、训练数据构造、reranker 数据构造和评估脚本。

## 已知注意事项

- 首次运行前需要确认 `RAG_BASE_DIR`、模型路径和 MongoDB 路径。
- `app.py` 启动时会加载检索器和 reranker，GPU 模型较大时启动会比较慢。
- 如果不想部署本地 vLLM，可以把生成端替换为 `src/client/llm_chat_client.py` 中的云端 LLM 调用。
