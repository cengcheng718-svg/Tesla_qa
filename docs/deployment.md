# 部署与压测说明

本文档覆盖三件面试中最容易被追问的工程化能力：压测、Docker 容器化、八卡分布式推理部署。

## 1. 服务分层

```text
Client / Locust
  -> FastAPI RAG API (:8080)
  -> BM25 + Milvus Lite + BGE-M3 embedding + BGE reranker
  -> vLLM OpenAI-compatible API (:8000)
  -> MongoDB (:27017)
```

关键改造点：

- `app.py` 增加 `/health`、`/ready`、`/metrics`，便于健康检查、就绪检查和 Prometheus 指标采集。
- `RAG_MAX_INFLIGHT_REQUESTS` 控制 API 同时进入 RAG/LLM 链路的请求数，压测时超额请求返回 `429`，这是对 GPU 服务的背压保护。
- `.env` 统一管理路径、模型、MongoDB、vLLM 和 topk 参数，Docker 和裸机部署只替换环境变量。

## 2. Docker 容器化部署

准备配置：

```bash
cp .env.example .env
```

单卡容器化启动：

```bash
docker compose up -d --build
curl http://127.0.0.1:8080/ready
```

请求验证：

```bash
curl -X POST http://127.0.0.1:8080/chat \
  -H 'Content-Type: application/json' \
  -d '{"query":"介绍一下离车后自动上锁功能"}'
```

镜像设计说明：

- `Dockerfile` 只打包代码和 Python 依赖，`data/`、`models/`、`LLaMA-Factory-main/output` 等大文件通过 volume 挂载。
- `docker-compose.yml` 编排 `mongo`、`vllm`、`rag-api`，并保留 `semantic-chunk` profile 给离线建库链路使用。
- API 容器使用非 root 用户运行，镜像内置 healthcheck。

## 3. 八卡分布式推理部署

八卡部署使用 vLLM 的 tensor parallel，将一个大模型切到 8 张 GPU 上做并行推理。

```bash
bash scripts/deploy_8gpu_vllm.sh
curl http://127.0.0.1:8080/ready
```

等价的显式命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
VLLM_TENSOR_PARALLEL_SIZE=8 \
RAG_MAX_INFLIGHT_REQUESTS=8 \
docker compose -f docker-compose.yml -f deploy/docker-compose.8gpu.yml up -d --build
```

八卡参数说明：

- `--tensor-parallel-size 8`：把模型权重按 tensor 维度切到 8 张卡。
- `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`：限定 vLLM 可见 GPU。
- `VLLM_GPU_MEMORY_UTILIZATION=0.86`：为 KV cache 预留更高显存比例，实际生产要结合 OOM 和 P99 延迟调。
- `RAG_MAX_INFLIGHT_REQUESTS=8`：API 层放大并发入口，但最终值以压测结果为准。

## 4. 压力测试

安装压测依赖：

```bash
python -m pip install -r load_tests/requirements.txt
```

带 Web UI 的压测：

```bash
locust -f load_tests/locustfile.py --host http://127.0.0.1:8080
```

无界面压测并导出结果：

```bash
mkdir -p reports
locust -f load_tests/locustfile.py \
  --host http://127.0.0.1:8080 \
  --headless -u 20 -r 2 -t 5m \
  --csv reports/tesla_qa
```

常用调参：

```bash
LOAD_TEST_BM25_TOPK=8 \
LOAD_TEST_MILVUS_TOPK=8 \
LOAD_TEST_RERANK_TOPK=4 \
LOAD_TEST_TIMEOUT_SECONDS=180 \
locust -f load_tests/locustfile.py --host http://127.0.0.1:8080 --headless -u 20 -r 2 -t 5m
```

如果只想压到服务背压上限、但不希望 `429` 计入 Locust 失败率，可追加：

```bash
LOAD_TEST_ACCEPT_429=1 locust -f load_tests/locustfile.py --host http://127.0.0.1:8080 --headless -u 50 -r 5 -t 5m
```

压测观察指标：

- `RPS`：服务吞吐。
- `P95/P99 latency`：面试中优先讲尾延迟，因为 RAG + LLM 的用户体验通常被慢请求决定。
- `429 ratio`：说明背压触发比例，比例过高要扩容或降低单请求成本。
- `/metrics`：可接 Prometheus，观察 FastAPI 请求耗时和状态码分布。
- GPU 指标：配合 `nvidia-smi dmon` 观察显存、利用率和 OOM。

## 5. 面试复述口径

可以这样讲：

> 我把原来的脚本式 RAG 服务改成了可部署的在线服务。FastAPI 层负责参数校验、健康检查、Prometheus 指标和并发背压；检索层保留 BM25 + Milvus/BGE-M3 的混合召回，再用 BGE reranker 精排；生成层通过 vLLM 暴露 OpenAI 兼容接口，实现业务服务和模型服务解耦。

> 压测用 Locust，流量来自已有 QA 测试集，不是随便构造的 hello world。压测关注 RPS、P95/P99、错误率和 429 背压比例，然后根据结果调整 topk、API 并发数和 vLLM 显存利用率。

> 容器化上，我把 API、MongoDB、vLLM 拆成独立服务，模型和索引不打进镜像，而是通过 volume 挂载，这样镜像更小、发布更稳定。八卡部署用 vLLM tensor parallel，`tensor_parallel_size=8`，API 只需要把 `LOCAL_LLM_BASE_URL` 指向 vLLM 服务，不需要改业务代码。
