# Tesla 车书问答系统工程化改造面试讲解 / Interview Guide

这份文档用于面试复述，重点解释本次改造如何实现：压力测试、Docker 容器化部署、八卡分布式部署、API 生产化增强。

This guide explains the production engineering improvements added to the Tesla manual RAG QA system: load testing, Docker deployment, 8-GPU distributed inference, and production-ready API enhancements.

---

## 一、中文讲解

### 1. 项目整体架构

这个项目是一个面向 Tesla Model 3 用户手册的 RAG 问答系统。整体链路是：

```text
用户请求
  -> FastAPI /chat
  -> BM25 关键词召回
  -> Milvus Lite + BGE-M3 向量召回
  -> merge_docs 合并召回结果
  -> BGE reranker 精排
  -> 拼接上下文
  -> vLLM OpenAI-compatible API 生成答案
  -> post_process 提取答案、引用页码、相关图片
```

本次改造不是改模型本身，而是把原来偏实验脚本的 RAG 系统，改造成更接近线上部署标准的服务。

核心文件：

- `app.py`：FastAPI 服务入口。
- `src/qa_pipeline.py`：RAG 主流程，供 API 和 CLI 复用。
- `Dockerfile`：API 镜像构建。
- `docker-compose.yml`：单卡或普通 GPU 部署编排。
- `deploy/docker-compose.8gpu.yml`：八卡 vLLM tensor parallel 部署。
- `load_tests/locustfile.py`：Locust 压测脚本。
- `docs/deployment.md`：部署和压测说明。

---

### 2. API 生产化是怎么做的？

原来的 API 只有 `/health` 和 `/chat`，能跑 demo，但面试中容易被追问线上服务可用性问题。所以我做了四个增强：

1. 增加 `.env` 加载  
   在 `app.py` 和 `src/qa_pipeline.py` 中使用 `python-dotenv`，让模型路径、MongoDB 地址、vLLM 地址、topk 参数都可以通过环境变量控制。

2. 增加 `/ready`  
   `/health` 只表示进程还活着，`/ready` 表示 Pipeline、索引、模型已经加载完成，可以接真实请求。生产环境通常会用 ready probe 控制服务是否进入流量池。

3. 增加 `/metrics`  
   使用 `prometheus-fastapi-instrumentator` 暴露 Prometheus 指标。这样压测时可以观察请求耗时、状态码分布和错误率。

4. 增加并发背压  
   用 `threading.BoundedSemaphore` 控制进入 RAG + LLM 链路的最大并发数，配置项是 `RAG_MAX_INFLIGHT_REQUESTS`。当 GPU 资源忙时，服务返回 `429`，而不是让所有请求堆积到显存 OOM。

面试可以这样说：

> 我把 API 从 demo 形态改成生产服务形态，主要补了配置外置化、健康检查、就绪检查、指标暴露和并发背压。这样服务不仅能跑，还能被部署系统调度、被监控系统观测，并且在高并发下能优雅降级。

---

### 3. 为什么要做并发背压？

RAG 服务的瓶颈通常不在 FastAPI 本身，而在 embedding、rerank 和 LLM 推理。尤其是 GPU 推理，请求无限进入会造成：

- GPU 显存被 KV cache 或模型中间结果撑爆。
- 请求排队时间过长，P95/P99 延迟急剧升高。
- 单个慢请求拖垮整体服务。

所以我在 `/chat` 入口处加了信号量：

```text
请求进入 /chat
  -> 尝试获取 semaphore
  -> 获取成功：进入 RAG pipeline
  -> 获取失败：返回 429 service is busy
```

这就是背压机制。它的作用是保护系统，让服务在压力过大时可控失败，而不是整体崩溃。

面试可以这样回答：

> 我没有简单地把并发开大，而是在 API 层做了限流和背压。因为 RAG 的核心瓶颈是 GPU 推理和 rerank，不加控制会导致显存 OOM 或尾延迟失控。返回 429 是一种更可控的降级方式，客户端可以重试，服务本身不会雪崩。

---

### 4. 压力测试是怎么实现的？

压测使用 Locust，文件是 `load_tests/locustfile.py`。

实现重点：

- 优先读取 `data/qa_pairs/test_qa_pair.json`，使用项目已有真实问题作为压测流量。
- 按 9:1 的比例访问 `/chat` 和 `/health`，模拟主要业务流量加少量健康检查。
- 支持通过环境变量控制 `bm25_topk`、`milvus_topk`、`rerank_topk`。
- 对返回结果做基础校验，比如是否有 `answer` 和 `timings` 字段。
- 默认把 `429` 作为失败，这样能真实暴露容量不足；也可以用 `LOAD_TEST_ACCEPT_429=1` 单独观察背压触发比例。

常用命令：

```bash
locust -f load_tests/locustfile.py \
  --host http://127.0.0.1:8080 \
  --headless -u 20 -r 2 -t 5m \
  --csv reports/tesla_qa
```

压测主要关注：

- RPS：整体吞吐。
- P95/P99 latency：用户体验主要由尾延迟决定。
- Failure rate：失败率。
- 429 ratio：背压触发比例。
- GPU utilization / memory：用 `nvidia-smi` 观察 GPU 是否吃满。

面试可以这样说：

> 我没有用固定 hello world 压测，而是复用了项目已有 QA 测试集，让压测流量更接近真实业务。压测关注的不只是平均延迟，还包括 P95/P99、错误率、429 背压比例和 GPU 利用率，然后根据结果调整 topk、API 并发数和 vLLM 参数。

---

### 5. Docker 容器化是怎么实现的？

我新增了 `Dockerfile` 和 `docker-compose.yml`。

Dockerfile 的设计：

- 使用 `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04` 作为基础镜像，适配 GPU 推理。
- 安装 Python 依赖。
- 设置 `PYTHONPATH=/app` 和 `RAG_BASE_DIR=/app`。
- 使用非 root 用户运行 API，降低安全风险。
- 增加 `HEALTHCHECK`。

Compose 的设计：

```text
mongo       -> 文档元信息和图片元信息存储
vllm        -> 本地大模型推理服务，提供 OpenAI-compatible API
rag-api     -> FastAPI RAG 服务
semantic-chunk -> 离线语义切分服务，放在 tools profile
```

为什么模型和数据不用打进镜像？

因为模型、索引、PDF、MongoDB 数据体积都很大，如果打进镜像：

- 镜像会非常大，构建和分发都慢。
- 模型更新会导致镜像频繁重建。
- 生产环境难以复用共享存储。

所以我采用 volume 挂载：

```text
./data              -> /app/data
./models            -> /app/models
./RAG-Retrieval     -> /app/RAG-Retrieval
./LLaMA-Factory-main/output/qwen3_lora_sft_int4 -> /models/qwen3_lora_sft_int4
```

面试可以这样回答：

> 容器化时我把代码和大文件解耦。镜像只包含运行代码和依赖，模型、索引、数据通过 volume 挂载。这样镜像更轻，发布更稳定，模型版本切换也更方便。服务拆成 API、vLLM、MongoDB 三个容器，符合线上微服务拆分思路。

---

### 6. 八卡分布式部署是怎么实现的？

八卡部署文件是 `deploy/docker-compose.8gpu.yml`，启动脚本是 `scripts/deploy_8gpu_vllm.sh`。

核心方式是 vLLM tensor parallel：

```bash
--tensor-parallel-size 8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

含义：

- `CUDA_VISIBLE_DEVICES` 控制 vLLM 可以看到哪几张 GPU。
- `--tensor-parallel-size 8` 表示把模型权重切到 8 张 GPU 上做张量并行。
- `VLLM_GPU_MEMORY_UTILIZATION=0.86` 表示 vLLM 可以使用每张 GPU 约 86% 的显存，剩余显存给系统和波动空间。
- API 层不需要知道模型是单卡还是八卡，只需要调用同一个 OpenAI-compatible 地址。

面试可以这样说：

> 八卡部署我没有改业务代码，而是在模型服务层使用 vLLM tensor parallel。API 仍然调用 `http://vllm:8000/v1`，vLLM 内部负责把模型切到 8 张卡上并行推理。这种方式实现了业务服务和模型部署解耦，单卡、八卡切换只需要改部署参数。

---

### 7. 可能的面试问题与中文回答

#### Q1：你这个项目的核心改造是什么？

A：核心是把一个能本地跑通的 RAG demo 改造成更接近生产环境的服务。具体包括 FastAPI 生产化、Prometheus 指标、并发背压、Locust 压测、Docker Compose 编排，以及八卡 vLLM tensor parallel 推理部署。

#### Q2：为什么要把 API 和 vLLM 拆开？

A：因为 API 负责业务逻辑和 RAG 编排，vLLM 负责大模型推理，两者资源特征不同。拆开后可以单独扩容、单独升级模型，也方便 API 保持轻量。API 只调用 OpenAI-compatible 接口，不依赖具体模型部署方式。

#### Q3：`/health` 和 `/ready` 有什么区别？

A：`/health` 表示进程还活着，通常给容器健康检查使用；`/ready` 表示 Pipeline、索引、模型加载完成，服务可以接流量。模型加载慢时，进程可能已经启动，但还不能处理请求，所以需要 ready check。

#### Q4：为什么要返回 429？

A：429 表示服务当前繁忙，是一种可控的背压机制。相比让请求无限排队导致 OOM 或超时，429 能保护服务稳定性，客户端也可以做退避重试。

#### Q5：压测时你看哪些指标？

A：主要看 RPS、P95/P99 延迟、错误率、429 比例、GPU 显存和利用率。平均延迟不够，因为 RAG + LLM 场景下用户体验往往由尾延迟决定。

#### Q6：如果压测发现 P99 很高，你怎么优化？

A：我会先定位瓶颈。如果是检索慢，可以降低 topk、优化索引或缓存热点问题；如果是 rerank 慢，可以减少候选数量或换更轻的 reranker；如果是 LLM 慢，可以调 vLLM batch、max tokens、GPU memory utilization，或者增加 GPU 并行度。

#### Q7：为什么 Docker 镜像里不放模型？

A：模型文件太大，放进镜像会导致构建和发布非常慢。用 volume 挂载可以让镜像只关注代码和依赖，模型版本可以独立切换，也更符合生产环境共享存储的做法。

#### Q8：八卡部署一定比单卡快吗？

A：不一定。八卡 tensor parallel 主要解决大模型显存和吞吐问题，但也会引入 GPU 间通信开销。对于小模型或低并发，单卡可能更简单；八卡更适合大模型、长上下文、高并发场景。最终要通过压测看 RPS 和 P95/P99。

#### Q9：怎么保证压测是真实的？

A：压测脚本读取项目已有 QA 测试集，而不是构造固定问题。这样请求长度、问题类型、检索路径更贴近真实业务。

#### Q10：这个系统线上还有哪些可以继续优化？

A：可以继续加入 Redis 缓存热点问题、引入异步队列、做请求 tracing、加入结构化日志、用 Kubernetes HPA 做弹性扩容、把 Milvus Lite 替换成独立 Milvus 集群，并加入灰度发布和模型版本管理。

---

## 二、English Explanation

### 1. Overall Architecture

This project is a RAG-based QA system for the Tesla Model 3 user manual. The request flow is:

```text
User request
  -> FastAPI /chat
  -> BM25 keyword retrieval
  -> Milvus Lite + BGE-M3 vector retrieval
  -> merge_docs
  -> BGE reranker
  -> context construction
  -> vLLM OpenAI-compatible API
  -> post_process for answer, citations, and images
```

The main goal of this improvement was not to change the model itself, but to upgrade the project from a local demo into a more production-ready service.

Key files:

- `app.py`: FastAPI service entrypoint.
- `src/qa_pipeline.py`: reusable RAG pipeline.
- `Dockerfile`: API image build file.
- `docker-compose.yml`: container orchestration for API, vLLM, and MongoDB.
- `deploy/docker-compose.8gpu.yml`: 8-GPU vLLM tensor parallel deployment.
- `load_tests/locustfile.py`: Locust load testing script.
- `docs/deployment.md`: deployment and load testing documentation.

---

### 2. How Did You Make the API Production-Ready?

I added four production-oriented capabilities:

1. Environment-based configuration  
   The API loads `.env` through `python-dotenv`. Model paths, MongoDB address, vLLM endpoint, and retrieval topk values can all be controlled through environment variables.

2. `/ready` endpoint  
   `/health` only means the process is alive. `/ready` means the RAG pipeline, indexes, and models are loaded and ready to serve real traffic.

3. `/metrics` endpoint  
   I used `prometheus-fastapi-instrumentator` to expose Prometheus metrics, so we can monitor latency, status codes, and error rates during load testing.

4. Backpressure control  
   I added a `threading.BoundedSemaphore` around the `/chat` path. The maximum number of concurrent RAG requests is controlled by `RAG_MAX_INFLIGHT_REQUESTS`. If the service is overloaded, it returns `429` instead of letting requests pile up and crash the GPU.

Interview answer:

> I upgraded the API from a demo service into a production-oriented service by adding externalized configuration, health checks, readiness checks, Prometheus metrics, and backpressure control. This makes the service deployable, observable, and more stable under high traffic.

---

### 3. Why Do You Need Backpressure?

For a RAG system, the bottleneck is usually not FastAPI itself. The expensive parts are embedding, reranking, and LLM inference. Without concurrency control:

- GPU memory may be exhausted.
- Request queues may grow too long.
- P95 and P99 latency may become unacceptable.
- One overloaded model service can slow down the whole system.

The backpressure flow is:

```text
Request enters /chat
  -> try to acquire semaphore
  -> success: execute RAG pipeline
  -> failure: return 429 service is busy
```

Interview answer:

> I added backpressure because GPU inference is a limited resource. Instead of allowing unlimited requests to queue up and potentially cause OOM, the API returns 429 when the system is busy. This gives clients a clear retry signal and protects overall service stability.

---

### 4. How Is Load Testing Implemented?

Load testing is implemented with Locust in `load_tests/locustfile.py`.

Key points:

- It reads real questions from `data/qa_pairs/test_qa_pair.json`.
- It sends mostly `/chat` requests and a smaller amount of `/health` requests.
- It supports environment variables for `bm25_topk`, `milvus_topk`, and `rerank_topk`.
- It validates whether the response contains `answer` and `timings`.
- By default, `429` is counted as a failure to reveal capacity limits. It can also be accepted with `LOAD_TEST_ACCEPT_429=1` when analyzing backpressure behavior.

Command:

```bash
locust -f load_tests/locustfile.py \
  --host http://127.0.0.1:8080 \
  --headless -u 20 -r 2 -t 5m \
  --csv reports/tesla_qa
```

Main metrics:

- RPS.
- P95/P99 latency.
- Failure rate.
- 429 ratio.
- GPU utilization and memory usage.

Interview answer:

> I used Locust for load testing and reused the existing QA test set instead of artificial hello-world requests. I mainly monitor RPS, P95/P99 latency, failure rate, 429 ratio, and GPU utilization, then tune retrieval topk, API concurrency, and vLLM parameters based on the results.

---

### 5. How Is Docker Deployment Implemented?

I added `Dockerfile` and `docker-compose.yml`.

Dockerfile design:

- Uses `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04` as the base image.
- Installs Python dependencies.
- Sets `PYTHONPATH=/app` and `RAG_BASE_DIR=/app`.
- Runs the API as a non-root user.
- Adds a container `HEALTHCHECK`.

Compose services:

```text
mongo       -> metadata storage
vllm        -> local LLM inference service
rag-api     -> FastAPI RAG service
semantic-chunk -> offline semantic chunking service under tools profile
```

Why not put models into the image?

Models and indexes are large. If we package them into the image:

- Image build and distribution become slow.
- Every model update requires rebuilding the image.
- It is harder to use shared storage in production.

So the deployment uses mounted volumes for data, models, indexes, and vLLM model output.

Interview answer:

> I separated code from large runtime assets. The Docker image contains only code and dependencies, while models, indexes, and data are mounted as volumes. This keeps the image lightweight and makes model version switching much easier.

---

### 6. How Is 8-GPU Distributed Deployment Implemented?

The 8-GPU deployment uses vLLM tensor parallelism.

Core parameters:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
--tensor-parallel-size 8
```

Meaning:

- `CUDA_VISIBLE_DEVICES` tells vLLM which GPUs are available.
- `--tensor-parallel-size 8` splits model tensors across 8 GPUs.
- `VLLM_GPU_MEMORY_UTILIZATION=0.86` controls the GPU memory ratio used by vLLM.
- The API does not need to know whether the model runs on one GPU or eight GPUs. It still calls the same OpenAI-compatible vLLM endpoint.

Interview answer:

> The 8-GPU deployment is implemented at the model serving layer with vLLM tensor parallelism. The business API still calls the same OpenAI-compatible endpoint, while vLLM handles model parallelism internally. This decouples business logic from model serving infrastructure.

---

### 7. Common Interview Questions and English Answers

#### Q1: What is the core improvement in this project?

A: I upgraded the RAG demo into a more production-ready service. The main improvements include FastAPI productionization, Prometheus metrics, concurrency backpressure, Locust load testing, Docker Compose deployment, and 8-GPU vLLM tensor parallel inference.

#### Q2: Why did you separate the API service and vLLM service?

A: The API handles business logic and RAG orchestration, while vLLM handles model inference. They have different resource requirements. By separating them, we can scale, upgrade, and monitor them independently.

#### Q3: What is the difference between `/health` and `/ready`?

A: `/health` means the process is alive. `/ready` means the pipeline, indexes, and models are loaded and the service is ready to receive real traffic.

#### Q4: Why return 429 under high load?

A: 429 is a controlled backpressure signal. It is better to reject excessive traffic early than to let requests pile up, cause GPU OOM, or make tail latency uncontrollable.

#### Q5: What metrics do you monitor during load testing?

A: I monitor RPS, P95/P99 latency, failure rate, 429 ratio, GPU memory, and GPU utilization. Tail latency is especially important for RAG and LLM systems.

#### Q6: How would you optimize high P99 latency?

A: I would first identify the bottleneck. If retrieval is slow, I would reduce topk or optimize the index. If reranking is slow, I would reduce candidates or use a lighter reranker. If LLM inference is slow, I would tune vLLM parameters, max tokens, batching, or increase GPU parallelism.

#### Q7: Why not include the model inside the Docker image?

A: Model files are too large. Including them would make image builds and deployments slow. Mounting models as volumes keeps the image lightweight and allows independent model version management.

#### Q8: Is 8-GPU deployment always faster than single-GPU deployment?

A: Not always. Tensor parallelism improves capacity for large models and high-concurrency workloads, but it also introduces inter-GPU communication overhead. The final decision should be based on load testing results.

#### Q9: How do you make the load test realistic?

A: The Locust script reads questions from the existing QA test set, so request patterns are closer to real user questions instead of synthetic hello-world traffic.

#### Q10: What would you improve next for production?

A: I would add Redis caching for hot questions, distributed tracing, structured logging, Kubernetes HPA, a standalone Milvus cluster, canary release, and model version management.

---

## 三、30 秒中英文总结

中文：

> 我这个改造主要是把一个本地 RAG demo 做成更接近生产标准的在线服务。API 层增加了健康检查、就绪检查、Prometheus 指标和并发背压；压测用 Locust，并复用真实 QA 测试集；部署上用 Docker Compose 把 API、vLLM 和 MongoDB 拆开；八卡推理通过 vLLM tensor parallel 实现，业务代码不需要改，只需要调整部署参数。

English:

> This improvement turns a local RAG demo into a more production-ready online service. I added health checks, readiness checks, Prometheus metrics, and backpressure control to the API. Load testing is implemented with Locust using real QA data. Deployment is containerized with Docker Compose, separating the API, vLLM, and MongoDB. For 8-GPU inference, I use vLLM tensor parallelism, so the business API stays unchanged while the model serving layer scales independently.
