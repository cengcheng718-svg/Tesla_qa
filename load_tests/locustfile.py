# -*- coding: utf-8 -*-
"""
Tesla 车书问答服务压测脚本。

典型用法：
  locust -f load_tests/locustfile.py --host http://127.0.0.1:8080
  locust -f load_tests/locustfile.py --host http://127.0.0.1:8080 --headless -u 20 -r 2 -t 5m --csv reports/tesla_qa
"""

import json
import os
import random
from pathlib import Path

from locust import HttpUser, between, task


DEFAULT_QUERIES = [
    "介绍一下离车后自动上锁功能",
    "Model 3 最多可以配对几部蓝牙手机？",
    "如果充电接口闩锁未锁上，交流充电会被限制到多少安？",
    "哨兵模式在什么情况下会向手机应用程序发送通知？",
]


def _load_queries() -> list[str]:
    """优先复用项目测试集，保证压测流量接近真实用户问题分布。"""
    qa_path = Path(os.getenv("LOAD_TEST_QA_PATH", "data/qa_pairs/test_qa_pair.json"))
    if not qa_path.exists():
        return DEFAULT_QUERIES

    with qa_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    queries = [
        item["question"].strip()
        for item in data
        if isinstance(item, dict) and item.get("question", "").strip()
    ]
    return queries or DEFAULT_QUERIES


QUERIES = _load_queries()
REQUEST_TIMEOUT = float(os.getenv("LOAD_TEST_TIMEOUT_SECONDS", "180"))
ACCEPT_429 = os.getenv("LOAD_TEST_ACCEPT_429", "0") == "1"


def _optional_int_env(name: str) -> int | None:
    value = os.getenv(name)
    return int(value) if value else None


class TeslaQAUser(HttpUser):
    wait_time = between(
        float(os.getenv("LOAD_TEST_MIN_WAIT_SECONDS", "0.2")),
        float(os.getenv("LOAD_TEST_MAX_WAIT_SECONDS", "1.2")),
    )

    @task(1)
    def health(self):
        self.client.get("/health", name="/health", timeout=10)

    @task(9)
    def chat(self):
        payload = {
            "query": random.choice(QUERIES),
            "return_context": os.getenv("LOAD_TEST_RETURN_CONTEXT", "0") == "1",
        }
        for env_name, field_name in [
            ("LOAD_TEST_BM25_TOPK", "bm25_topk"),
            ("LOAD_TEST_MILVUS_TOPK", "milvus_topk"),
            ("LOAD_TEST_RERANK_TOPK", "rerank_topk"),
        ]:
            value = _optional_int_env(env_name)
            if value is not None:
                payload[field_name] = value

        with self.client.post(
            "/chat",
            json=payload,
            name="/chat",
            timeout=REQUEST_TIMEOUT,
            catch_response=True,
        ) as response:
            if response.status_code == 429 and ACCEPT_429:
                response.success()
                return
            if response.status_code != 200:
                response.failure(f"unexpected status code: {response.status_code}")
                return

            try:
                body = response.json()
            except ValueError:
                response.failure("response is not json")
                return

            if not body.get("answer"):
                response.failure("empty answer")
            elif "timings" not in body:
                response.failure("missing timings")
            else:
                response.success()
