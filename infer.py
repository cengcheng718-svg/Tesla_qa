# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: LLM任务型对话Agent
# 版权所有  ©丁师兄大模型
# --------------------------------------------


from src.client.llm_local_client import request_chat
from src.qa_pipeline import QAPipeline

# warmstart
pipeline = QAPipeline()


while True:
    query = input("输入—>")

    retrieval = pipeline.retrieve(query)

    print("BM25召回样例:")
    print(retrieval["bm25_docs"])
    print("="*100)

    print("BGE-M3召回样例:")
    print(retrieval["milvus_docs"])
    print("="*100)

    print(retrieval["merged_docs"])
    print("="*100)

    ranked_docs = retrieval["ranked_docs"]
    print(ranked_docs)
    print("="*100)


    # 答案
    context = pipeline.build_context(ranked_docs)
    res_handler = request_chat(query, context, stream=True)
    response = ""
    for r in res_handler:
        uttr = r.choices[0].delta.content or ""
        response += uttr 
        print(uttr, end='')
    print("\n" + "="*100)

    # 后处理
    answer = pipeline.post_process(response, ranked_docs)
    print("\n答案—>", answer)
