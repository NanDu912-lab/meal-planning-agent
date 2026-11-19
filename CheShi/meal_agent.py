import os
import sys
from pathlib import Path
from typing import Any

# ============================================================
# 0. 引入 util 目录（按要求写法）
# ============================================================
sys.path.insert(0, '../util')

import tools, rag, agent
from agentscope.tool import Toolkit, ToolResponse
from agentscope.message import Msg, TextBlock
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.dashscope import DashScopeEmbedding
os.environ.setdefault(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
os.environ.setdefault(
    "LLM_BASE_URL_QWEN",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ============================================================
# 1. 常量
# ============================================================
DOCS_DIR = "docs"                    # 菜品文档目录（dishes.7z 解压后）
INDEX_DIR = "../knowledge/dishes"    # 向量库持久化目录

MEAL_QA_TPL = (
    "你是一名专业的配餐顾问。请严格依据下方【参考资料】中的菜品信息来回答问题，"
    "不要编造资料里没有的菜品或营养数据。\n"
    "如果参考资料不足以回答，请明确说明“资料不足”，并给出一般性建议。\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "问题：{query_str}\n"
    "回答："
)

SYSTEM_PROMPT = """你是「智能配餐小助手」，一名专业的营养师兼厨师。
你的工作方式：
1. 当用户提出配餐需求时，先调用工具 search_dishes 从菜品知识库中检索候选菜品；
2. 根据检索结果，结合用户的健康目标（健康/请客/减肥）挑选合适菜品；
3. 最终给出结构清晰的菜单：包含菜名、简要理由，必要时标注份量/热量；
4. 不要编造知识库中没有的菜品；如果检索结果不足，要如实说明。
"""

QUESTIONS = [
    "请提供一周的健康早餐菜单。",
    "周末请客，想吃鱼，应该提供哪些菜单？",
    "提供一天的减肥菜单。"
]

# ============================================================
# 2. 准备 docs/ 文档目录（自动解压 dishes.7z）
# ============================================================
def prepare_docs_dir(docs_dir=DOCS_DIR):
    root = Path(docs_dir)
    def _list_files(directory):
        """列出目录下所有文件（递归），返回相对路径列表。"""
        if not directory.is_dir():
            return []
        return sorted(
            str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()
        )

    if root.is_dir() and _list_files(root):
        files = _list_files(root)
        print(f"[prepare_docs_dir] 文档目录已存在: {docs_dir}，共 {len(files)} 个文件：")
        for f in files[:10]:
            print(f"    - {f}")
        if len(files) > 10:
            print(f"    ... 其余 {len(files) - 10} 个略")
        # 已经有文件，直接返回
        return

    candidates = [p / "dishes.7z" for p in [Path("."), *Path(".").resolve().parents]]
    archive = next((c for c in candidates if c.exists()), None)
    if archive is None:
        raise SystemExit(
            f"\n[prepare_docs_dir] 错误：未找到文档目录 '{docs_dir}'，"
            f"也没有找到 dishes.7z。\n"
            f"请把 dishes.7z 放到脚本同目录后再运行。"
        )

    try:
        import py7zr
    except ImportError:
        raise SystemExit(
            "[prepare_docs_dir] 需要 py7zr，请先执行: pip install py7zr"
        )

    print(f"[prepare_docs_dir] 找到 {archive}，正在解压到 {docs_dir} ...")
    root.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(archive, mode="r") as z:
        z.extractall(path=docs_dir)

    # 3. 解压后扁平化：如果 docs/ 根目录下没有直接文件、只有一个子目录，
    #    则把那个子目录里的所有文件提升到 docs/ 根目录。
    top_items = list(root.iterdir())
    top_files = [p for p in top_items if p.is_file()]
    top_dirs = [p for p in top_items if p.is_dir()]
    if not top_files and len(top_dirs) == 1:
        sub = top_dirs[0]
        print(f"[prepare_docs_dir] 发现文件都在子目录 {sub}/，正在提升到 {docs_dir}/ ...")
        for f in sub.rglob("*"):
            if f.is_file():
                target = root / f.relative_to(sub)
                target.parent.mkdir(parents=True, exist_ok=True)
                f.rename(target)
        # 删掉空的子目录
        import shutil
        shutil.rmtree(sub, ignore_errors=True)

    files = _list_files(root)
    print(f"[prepare_docs_dir] 解压完成，共 {len(files)} 个文件：")
    for f in files[:10]:
        print(f"    - {f}")
    if len(files) > 10:
        print(f"    ... 其余 {len(files) - 10} 个略")


# ============================================================
# 3. 准备向量库
# ============================================================
def ensure_index_ready(index_dir=INDEX_DIR):
    if os.path.exists(index_dir):
        print(f"[ensure_index_ready] 发现已有向量库: {index_dir}")
        return

    print("[ensure_index_ready] 未发现向量库，开始构建...")
    prepare_docs_dir()

    print(f"[ensure_index_ready] 正在读取 {DOCS_DIR}/ 下的所有文件...")
    reader = SimpleDirectoryReader(DOCS_DIR, recursive=True)
    all_files = reader.input_files
    print(f"[ensure_index_ready] SimpleDirectoryReader 识别到 {len(all_files)} 个文件：")
    for f in all_files[:20]:
        print(f"    - {f}")

    documents = reader.load_data()
    print(f"[ensure_index_ready] 共解析出 {len(documents)} 个文档块，正在建索引...")

    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=DashScopeEmbedding(
            model_name="text-embedding-v4",
            embed_batch_size=10,
        ),
    )
    rag.store_index(index, index_dir)
    print(f"[ensure_index_ready] 向量库已保存到 {index_dir}")


# ============================================================
# 4. 把 RAG 查询封装成 Agent 工具
# ============================================================
_rag_engine = None
def get_rag_engine(index_dir=INDEX_DIR):
    """懒加载 RAG 引擎，并注入配餐专用提示词。"""
    global _rag_engine
    if _rag_engine is None:
        ensure_index_ready(index_dir)
        # 调用 rag.py 里的 create_rag_engine
        engine = rag.create_rag_engine(index_dir)
        # 配餐模板覆盖默认模板
        rag.update_prompt_tpl(engine, MEAL_QA_TPL)
        _rag_engine = engine
    return _rag_engine

def search_dishes(query: str) -> ToolResponse:
    """
    【工具：菜品知识库检索】
    根据自然语言从菜品库中检索相关菜品，返回由 LLM 整理后的说明。

    :param query: 检索词，如“低脂早餐”“清蒸鱼”“减肥餐”
    :return: ToolResponse，内含菜品说明文本
    """
    engine = get_rag_engine()
    resp = rag.ask(query, engine)          # rag.py 里的 ask(question, engine)
    text = str(resp)
    preview = text[:120].replace("\n", " ")
    print(f"\n[tool:search_dishes] query={query}\n  -> {preview}...\n")
    return ToolResponse(
        content=[TextBlock(type="text", text=text)]
    )

# ============================================================
# 5. 创建 ReAct 智能体（调用 agent.py 的工厂函数）
# ============================================================
def build_agent(toolkit: Toolkit, model_name: str = "qwen-max") -> Any:
    return agent.create_react_agent(
        name="配餐助手",
        sys_prompt=SYSTEM_PROMPT,
        model_name=model_name,
        multi_agent=False,
        toolkit=toolkit,
        max_iters=6,
    )


# ============================================================
# 6. 跑三个必答题
# ============================================================
async def run_questions(ag, questions=QUESTIONS):
    results = []
    for i, q in enumerate(questions, 1):
        print(f"\n{'=' * 60}\n问题 {i}: {q}\n{'=' * 60}")
        # 必须用 Msg 对象包装输入，不能直接传字符串
        msg = Msg(name="user", content=q, role="user")
        # ReActAgent.reply() 是异步协程，必须 await
        reply = await ag.reply(msg)
        # 从返回的 Msg 中取文本内容
        answer = reply.get_text_content() if hasattr(reply, "get_text_content") else str(reply)
        print(f"\n【回答】\n{answer}\n")
        results.append({"question": q, "answer": answer})
    return results


# ============================================================
# 7. 附加题：ragas 评估
# ============================================================
def build_eval_dataset(results):
    engine = get_rag_engine()
    gt_map = {
        "请提供一周的健康早餐菜单。":
            "一周早餐应主食、蛋白、蔬果搭配，热量适中、营养均衡，避免高油高糖。",
        "周末请客，想吃鱼，应该提供哪些菜单？":
            "应以鱼为主菜，搭配冷菜、热炒、汤和主食，口味兼顾宴请档次与宾客接受度。",
        "提供一天的减肥菜单。":
            "一天减肥餐应控制总热量，高蛋白、高纤维、低精制碳水，三餐规律。",
    }
    rows = []
    for r in results:
        q = r["question"]
        retrieved = engine.retriever.retrieve(q)
        contexts = [n.node.get_content() for n in retrieved]
        rows.append({
            "question": q,
            "answer": r["answer"],
            "contexts": contexts,
            "ground_truth": gt_map.get(q, "答案应贴合用户的配餐目标。"),
        })
    return rows


def evaluate_with_ragas(rows):
    """
    用 ragas 的 faithfulness + answer_relevancy 评估。
    配置 DashScope（qwen-max + text-embedding-v4）作为评估 LLM/Embedding。
    """
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import EvaluationDataset, evaluate
        from ragas.metrics import Faithfulness, AnswerRelevancy
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except Exception as e:
        print(
            f"[ragas] 导入失败"
            f"  错误详情: {e}"
        )
        return None
    base = os.getenv("LLM_BASE_URL_QWEN")
    key = os.getenv("DASHSCOPE_API_KEY")

    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(
        model="qwen-max",
        base_url=base,
        api_key=key,
    ))
    evaluator_emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
        model="text-embedding-v4",
        base_url=base,
        api_key=key,
    ))
    eval_rows = [
        {
            "user_input": r["question"],
            "response": r["answer"],
            "retrieved_contexts": r["contexts"],
            "reference": r["ground_truth"],
        }
        for r in rows
    ]
    dataset = EvaluationDataset.from_list(eval_rows)
    print("\n[ragas] 开始评估（faithfulness；answer_relevancy 因 embedding 兼容问题暂略）...")
    result = evaluate(
        dataset,
        metrics=[Faithfulness()],
        llm=evaluator_llm,
    )
    print("[ragas] 评估结果：")
    print(result)
    return result


async def async_main():
    # 1) 注册工具（注意：agentscope 的方法名是 register_tool_function）
    toolkit = Toolkit()
    toolkit.register_tool_function(search_dishes)
    # 2) 创建智能体
    ag = build_agent(toolkit)
    # 3) 跑三道必答题
    results = await run_questions(ag)
    # 4) 附加题：ragas 评估
    rows = build_eval_dataset(results)
    evaluate_with_ragas(rows)


def main():
    import asyncio
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
