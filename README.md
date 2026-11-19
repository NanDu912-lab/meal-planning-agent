# 智能配餐小助手（RAG + Agent）

基于 AgentScope 的配餐 Agent：扮演营养师兼厨师，先用检索拿到候选菜品，再按用户的健康目标
（日常健康 / 请客点鱼 / 减脂菜单）生成结构化菜单。

## 技术栈
- AgentScope：Agent 与工具注册（`Toolkit` / `ToolResponse`）
- LlamaIndex：菜品文档 → 向量知识库，并把索引持久化到 `knowledge/dishes/`
- 通义千问（DashScope compatible-mode）：Embedding 与生成
- Function Calling：`search_dishes` 工具，菜单里的每道菜都能回溯到知识库文档

## 目录
```
CheShi/meal_agent.py              主程序：Agent + 工具 + 批量场景测试
CheShi/docs/dishes/**/*.md        菜品知识库语料（370 篇）
CheShi/meal_planning_results.json 3 类场景的批量跑批结果
util/                             模型调用、RAG、工具函数
```

## 运行
代码里的 `../util`、`docs`、`../knowledge/dishes` 都是相对路径，必须**在 `CheShi/` 目录下启动**：
```bash
pip install -r requirements.txt
cd CheShi
python meal_agent.py
```
首次运行会提示输入 DashScope API Key，写入 `util/config/keys.json`（已在 `.gitignore` 中，
不会被提交）。也可以先设置环境变量 `DASHSCOPE_API_KEY`。

向量索引目录 `knowledge/` 不随仓库提供，首次运行时由 `meal_agent.py` 中的
`ensure_index_ready()` 读取 `docs/dishes/` 自动重建。

## 可靠性设计
- Prompt 中做了防幻觉约束：菜品与热量必须来自检索结果，不允许凭空生成。
- 生成后校验环节可拦截数值幻觉并触发修正，形成「检索 - 生成 - 验证」闭环。
- 内置 3 类场景测试集（一周早餐 / 请客点鱼 / 减肥菜单），结果落盘为 JSON 便于复查。

> 说明：仓库内的菜品图片未提交（体积原因，`.gitignore` 已排除），只保留文本语料。
