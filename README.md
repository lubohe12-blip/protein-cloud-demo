# 面向蛋白质预测领域的智能文献问答与数据查询系统（华为云大模型预留版）

## 本地快速启动
1) 创建虚拟环境并安装依赖
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
2) 运行 Streamlit
```
streamlit run app.py
```

## 功能概览
- 文献问答：基于 `data/papers.json` 的简单检索，拼接上下文 prompt 调用 `cloud/hw_llm_client.call_llm`（未配置时返回占位）。
- 实验查数：先让大模型生成“伪查询说明”，再用 Pandas 在 `data/experiments.csv` 上执行过滤/排序，结果可再由大模型润色。

## 华为云大模型配置（预留）
`cloud/hw_llm_client.py` 读取以下环境变量：
- `HW_LLM_ENDPOINT`：如 `https://xxx.pangu.huaweicloud.com/api/v2/chat/completions`
- `HW_LLM_TOKEN`：Bearer token 或签名后 token
- `HW_LLM_MODEL`：部署的模型名称/ID

未配置时将返回占位回复，方便本地链路先跑通。

## 部署到华为云 ECS 的建议步骤
1) 安装依赖：`pip install -r requirements.txt`
2) 设置环境变量：`export HW_LLM_ENDPOINT=...`, `HW_LLM_TOKEN=...`, `HW_LLM_MODEL=...`
3) 启动服务：`streamlit run app.py --server.port 8000 --server.address 0.0.0.0`
4) 安全组放行端口 8000（或用 Nginx 反代 80 → 8000）。
