import os
from typing import List, Mapping

import requests


def _get_config():
    
    return {
        "endpoint": os.getenv("HW_LLM_ENDPOINT", "").strip(),
        "token": os.getenv("HW_LLM_TOKEN", "").strip(),
        "model": os.getenv("HW_LLM_MODEL", "").strip(),
    }


def call_llm(messages: List[Mapping[str, str]]) -> str:
    """
    messages: list of {"role": "user"/"system"/"assistant", "content": "..."}
    return: str, 大模型回复
    """
    cfg = _get_config()
    if not cfg["endpoint"] or not cfg["token"] or not cfg["model"]:
        return "【占位：未配置华为云大模型 endpoint/token/model，返回占位回复】"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg['token']}",
    }
    payload = {
        "model": cfg["model"],
        "messages": messages,
    }

    try:
        resp = requests.post(cfg["endpoint"], headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # 华为云 V2 Chat Completions 格式与 OpenAI 类似，后续可根据文档微调解析
        return (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "【未从响应中解析到内容】")
        )
    except Exception as exc:  # pragma: no cover - 运行时错误直接提示
        return f"【调用华为云大模型失败: {exc}】"
