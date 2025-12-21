from pathlib import Path
from typing import Optional

import pandas as pd

from cloud.hw_llm_client import call_llm


def _load_experiments(base_dir: Optional[Path]):
    root = base_dir if base_dir else Path(__file__).resolve().parents[1]
    data_path = root / "data" / "experiments.csv"
    return pd.read_csv(data_path)


def _build_query_prompt(question: str) -> list[dict[str, str]]:
    schema = (
        "我有一张实验结果表，字段如下：\n"
        "- model: 模型名称\n"
        "- dataset: 数据集名称\n"
        "- metric: 指标名称，如 micro-F1, macro-F1\n"
        "- value: 指标数值，float\n"
        "- note: 备注\n\n"
        "请先用自然语言说明要过滤哪些行、按什么排序，"
        "再用伪代码表示，例如 filter(dataset='CAFA3', metric='micro-F1') -> sort_by(value, desc)。"
        "不要编造具体数值。"
    )
    return [
        {"role": "system", "content": "你是数据查询助手，负责把自然语言转成查询意图。"},
        {"role": "user", "content": schema + f"\n用户问题：{question}"},
    ]


def _apply_simple_query(df: pd.DataFrame, plan_text: str) -> pd.DataFrame:
    text = plan_text.lower()
    dataset = "CAFA3" if "cafa3" in text else None
    metric = None
    if "micro" in text:
        metric = "micro-F1"
    elif "macro" in text:
        metric = "macro-F1"

    filtered = df.copy()
    if dataset:
        filtered = filtered[filtered["dataset"] == dataset]
    if metric:
        filtered = filtered[filtered["metric"] == metric]
    if "desc" in text or "最高" in text or "最大" in text:
        filtered = filtered.sort_values("value", ascending=False)
    elif "asc" in text or "最低" in text or "最小" in text:
        filtered = filtered.sort_values("value", ascending=True)
    return filtered


def _describe_filters(plan_text: str) -> str:
    text = plan_text.lower()
    parts = []
    if "cafa3" in text:
        parts.append("dataset=CAFA3")
    if "micro" in text:
        parts.append("metric=micro-F1")
    elif "macro" in text:
        parts.append("metric=macro-F1")
    if "desc" in text or "最高" in text or "最大" in text:
        parts.append("value 降序")
    elif "asc" in text or "最低" in text or "最小" in text:
        parts.append("value 升序")
    return "，".join(parts) if parts else "无明显筛选/排序"


def _format_results(df: pd.DataFrame) -> str:
    if df.empty:
        return "没有查到符合条件的记录。"
    lines = [f"{row.model} | {row.dataset} | {row.metric} | {row.value} | {row.note}" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def answer_query_question(question: str, base_dir: Optional[Path] = None) -> str:
    """
    自然语言查数：先让大模型输出伪查询说明，再用 Pandas 真查询，必要时再让大模型润色。
    """
    df = _load_experiments(base_dir)
    plan_messages = _build_query_prompt(question)
    plan_text = call_llm(plan_messages)

    # 基于大模型生成的计划做土法解析；若占位则退化到关键字判断
    if "占位" in plan_text:
        plan_text = question  # 使用用户原始问题做关键字解析

    filtered = _apply_simple_query(df, plan_text)
    result_text = _format_results(filtered)

    # 让大模型润色最终回答，若不可用则返回原始结果
    answer_messages = [
        {
            "role": "system",
            "content": (
                "请将查询结果转为简短自然语言，必须基于给定的结果文本，不要编造或猜测数值；"
                "如果结果为空，请直接说明无法确定。"
            ),
        },
        {
            "role": "user",
            "content": f"用户问题：{question}\n查询计划：{plan_text}\n结果：\n{result_text}",
        },
    ]
    if filtered.empty:
        source_note = f"来源：data/experiments.csv；依据：{_describe_filters(plan_text)}"
        return f"未查到符合条件的实验记录，当前资料不足，无法确定答案。\n\n{source_note}"

    final_answer = call_llm(answer_messages)
    if "占位" in final_answer:
        final_answer = f"查询计划（推测）：{plan_text}\n结果：\n{result_text}"

    source_note = f"来源：data/experiments.csv；依据：{_describe_filters(plan_text)}"
    return f"{final_answer}\n\n{source_note}"
