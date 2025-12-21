import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from cloud.hw_llm_client import call_llm  # 预留大模型接口


def _load_papers(base_dir: Optional[Path]) -> List[Dict[str, Any]]:
    root = base_dir if base_dir else Path(__file__).resolve().parents[1]
    data_path = root / "data" / "papers.json"
    with data_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _match_papers(question: str, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    question_lower = question.lower()
    matches: List[Dict[str, Any]] = []
    for paper in papers:
        title_hit = paper.get("title", "").lower()
        keyword_hits = " ".join(paper.get("keywords", [])).lower()
        if any(term in question_lower for term in [paper.get("id", ""), title_hit]) or any(
            kw in question_lower for kw in keyword_hits.split()
        ):
            matches.append(paper)
    return matches


def _build_prompt(question: str, contexts: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    context_blocks = []
    for idx, paper in enumerate(contexts, start=1):
        context_blocks.append(
            f"[文献{idx}] {paper['title']} ({paper['year']})\n"
            f"关键词: {', '.join(paper.get('keywords', []))}\n"
            f"摘要: {paper.get('summary', '')}\n"
            f"笔记: {paper.get('details', '')}"
        )
    context_text = "\n\n".join(context_blocks) if context_blocks else "（无匹配上下文，仅用常识简要回答）"

    system_prompt = (
        "你是一个蛋白质结构与功能预测方向的科研助手。"
        "必须基于我提供的参考内容组织回答；"
        "如果参考内容不足以支持结论，直接说明“无法确定”，不要臆测或编造。"
    )
    user_prompt = (
         f"【可能有用的参考内容】\n{context_text}\n\n"
         f"【用户问题】\n{question}\n\n"
        "请给出条理清晰的回答，并标注使用了哪些文献编号。"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _format_sources(papers: List[Dict[str, Any]]) -> str:
    if not papers:
        return "来源：data/papers.json（未检索到匹配文献）"

    parts = [
        f"[文献{idx}] {paper.get('title', '未知标题')} ({paper.get('year', '未知年份')})"
        for idx, paper in enumerate(papers, start=1)
    ]
    return "来源：" + "；".join(parts) + "；数据文件：data/papers.json"


def answer_literature_question(question: str, base_dir: Optional[Path] = None) -> str:
    """
    简单检索 + prompt 构造后调用华为云大模型占位接口。
    本地仍保留关键字匹配逻辑，确保在未配置大模型时也能返回结果。
    """
    papers = _load_papers(base_dir)
    matched = _match_papers(question, papers)

    if not matched:
        source_note = _format_sources([])
        return f"未检索到与问题直接相关的文献，当前资料不足，无法确定答案。\n\n{source_note}"

    messages = _build_prompt(question, matched)
    response = call_llm(messages)
    if "占位" in response:
        response = "检索到相关文献，但大模型未返回有效内容，无法确定详细答案。"

    source_note = _format_sources(matched)
    return f"{response}\n\n{source_note}"
