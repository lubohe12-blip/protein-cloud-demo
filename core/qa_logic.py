import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from cloud.hw_llm_client import call_llm  # 预留大模型接口


def _load_papers(base_dir: Optional[Path]) -> List[Dict[str, Any]]:
    root = base_dir if base_dir else Path(__file__).resolve().parents[1]
    data_path = root / "data" / "papers.json"
    with data_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _search_papers(question: str, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    宽松匹配：根据 id/标题关键词/keywords/摘要中出现的片段计分，返回前 2 篇。
    """
    q = question.lower()
    scored: List[tuple[int, Dict[str, Any]]] = []
    for paper in papers:
        score = 0
        pid = str(paper.get("id", "")).lower()
        title = paper.get("title", "").lower()
        keywords = [kw.lower() for kw in paper.get("keywords", [])]
        summary = paper.get("summary", "").lower()
        details = paper.get("details", "").lower()

        if pid and pid in q:
            score += 3
        if title:
            # 按词分割，避免必须完整标题匹配
            score += sum(1 for token in title.split() if token and token in q)
        score += sum(2 for kw in keywords if kw and kw in q)
        if summary and any(token in q for token in summary.split()):
            score += 1
        if details and any(token in q for token in details.split()):
            score += 1

        if score > 0:
            scored.append((score, paper))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:2]]


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
    matched = _search_papers(question, papers)

    if not matched:
        source_note = _format_sources([])
        return f"未检索到与问题直接相关的文献，当前资料不足，无法确定答案。\n\n{source_note}"

    messages = _build_prompt(question, matched)
    response = call_llm(messages)
    if not response or "占位" in response:
        summaries = [
            f"{p['title']} ({p['year']}): {p.get('summary', '暂无摘要')}"
            for p in matched
        ]
        response = "基于检索到的文献摘要：\n" + "\n".join(summaries)

    source_note = _format_sources(matched)
    return f"{response}\n\n{source_note}"
