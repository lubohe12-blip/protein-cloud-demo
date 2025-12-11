import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from cloud.hw_llm_client import call_llm  # 预留大模型接口


def _load_papers(base_dir: Optional[Path]) -> List[Dict[str, Any]]:
    root = base_dir if base_dir else Path(__file__).resolve().parents[1]
    data_path = root / "data" / "papers.json"
    with data_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _match_paper(question: str, papers: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    question_lower = question.lower()
    for paper in papers:
        title_hit = paper.get("title", "").lower()
        keyword_hits = " ".join(paper.get("keywords", [])).lower()
        if any(term in question_lower for term in [paper.get("id", ""), title_hit]) or any(
            kw in question_lower for kw in keyword_hits.split()
        ):
            return paper
    return None


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
        "如果我提供了参考内容，你优先基于参考内容回答；"
        "否则就用你自己的知识回答。不要在回答中提到“文献片段”“参考资料”等字眼。"
    )
    user_prompt = (
         f"【可能有用的参考内容】\n{context_text}\n\n"
         f"【用户问题】\n{question}\n\n"
        "请给出条理清晰的回答。如果参考内容与问题关系不大，可以只用你自己的知识回答。"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def answer_literature_question(question: str, base_dir: Optional[Path] = None) -> str:
    """
    简单检索 + prompt 构造后调用华为云大模型占位接口。
    本地仍保留关键字匹配逻辑，确保在未配置大模型时也能返回结果。
    """
    papers = _load_papers(base_dir)
    matched = _match_paper(question, papers)

    if matched:
        messages = _build_prompt(question, [matched])
        return call_llm(messages)

    available = ", ".join(p["title"] for p in papers)
    fallback = (
        "当前知识库很小，只包含几篇示例论文。"
        f" 你可以尝试提问：{available}。"
    )
    messages = _build_prompt(question, [])
    response = call_llm(messages)
    # 若仍为占位，返回引导语
    if "占位" in response:
        return fallback
    return response
