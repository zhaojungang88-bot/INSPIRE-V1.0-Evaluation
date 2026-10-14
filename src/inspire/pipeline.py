"""Basic INSPIRE V1.0 retrieval flow.

The study configuration is read from ``config/pipeline.json``. This module
implements that control flow with the Python standard library. It does not
bundle the embedding model, reranker weights, Chroma index, knowledge base,
or study data.
"""

from __future__ import annotations

from pathlib import Path
import json
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "pipeline.json"

USER_REQUEST = (
    "你是一名儿科医生，请根据以上内容对患儿进行诊断，并给出最可能的5种诊断结果，"
    "按照概率排序，做成表格仅用一列呈现疾病名称。"
)


def load_config(path: Path = CONFIG_PATH) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def tokenize(text: str) -> list[str]:
    """Count tokens for this public code.

    Each CJK character is one token. Other characters form tokens separated by
    whitespace. This is the counter used here; a separate historical tokenizer
    name was not recorded.
    """

    tokens: list[str] = []
    word: list[str] = []
    for char in text:
        if _is_cjk(char):
            if word:
                tokens.append("".join(word))
                word = []
            tokens.append(char)
        elif char.isspace():
            if word:
                tokens.append("".join(word))
                word = []
        else:
            word.append(char)
    if word:
        tokens.append("".join(word))
    return tokens


def detokenize(tokens: Iterable[str]) -> str:
    parts: list[str] = []
    previous_cjk = False
    for token in tokens:
        current_cjk = _is_cjk(token)
        if parts and not previous_cjk and not current_cjk:
            parts.append(" ")
        parts.append(token)
        previous_cjk = current_cjk
    return "".join(parts)


def build_query(case_text: str) -> str:
    """Use the supplied case text as the retrieval query.

    There is no intent label and no rewritten keyword query.
    """

    query = " ".join(case_text.split())
    if not query:
        raise ValueError("case_text is required")
    return query


def chunk_text(text: str, size: int = 512, overlap: int = 64) -> list[dict]:
    """Split one document into overlapping token windows."""

    if size <= 0:
        raise ValueError("chunk size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")
    tokens = tokenize(text)
    if not tokens:
        return []

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + size, len(tokens))
        window = tokens[start:end]
        chunks.append(
            {
                "text": detokenize(window),
                "token_count": len(window),
                "start_token": start,
            }
        )
        if end == len(tokens):
            break
        start = end - overlap
    return chunks


def select_candidates(query: str, chunks: list[dict], k: int = 20) -> list[dict]:
    """Return the k chunks with the highest lexical overlap.

    The evaluated system used Qwen3-Embedding-8B, Chroma, and cosine similarity.
    Those components are named in the configuration. This function is the
    runnable stand-in in the public code when the index is not shipped.
    """

    if k <= 0:
        raise ValueError("candidate count must be positive")
    ranked = []
    for index, chunk in enumerate(chunks):
        item = dict(chunk)
        item["retrieval_score"] = _overlap_score(query, chunk.get("text", ""))
        item["order"] = index
        ranked.append(item)
    ranked.sort(key=lambda item: (-item["retrieval_score"], item["order"]))
    return ranked[:k]


def rerank(query: str, candidates: list[dict]) -> list[dict]:
    """Order candidates for the evidence block.

    The evaluated reranker was BAAI/bge-reranker-v2-m3. Its weights are not in
    this repository. The public code keeps the same candidate order produced by
    ``select_candidates`` and records that score as the rerank score.
    """

    reranked = []
    for index, candidate in enumerate(candidates):
        item = dict(candidate)
        item["rerank_score"] = item.get("retrieval_score", _overlap_score(query, item.get("text", "")))
        item["rerank_order"] = index
        reranked.append(item)
    return reranked


def pack_evidence(ranked_chunks: list[dict], max_chunks: int = 10, max_tokens: int = 4000) -> dict:
    """Pack whole chunks in rank order under both public limits.

    A chunk is added only when the selected count is still below ``max_chunks``
    and the chunk's own token count still fits in ``max_tokens``. The overlap
    used at indexing time is not subtracted here. This code stops at the first
    chunk that does not fit; it does not pull a later, shorter chunk forward.
    """

    if max_chunks <= 0 or max_tokens <= 0:
        raise ValueError("evidence limits must be positive")
    selected = []
    used = 0
    for chunk in ranked_chunks:
        if len(selected) >= max_chunks:
            break
        count = int(chunk["token_count"])
        if count < 0:
            raise ValueError("token_count must be non-negative")
        if used + count > max_tokens:
            break
        selected.append(dict(chunk))
        used += count
    return {
        "chunks": selected,
        "chunk_count": len(selected),
        "token_count": used,
        "max_chunks": max_chunks,
        "max_tokens": max_tokens,
    }


def prepare(case_text: str, documents: list[dict], config: dict | None = None) -> dict:
    """Run query construction, candidate selection, reranking, and packing."""

    settings = config or load_config()
    query = build_query(case_text)
    chunks = []
    for document in documents:
        doc_id = document["id"]
        for chunk in chunk_text(
            document["text"],
            size=int(settings["chunk_size_tokens"]),
            overlap=int(settings["chunk_overlap_tokens"]),
        ):
            chunk["doc_id"] = doc_id
            chunks.append(chunk)

    candidates = select_candidates(query, chunks, k=int(settings["dense_candidates"]))
    reranked = rerank(query, candidates)
    evidence = pack_evidence(
        reranked,
        max_chunks=int(settings["selected_chunks"]),
        max_tokens=int(settings["evidence_token_limit"]),
    )
    return {
        "platform": settings["platform"],
        "query": query,
        "intent_recognition": None,
        "user_request": USER_REQUEST,
        "retrieval_backend": "public-lexical-stand-in",
        "candidate_count": len(candidates),
        "evidence": evidence,
        "diagnoses": None,
    }


def _overlap_score(query: str, text: str) -> float:
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0
    text_counts: dict[str, int] = {}
    for token in tokenize(text):
        text_counts[token] = text_counts.get(token, 0) + 1
    hits = sum(text_counts.get(token, 0) for token in query_tokens)
    return hits / len(query_tokens)


def _is_cjk(value: str) -> bool:
    return len(value) == 1 and "\u4e00" <= value <= "\u9fff"
