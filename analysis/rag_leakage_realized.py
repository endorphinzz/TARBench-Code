"""RAG 检索实现率（离线，不调用模型）。

用发布版知识库与检索器（同 top_k 配置）对 RAG 评测集的异常样本逐一检索，
统计 top-k 异常案例文档中包含该样本片段（逐字规则重叠）的比例。

输出: analysis/outputs/rag_leakage_realized.json"""

import json
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "experiments" / "rag"
sys.path.insert(0, str(BASE))
from retriever import KnowledgeBaseRetriever  # noqa: E402

KB = BASE / "knowledge_base" / "knowledge_base.jsonl"
RAG_DS = BASE.parent / "datasets" / "rag_dataset.json"
TOP_K = {"abnormal_type_definition": 4, "abnormal_case": 5, "benign_pattern": 5, "contrastive_guideline": 4}

norm = lambda t: re.sub(r"\s+", " ", t.strip().lower())


def fragment_rule_texts(sample):
    gt = sample.get("ground_truth") or {}
    st, ln = gt.get("abnormal_fragment_start_index"), gt.get("abnormal_fragment_length")
    if st is None or ln is None:
        return []
    return [norm(sample["rules"][f"Rule_{i}"]) for i in range(int(st), int(st) + int(ln))
            if f"Rule_{i}" in sample["rules"]]


def doc_rule_texts(doc):
    m = re.search(r"Involved Rule Text:\n(.*?)\nFull Rule Chain:", doc["text"], re.S)
    if not m:
        return []
    return [norm(x.split(":", 1)[1]) for x in m.group(1).strip().split("\n") if ":" in x]


def main():
    kb = KnowledgeBaseRetriever.from_jsonl(KB)
    rag = json.load(open(RAG_DS, encoding="utf-8-sig"))
    samples = rag["samples"] if isinstance(rag, dict) else rag
    abn = [s for s in samples if s.get("label") == "abnormal"]

    any_overlap = full_overlap = rank1 = 0
    ranks = []
    for idx, s in enumerate(abn, 1):
        frag = set(fragment_rule_texts(s))
        retrieved = kb.retrieve(s, TOP_K).get("abnormal_case", [])
        hit_rank = None
        for rank, hit in enumerate(retrieved, 1):
            doc_rules = set(doc_rule_texts(hit["doc"]))
            if frag and doc_rules & frag:
                hit_rank = rank
                if frag <= doc_rules:
                    full_overlap += 1
                break
        if hit_rank is not None:
            any_overlap += 1
            ranks.append(hit_rank)
            if hit_rank == 1:
                rank1 += 1
        if idx % 200 == 0:
            print(f"{idx}/{len(abn)}: any={any_overlap} ({any_overlap/idx:.1%})", flush=True)

    out = {
        "n_eval_abnormal": len(abn),
        "retrieved_doc": "top-5 abnormal_case, published KB, published retriever, published top_k config",
        "at_least_one_rule_overlap": any_overlap,
        "rate_any": any_overlap / len(abn),
        "full_fragment_contained": full_overlap,
        "rate_full": full_overlap / len(abn),
        "hit_at_rank1": rank1,
        "rank_distribution": {str(r): ranks.count(r) for r in sorted(set(ranks))},
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    Path(__file__).resolve().parent.joinpath("outputs", "rag_leakage_realized.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
