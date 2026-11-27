"""构建不含案例文档的知识库；必要时重建 docs/abnormal_type_definitions.md。

发布版知识库的 abnormal_case / benign_pattern 文档含有评测片段的逐字副本与标注描述
（排除只在 chain_id 层面进行，而同一片段模板会出现在多条链中）。本脚本给出一个可执行的
对照配置：只保留不含任何评测样本内容的两类文档（abnormal_type_definition 与
contrastive_guideline），供 RAG 重跑对比。输出到 experiments/rag/knowledge_base_no_cases/。

docs/abnormal_type_definitions.md 被 build_knowledge_base.py 与 build_icl_library.py 引用；
若该文件缺失，本脚本从已发布知识库的类型定义文档逐字重建它。"""

import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
KB = BASE / "knowledge_base" / "knowledge_base.jsonl"
DOCS_DIR = ROOT / "docs"
TYPE_DEFS = DOCS_DIR / "abnormal_type_definitions.md"
OUT_DIR = BASE / "knowledge_base_no_cases"

KEEP_TYPES = {"abnormal_type_definition", "contrastive_guideline"}


def reconstruct_type_definitions(docs):
    """从 KB 的 TD 文档反重建 docs/abnormal_type_definitions.md。

    parse_type_definitions 的拼接是 name/definition/examples 的纯函数，
    将 "Examples: a b" 整体作为单条 example 写回，重建后 KB 文本逐字一致。
    """
    lines = []
    td_docs = sorted(
        (d for d in docs if d["doc_type"] == "abnormal_type_definition"),
        key=lambda d: d["doc_id"],
    )
    for doc in td_docs:
        index = int(doc["doc_id"].split("_")[1])
        definition = examples = None
        for line in doc["text"].split("\n"):
            if line.startswith("Definition: "):
                definition = line[len("Definition: "):]
            elif line.startswith("Examples: "):
                examples = line[len("Examples: "):]
        # text 行序: Anomaly Type / Definition / [Dataset Alias] / [Examples] / Diagnosis cue
        name = doc["metadata"]["type_name"]
        lines.append(f"{index}. {name}: {definition}")
        if examples:
            lines.append(f"- Example: {examples}")
        lines.append("")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    TYPE_DEFS.write_text("\n".join(lines), encoding="utf-8")
    return len(td_docs)


def main():
    docs = [json.loads(line) for line in KB.open(encoding="utf-8")]
    kept = [d for d in docs if d["doc_type"] in KEEP_TYPES]
    dropped = len(docs) - len(kept)

    if not TYPE_DEFS.exists():
        n = reconstruct_type_definitions(docs)
        print(f"Reconstructed {TYPE_DEFS} ({n} type definitions)")
    else:
        print(f"{TYPE_DEFS} already exists; left untouched")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "knowledge_base.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for doc in kept:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    counts = {}
    for d in kept:
        counts[d["doc_type"]] = counts.get(d["doc_type"], 0) + 1
    summary = {
        "purpose": "W4 ablation: KB without case docs (no abnormal_case / benign_pattern), "
        "so no evaluation fragment or gold description can be retrieved",
        "kept_documents": len(kept),
        "dropped_documents": dropped,
        "doc_type_counts": counts,
        "source_kb": str(KB.relative_to(ROOT)),
    }
    (OUT_DIR / "knowledge_base_no_cases_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
