"""E9 微调补充实验：SFT 数据构造与三重防泄漏切分。

产出（experiments/finetune/）:
  datasets/splits/   train_full / dev / test_t1 / test_t2 / test_t3 / train_scale_{500,2000,8000}
  datasets/sft_*.json        LLaMA-Factory sharegpt 格式（含 cls_only 变体）
  datasets/prompt_contract.json  system/user 模板（单一事实来源 = icl/zero_shot_prompt.md）
  split_audit.json           切分审计报告（排除计数 / 不相交断言 / token 长度分布）

防泄漏三重约束:
  (1) chain_id 级: 测试链 (T1/T2/T3) 不入 train/dev
  (2) 片段模板级: 同一片段模板 (插入区间规则文本的规范化元组) 的链不跨 train/test
      —— 注: ground_truth.abnormal_fragment_id_list 只是插入位置顺序编号 (R_{start}..)，
         不是片段身份; 实测 4,600 异常链 = 918 个片段模板 × 5 条链 (2026-10-08)
  (3) 规则文本级: train 样本与任一测试样本的规则文本集 Jaccard >= 0.8 则剔除

测试集 (训练前冻结, SEED=20261008):
  T1 主测试   ~2,000  (分层留出: 异常按片段模板组整组采样 ~667, 良性 2x)
  T2 交叉引用   750   (icl_dataset.json 原样, 与论文 Table 3 同集)
  T3 OOD       584   (icl_eval_variants_300_284.json, 设备共享构造)
  T3 与 train 的片段模板重合仅审计报告 (OOD 压力集, 对照锚 = B-LEX transfer 同样全量训练)

用法:
  python experiments/finetune/build_sft_data.py
  python experiments/finetune/build_sft_data.py --skip-tokenizer   # 本地无 transformers 时
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

FT_DIR = Path(__file__).resolve().parent
EXP_DIR = FT_DIR.parent
# TARBench_Code/experiments/finetune/ -> TARBench_Code/
ROOT_DIR = FT_DIR.parents[1]

FULL_DATASET = ROOT_DIR / "experiments" / "datasets" / "full_dataset.json"
ICL_DATASET = ROOT_DIR / "experiments" / "datasets" / "icl_dataset.json"
VARIANTS_DATASET = ROOT_DIR / "experiments" / "icl" / "datasets" / "icl_eval_variants_300_284.json"
PROMPT_MD = ROOT_DIR / "experiments" / "icl" / "zero_shot_prompt.md"

OUTPUT_DIR = FT_DIR / "datasets"
SPLITS_DIR = OUTPUT_DIR / "splits"

SEED = 20261008
T1_ABNORMAL_TARGET = 667   # 异常目标数; 片段模板整组采样会轻微超出
DEV_TOTAL = 500
SCALE_SUBSETS = [500, 2000]  # 全量点 = train_full 本身 (2:1 约束下 ~7.5k)
JACCARD_THRESHOLD = 0.8
TRAIN_BENIGN_PER_ABNORMAL = 2  # 训练类别先验与所有评测集一致 (2:1), 避免 FT 的 FPR 先验优势
BENIGN_KEY_FIELDS = ("scenario_group", "method", "chain_length")
ABNORMAL_KEY_FIELDS = ("anomaly_type", "sub_category")


# ---------------------------------------------------------------- 基础工具

def load_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as f:  # full_dataset.json 带 BOM
        return json.load(f)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def norm_rule(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def rule_ids(sample) -> list[str]:
    return sorted(sample["rules"].keys(), key=lambda k: int(k.split("_")[1]))


def sample_rule_set(sample) -> set[str]:
    return {norm_rule(sample["rules"][rid]) for rid in rule_ids(sample)}


def fragment_span(sample) -> tuple[int, int] | None:
    gt = sample.get("ground_truth", {})
    start = gt.get("abnormal_fragment_start_index")
    length = gt.get("abnormal_fragment_length")
    if gt.get("has_anomaly") is not True or start is None or length is None:
        return None
    return int(start), int(length)


def fragment_template(sample) -> tuple[str, ...] | None:
    """片段模板身份 = 插入区间规则文本的规范化元组。"""
    span = fragment_span(sample)
    if span is None:
        return None
    start, length = span
    rules = sample["rules"]
    return tuple(norm_rule(rules[f"Rule_{i}"]) for i in range(start, start + length) if f"Rule_{i}" in rules)


def gt_involved_rules(sample) -> list[str]:
    """与 evaluate_icl.ground_truth_involved_rules 完全一致的定位 GT。"""
    gt = sample.get("ground_truth", {})
    if gt.get("has_anomaly") is not True:
        return []
    span = fragment_span(sample)
    if span is None:
        return []
    start, length = span
    return [f"Rule_{i}" for i in range(start, start + length)]


def parse_prompt_md(path: Path) -> dict[str, str]:
    """镜像 run_icl_benchmark.load_prompt_template 的宽松解析。"""
    text = path.read_text(encoding="utf-8")
    sections: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        m = re.match(r"^##\s+(.+?)\s*$", line)
        if m:
            current = m.group(1).strip().lower()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    system = "\n".join(sections.get("system prompt", [])).strip()
    user = "\n".join(sections.get("user prompt", [])).strip()
    if "{samples_text}" not in user:
        raise ValueError("user prompt 模板缺少 {samples_text} 占位符")
    return {"system": system, "user": user}


def stratified_sample(samples: list[dict], count: int, key_fields: tuple[str, ...] | list[tuple],
                      seed: int) -> list[dict]:
    """round-robin 分层采样 (与 build_datasets.diverse_sample 同策略)。

    key_fields 为取值元组的字段名列表; 异常样本的 anomaly_type/sub_category
    从 ground_truth 取。
    """
    rng = random.Random(seed)

    def key_of(s: dict) -> tuple:
        vals = []
        for f in key_fields:
            if f in ("anomaly_type", "sub_category"):
                vals.append(s.get("ground_truth", {}).get(f) or "__missing__")
            else:
                vals.append(s.get(f) if s.get(f) is not None else "__missing__")
        return tuple(vals)

    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for s in samples:
        buckets[key_of(s)].append(s)
    bucket_keys = sorted(buckets, key=lambda k: tuple(str(x) for x in k))
    for b in buckets.values():
        rng.shuffle(b)
    rng.shuffle(bucket_keys)

    selected: list[dict] = []
    while len(selected) < count:
        added = False
        for k in bucket_keys:
            b = buckets[k]
            if b:
                selected.append(b.pop())
                added = True
                if len(selected) >= count:
                    break
        if not added:
            break
    return selected


# ---------------------------------------------------------------- SFT 渲染

def render_prompt_item(sample: dict, sample_index: int = 1) -> str:
    """与 run_icl_benchmark.build_zero_shot_prompt_item 完全一致的单样本条目。"""
    return json.dumps(
        {
            "sample_index": sample_index,
            "case_id": sample["chain_id"],
            "scenario": sample.get("scenario"),
            "scenario_group": sample.get("scenario_group"),
            "chain_length": sample.get("chain_length"),
            "rules": sample["rules"],
        },
        ensure_ascii=False,
        indent=2,
    )


def build_user_text(sample: dict, template: dict) -> str:
    samples_text = "[\n" + render_prompt_item(sample) + "\n]"
    return template["user"].replace("{samples_text}", samples_text)


def build_assistant_text(sample: dict, cls_only: bool = False) -> str:
    gt = sample.get("ground_truth", {})
    if gt.get("has_anomaly") is not True:
        result = {
            "sample_index": 1,
            "has_anomaly": False,
            "reason": "No anomalies identified.",
            "involved_rules": [],
        }
    else:
        description = str(gt.get("description") or "").strip()
        anomaly_type = gt.get("anomaly_type") or "Anomaly"
        result = {
            "sample_index": 1,
            "has_anomaly": True,
            "reason": f"1.[{anomaly_type}]: {description}",
            "involved_rules": gt_involved_rules(sample),
        }
    if cls_only:
        result = {"sample_index": 1, "has_anomaly": result["has_anomaly"]}
    return json.dumps({"results": [result]}, ensure_ascii=False)


def render_sft(samples: list[dict], template: dict, cls_only: bool = False) -> list[dict]:
    rows = []
    for s in samples:
        rows.append(
            {
                "conversations": [
                    {"from": "human", "value": build_user_text(s, template)},
                    {"from": "gpt", "value": build_assistant_text(s, cls_only=cls_only)},
                ],
                "system": template["system"],
            }
        )
    return rows


def write_sft(path: Path, samples: list[dict], template: dict, cls_only: bool = False) -> None:
    write_json(path, render_sft(samples, template, cls_only=cls_only))
    with path.with_suffix(".meta.jsonl").open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps({"sample_id": s["sample_id"], "chain_id": s["chain_id"]},
                               ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- 数据集封装

def pack(name: str, samples: list[dict], extra_meta: dict | None = None) -> dict:
    meta = {
        "dataset": name,
        "sampling_seed": SEED,
        "sample_counts": {
            "benign": sum(1 for s in samples if s["label"] == "benign"),
            "abnormal": sum(1 for s in samples if s["label"] == "abnormal"),
            "total": len(samples),
        },
    }
    if extra_meta:
        meta.update(extra_meta)
    return {"metadata": meta, "samples": samples}


def renumber(samples: list[dict], prefix: str, seed: int) -> list[dict]:
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    for i, s in enumerate(shuffled, start=1):
        s["sample_index"] = i
        s["sample_id"] = f"{prefix}_{s['label'].upper()}_{i:06d}"
        s["mixed_index"] = i
    return shuffled


# ---------------------------------------------------------------- Jaccard 过滤

def build_rule_index(samples: list[dict]) -> dict[str, set[str]]:
    idx: dict[str, set[str]] = defaultdict(set)
    for s in samples:
        for r in sample_rule_set(s):
            idx[r].add(s["sample_id"])
    return idx


def jaccard_filter(candidates: list[dict], test_samples: list[dict],
                   threshold: float) -> tuple[list[dict], list[dict], float]:
    """返回 (保留, 剔除, 保留集中最大残余 Jaccard)。"""
    test_sets: dict[str, set[str]] = {s["sample_id"]: sample_rule_set(s) for s in test_samples}
    index = build_rule_index(test_samples)
    kept, removed = [], []
    max_residual = 0.0
    for s in candidates:
        rs = sample_rule_set(s)
        cand_tests: set[str] = set()
        for r in rs:
            cand_tests.update(index.get(r, ()))
        drop = False
        best = 0.0
        for tid in cand_tests:
            ts = test_sets[tid]
            inter = len(rs & ts)
            union = len(rs | ts)
            j = inter / union if union else 0.0
            best = max(best, j)
            if j >= threshold:
                drop = True
                break
        if drop:
            removed.append(s)
        else:
            kept.append(s)
            max_residual = max(max_residual, best)
    return kept, removed, max_residual


# ---------------------------------------------------------------- token 统计

def token_stats(texts: list[str], tokenizer_name: str | None) -> dict:
    texts = texts[:3000]  # 采样上限, 足够估分布
    if tokenizer_name:
        try:
            import os
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
            lens = [len(tok.encode(t)) for t in texts]
            method = f"transformers:{tokenizer_name} (n={len(texts)})"
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] tokenizer 不可用, 退化为空白符估计: {exc}")
            lens = None
    else:
        lens = None
    if lens is None:
        lens = [len(t.split()) for t in texts]
        method = f"whitespace (n={len(texts)})"
    lens_sorted = sorted(lens)

    def pct(p: float) -> int:
        return lens_sorted[min(len(lens_sorted) - 1, int(p * len(lens_sorted)))]

    return {
        "method": method,
        "p50": pct(0.50), "p90": pct(0.90), "p99": pct(0.99),
        "max": lens_sorted[-1],
        "gt_2048": sum(1 for x in lens if x > 2048),
        "gt_4096": sum(1 for x in lens if x > 4096),
    }


# ---------------------------------------------------------------- 主流程

def main() -> None:
    global SEED
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--t1-abnormal", type=int, default=T1_ABNORMAL_TARGET)
    ap.add_argument("--dev-total", type=int, default=DEV_TOTAL)
    ap.add_argument("--jaccard", type=float, default=JACCARD_THRESHOLD)
    ap.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-8B")
    ap.add_argument("--skip-tokenizer", action="store_true")
    args = ap.parse_args()
    SEED = args.seed

    full = load_json(FULL_DATASET)["samples"]
    icl = load_json(ICL_DATASET)["samples"]
    variants = load_json(VARIANTS_DATASET)["samples"]
    template = parse_prompt_md(PROMPT_MD)
    audit: dict = {"seed": SEED, "jaccard_threshold": args.jaccard,
                   "inputs": {"full": len(full), "icl": len(icl), "variants": len(variants)}}

    # ---- T2 / T3 直接复用 (保留原 sample_id 以便与既有结果交叉对照)
    t2 = list(icl)
    t3 = list(variants)
    test_chain_ids = {s["chain_id"] for s in t2} | {s["chain_id"] for s in t3}

    # ---- T1: 从剩余池分层留出 (异常按片段模板整组采样)
    pool = [s for s in full if s["chain_id"] not in test_chain_ids]
    t2_templates = {fragment_template(s) for s in t2 if s["label"] == "abnormal"}
    t2_templates.discard(None)

    abnormal_groups: dict[tuple, list[dict]] = defaultdict(list)
    abnormal_group_keys: dict[tuple, tuple] = {}
    for s in pool:
        if s["label"] != "abnormal":
            continue
        tpl = fragment_template(s)
        if tpl is None:
            tpl = ("__ungrouped__", s["chain_id"])  # 缺区间信息的链自成一组
            audit.setdefault("ungrouped_abnormal", 0)
            audit["ungrouped_abnormal"] += 1
        if tpl in t2_templates:
            continue  # T2 已部分使用的模板, 不再进 T1 (保持测试模板互斥)
        abnormal_groups[tpl].append(s)
        g = s.get("ground_truth", {})
        abnormal_group_keys[tpl] = (g.get("anomaly_type") or "__missing__",
                                    g.get("sub_category") or "__missing__")

    rng = random.Random(SEED)
    group_items = list(abnormal_groups.items())
    rng.shuffle(group_items)
    buckets: dict[tuple, list[tuple]] = defaultdict(list)
    for tpl, chains in group_items:
        buckets[abnormal_group_keys[tpl]].append((tpl, chains))
    bucket_order = sorted(buckets, key=lambda k: tuple(str(x) for x in k))

    t1_abnormal: list[dict] = []
    used_templates: set[tuple] = set()
    bi = 0
    while len(t1_abnormal) < args.t1_abnormal and any(buckets[k] for k in bucket_order):
        k = bucket_order[bi % len(bucket_order)]
        bi += 1
        if not buckets[k]:
            continue
        tpl, chains = buckets[k].pop(rng.randrange(len(buckets[k])))
        t1_abnormal.extend(chains)
        used_templates.add(tpl)

    benign_pool = [s for s in pool if s["label"] == "benign"]
    t1_benign = stratified_sample(benign_pool, 2 * len(t1_abnormal), BENIGN_KEY_FIELDS, SEED + 1)
    t1 = renumber(t1_benign + t1_abnormal, "T1", SEED + 2)
    audit["t1"] = {
        "abnormal_chains": len(t1_abnormal), "benign": len(t1_benign), "total": len(t1),
        "fragment_templates": len(used_templates),
    }

    # ---- 训练池三重防泄漏
    test_templates = {fragment_template(s) for s in t1_abnormal}
    test_templates.update(t2_templates)
    train_pool = [s for s in pool if s["chain_id"] not in {x["chain_id"] for x in t1}]

    chain_excluded = [s for s in train_pool if s["chain_id"] in test_chain_ids]
    train_pool = [s for s in train_pool if s["chain_id"] not in test_chain_ids]

    frag_excluded = []
    kept_after_frag = []
    for s in train_pool:
        tpl = fragment_template(s)
        if tpl is not None and tpl in test_templates:
            frag_excluded.append(s)
        else:
            kept_after_frag.append(s)

    all_test = t1 + t2 + t3
    train_final, jac_excluded, max_residual = jaccard_filter(
        kept_after_frag, all_test, args.jaccard)

    # ---- 类别比例对齐: 良性下采样至 2:1 (与 T1/T2/T3 评测先验一致)
    pre_bn = [s for s in train_final if s["label"] == "benign"]
    pre_ab = [s for s in train_final if s["label"] == "abnormal"]
    bn_cap = TRAIN_BENIGN_PER_ABNORMAL * len(pre_ab)
    if len(pre_bn) > bn_cap:
        kept_bn = stratified_sample(pre_bn, bn_cap, BENIGN_KEY_FIELDS, SEED + 5)
        audit["benign_ratio_cap"] = {
            "before": len(pre_bn), "after": len(kept_bn),
            "abnormal": len(pre_ab),
            "note": f"下采样至 {TRAIN_BENIGN_PER_ABNORMAL}:1, 与评测集先验一致",
        }
        train_final = kept_bn + pre_ab

    audit["exclusions"] = {
        "chain_id_overlap_with_T2_T3": len(chain_excluded),
        "fragment_template_overlap_with_T1_T2": len(frag_excluded),
        "rule_jaccard_ge_threshold": len(jac_excluded),
        "max_residual_jaccard_in_train": round(max_residual, 4),
    }

    # ---- T3 片段模板重合审计 (默认只报告不剔除; 对照锚 B-LEX transfer 同为全量训练)
    t3_templates = {fragment_template(s) for s in t3 if s["label"] == "abnormal"}
    t3_templates.discard(None)
    train_templates_now = {fragment_template(s) for s in train_final if s["label"] == "abnormal"}
    train_templates_now.discard(None)
    audit["t3_fragment_overlap_report_only"] = {
        "t3_templates": len(t3_templates),
        "also_in_train_exact_match": len(t3_templates & train_templates_now),
        "note": "T3 为 OOD 压力集; 变体片段经改写, 精确匹配为重合下界",
    }

    # ---- 断言: 不相交性
    train_ids = {s["chain_id"] for s in train_final}
    test_ids = {s["chain_id"] for s in all_test}
    assert not (train_ids & test_ids), "chain_id 泄漏"
    train_templates_final = {t for t in (fragment_template(s) for s in train_final
                                         if s["label"] == "abnormal") if t is not None}
    assert not ({t for t in test_templates if t is not None} & train_templates_final), "片段模板泄漏"

    # ---- dev / scale 子集
    tr_benign = [s for s in train_final if s["label"] == "benign"]
    tr_abnormal = [s for s in train_final if s["label"] == "abnormal"]
    dev_ab_n = min(args.dev_total // 3, len(tr_abnormal))
    dev = (stratified_sample(tr_benign, args.dev_total - dev_ab_n, BENIGN_KEY_FIELDS, SEED + 3)
           + stratified_sample(tr_abnormal, dev_ab_n, ABNORMAL_KEY_FIELDS, SEED + 3))
    dev_ids = {s["chain_id"] for s in dev}
    train_final = [s for s in train_final if s["chain_id"] not in dev_ids]

    scale_sets: dict[int, list[dict]] = {}
    tr_benign = [s for s in train_final if s["label"] == "benign"]
    tr_abnormal = [s for s in train_final if s["label"] == "abnormal"]
    for n in SCALE_SUBSETS:
        ab_n = min(n // 3, len(tr_abnormal))
        bn_n = min(n - ab_n, len(tr_benign))
        scale_sets[n] = (stratified_sample(tr_benign, bn_n, BENIGN_KEY_FIELDS, SEED + 10 + n)
                         + stratified_sample(tr_abnormal, ab_n, ABNORMAL_KEY_FIELDS, SEED + 10 + n))

    # ---- 落盘: splits
    write_json(SPLITS_DIR / "train_full.json", pack("ft_train", train_final))
    write_json(SPLITS_DIR / "dev.json", pack("ft_dev", renumber(dev, "DEV", SEED + 4)))
    write_json(SPLITS_DIR / "test_t1.json", pack("ft_test_t1", t1, {"role": "held-out main test"}))
    write_json(SPLITS_DIR / "test_t2_icl750.json", pack("ft_test_t2", t2, {"role": "cross-reference with Table 3"}))
    write_json(SPLITS_DIR / "test_t3_variants.json", pack("ft_test_t3", t3, {"role": "OOD stress: device-shared variants"}))
    for n, ss in scale_sets.items():
        write_json(SPLITS_DIR / f"train_scale_{n}.json", pack(f"ft_train_scale_{n}", ss))

    # ---- 落盘: SFT (sharegpt) + prompt 契约
    write_sft(OUTPUT_DIR / "sft_train_full.json", train_final, template)
    write_sft(OUTPUT_DIR / "sft_dev.json", dev, template)
    write_sft(OUTPUT_DIR / "sft_train_cls_only.json", train_final, template, cls_only=True)
    for n, ss in scale_sets.items():
        write_sft(OUTPUT_DIR / f"sft_train_scale_{n}.json", ss, template)
    write_json(OUTPUT_DIR / "prompt_contract.json", {
        "source": str(PROMPT_MD),
        "system": template["system"],
        "user_template": template["user"],
        "assistant_contract": '{"results": [{"sample_index": 1, "has_anomaly": bool, "reason": str, "involved_rules": [str]}]}',
    })

    # ---- token 长度分布
    if not args.skip_tokenizer:
        probe = render_sft(train_final[:3000], template)
        full_texts = [p["system"] + p["conversations"][0]["value"] + p["conversations"][1]["value"]
                      for p in probe]
        audit["token_stats_train"] = token_stats(full_texts, args.tokenizer)
    else:
        audit["token_stats_train"] = token_stats(
            [json.dumps(s["rules"], ensure_ascii=False) for s in train_final[:3000]], None)

    # ---- 审计汇总
    def dist(samples: list[dict], field: str) -> dict:
        c = Counter()
        for s in samples:
            if field in ("anomaly_type", "sub_category"):
                if s["label"] == "abnormal":
                    c[s.get("ground_truth", {}).get(field) or "__missing__"] += 1
            else:
                c[str(s.get(field))] += 1
        return dict(sorted(c.items(), key=lambda kv: kv[0]))

    audit["final_counts"] = {
        "train": {"benign": len(tr_benign), "abnormal": len(tr_abnormal), "total": len(train_final)},
        "dev": {"total": len(dev), "abnormal": dev_ab_n},
        "t1": {"total": len(t1), **audit["t1"]},
        "scales": {n: len(ss) for n, ss in scale_sets.items()},
    }
    audit["distributions"] = {
        "train_abnormal_type": dist(train_final, "anomaly_type"),
        "t1_abnormal_type": dist(t1, "anomaly_type"),
    }
    write_json(FT_DIR / "split_audit.json", audit)

    print(f"[E9] train={len(train_final)} (bn {len(tr_benign)} / ab {len(tr_abnormal)}), "
          f"dev={len(dev)}, T1={len(t1)}, T2={len(t2)}, T3={len(t3)}")
    print(f"[E9] exclusions: {audit['exclusions']}")
    print(f"[E9] outputs -> {OUTPUT_DIR}")
    if "token_stats_train" in audit:
        print(f"[E9] token stats: {audit['token_stats_train']}")


if __name__ == "__main__":
    main()
