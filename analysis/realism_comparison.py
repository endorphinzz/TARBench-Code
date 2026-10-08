"""E5 真实性对比：TARBench 合成良性链 vs 种子规则的分布差异。

对比对象：
  - 合成侧：benign_long_chains_gpt54.json（10,583 条 LLM 合成链，2-10 规则）
  - 种子侧：benign_single_rule_chains.json（1,380 条原始单规则，非 LLM 生成）

维度（规则级为主，链级差异如实注明为设计差异）：
  1) trigger / action attribute 边际分布（JS 散度）
  2) device 类别边际分布（JS 散度）
  3) trigger 比较词分布（== / < / > / != ...）
  4) 值类型分布（numeric / enum-word / boolean / time）
  5) trigger→action 设备对 top-K
  6) 规则文本长度（词数）分布

输出: analysis/outputs/realism_comparison.json + figures/realism_distributions.pdf
"""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYNTH = PROJECT_ROOT / "data" / "benign" / "benign_long_chains_gpt54.json"
SEED_FILE = PROJECT_ROOT / "data" / "benign" / "benign_single_rule_chains.json"
OUT_DIR = PROJECT_ROOT / "analysis" / "outputs"
OUT_PATH = OUT_DIR / "realism_comparison.json"
FIG_PATH = OUT_DIR / "figures" / "realism_distributions.pdf"

DEV_ATTR = re.compile(r"\b([a-z_]+)\.([a-z_]+)\b")
TIME_VAL = re.compile(r"^\d{1,2}:\d{2}$")


def split_rule(rule_text: str):
    if " THEN " in rule_text:
        cond, action = rule_text.split(" THEN ", 1)
        return cond.removeprefix("IF ").strip(), action.strip()
    return rule_text.strip(), ""


def value_type(value_token: str):
    v = value_token.strip().strip(";,")
    if TIME_VAL.match(v):
        return "time"
    if v in {"true", "false"}:
        return "boolean"
    if re.fullmatch(r"-?\d+(\.\d+)?", v):
        return "numeric"
    if re.fullmatch(r"[A-Za-z_]+", v):
        return "enum_word"
    return "other"


def extract_atoms(rules_iter):
    """rules_iter 产出 (trigger_text, action_text)。返回统计原子。"""
    trig_attr, act_attr, trig_dev, act_dev = Counter(), Counter(), Counter(), Counter()
    trig_op, val_types, dev_pairs, rule_len_words = Counter(), Counter(), Counter(), []
    for trigger, action in rules_iter:
        for token in DEV_ATTR.findall(trigger):
            trig_attr[token[1]] += 1
            trig_dev[token[0].split("_", 1)[-1] if "_" in token[0] else token[0]] += 1
        for token in DEV_ATTR.findall(action):
            act_attr[token[1]] += 1
            act_dev[token[0].split("_", 1)[-1] if "_" in token[0] else token[0]] += 1
        ops = re.findall(r"==|!=|>=|<=|>|<", trigger)
        trig_op.update(ops)
        values = [v for v in re.findall(r"(?:==|!=|>=|<=|>|<)\s*([^\s;]+)", trigger)]
        val_types.update(value_type(v) for v in values)
        pairs = set()
        for td in DEV_ATTR.findall(trigger):
            for ad in DEV_ATTR.findall(action):
                pairs.add((td[0].split("_", 1)[-1], ad[0].split("_", 1)[-1]))
        dev_pairs.update(pairs)
        rule_len_words.append(len(trigger.split()) + len(action.split()))
    return {
        "trig_attr": trig_attr, "act_attr": act_attr, "trig_dev": trig_dev, "act_dev": act_dev,
        "trig_op": trig_op, "val_types": val_types, "dev_pairs": dev_pairs,
        "rule_len_words": rule_len_words,
    }


def js_divergence(c1: Counter, c2: Counter):
    """对称 JS 散度（log2 底，0~1）。"""
    keys = set(c1) | set(c2)
    if not keys:
        return None
    p = np.array([c1.get(k, 0) for k in keys], dtype=float)
    q = np.array([c2.get(k, 0) for k in keys], dtype=float)
    p /= p.sum()
    q /= q.sum()
    m = (p + q) / 2
    mask_p, mask_q = p > 0, q > 0
    kl_pm = float(np.sum(p[mask_p] * np.log2(p[mask_p] / m[mask_p])))
    kl_qm = float(np.sum(q[mask_q] * np.log2(q[mask_q] / m[mask_q])))
    return float((kl_pm + kl_qm) / 2)


def top_table(counter: Counter, k=15):
    total = sum(counter.values())
    return [{"key": key, "count": count, "freq": round(count / total, 4)}
            for key, count in counter.most_common(k)]


def main():
    synth = json.loads(SYNTH.read_text(encoding="utf-8"))["chains"]
    seed_data = json.loads(SEED_FILE.read_text(encoding="utf-8"))["chains"]

    def synth_rules():
        for chain in synth:
            for key in sorted((k for k in chain if k.startswith("Rule_")), key=lambda k: int(k.split("_")[1])):
                yield split_rule(chain[key])

    def seed_rules():
        for chain in seed_data:
            for rule in chain["rules"]:
                yield rule["trigger"], rule["action"]

    s, g = extract_atoms(synth_rules()), extract_atoms(seed_rules())
    n_s_rules = sum(s["trig_attr"].values() and 1 for _ in [0])  # placeholder
    n_s = len(synth)  # chains
    n_g = len(seed_data)

    comparison = {
        "meta": {
            "synthetic_chains": n_s, "seed_chains": n_g,
            "note": "seed chains are single-rule by design; chain-level length comparison is a design difference, not a realism defect",
        },
        "rule_level": {},
        "chain_level": {},
    }

    # 规则级分布对比
    for name, c_s, c_g in [
        ("trigger_attribute", s["trig_attr"], g["trig_attr"]),
        ("action_attribute", s["act_attr"], g["act_attr"]),
        ("trigger_device", s["trig_dev"], g["trig_dev"]),
        ("action_device", s["act_dev"], g["act_dev"]),
        ("trigger_operator", s["trig_op"], g["trig_op"]),
        ("value_type", s["val_types"], g["val_types"]),
    ]:
        comparison["rule_level"][name] = {
            "jsd": js_divergence(c_s, c_g),
            "synthetic_top": top_table(c_s),
            "seed_top": top_table(c_g),
        }

    comparison["rule_level"]["trigger_to_action_device_pairs"] = {
        "synthetic_top": top_table(s["dev_pairs"]),
        "seed_top": top_table(g["dev_pairs"]),
    }
    comparison["rule_level"]["rule_text_length_words"] = {
        "synthetic_mean": float(np.mean(s["rule_len_words"])),
        "synthetic_std": float(np.std(s["rule_len_words"])),
        "seed_mean": float(np.mean(g["rule_len_words"])),
        "seed_std": float(np.std(g["rule_len_words"])),
    }

    # 链级：规则数分布（如实报告设计差异）
    synth_len = Counter()
    for chain in synth:
        synth_len[sum(1 for k in chain if k.startswith("Rule_"))] += 1
    comparison["chain_level"]["rules_per_chain"] = {
        "synthetic": {str(k): v for k, v in sorted(synth_len.items())},
        "seed": {"1": n_g},
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {OUT_PATH}")

    # 汇总打印
    print("\n== JSD (0=identical, 1=maximally different, log2) ==")
    for name in ["trigger_attribute", "action_attribute", "trigger_device", "action_device", "trigger_operator", "value_type"]:
        print(f"  {name:20s} JSD = {comparison['rule_level'][name]['jsd']:.4f}")
    print("\n== value_type distribution ==")
    for side, c in [("synthetic", s["val_types"]), ("seed", g["val_types"])]:
        t = sum(c.values())
        print(f"  {side:10s}", {k: f"{v / t:.3f}" for k, v in c.most_common()})
    print("\n== operator distribution ==")
    for side, c in [("synthetic", s["trig_op"]), ("seed", g["trig_op"])]:
        t = sum(c.values())
        print(f"  {side:10s}", {k: f"{v / t:.3f}" for k, v in c.most_common()})
    print("\n== rule length (words) ==")
    print(f"  synthetic {comparison['rule_level']['rule_text_length_words']['synthetic_mean']:.1f} ± "
          f"{comparison['rule_level']['rule_text_length_words']['synthetic_std']:.1f}")
    print(f"  seed      {comparison['rule_level']['rule_text_length_words']['seed_mean']:.1f} ± "
          f"{comparison['rule_level']['rule_text_length_words']['seed_std']:.1f}")

    # 附录图：4 面板分布对比
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
        panels = [
            ("trigger_attribute", "Trigger attribute frequency (top 12)"),
            ("action_attribute", "Action attribute frequency (top 12)"),
            ("trigger_operator", "Trigger operator distribution"),
            ("value_type", "Trigger value type distribution"),
        ]
        for ax, (key, title) in zip(axes.flat, panels):
            keys_s = [e["key"] for e in comparison["rule_level"][key]["synthetic_top"][:12]]
            keys_g = [e["key"] for e in comparison["rule_level"][key]["seed_top"][:12]]
            keys = list(dict.fromkeys(keys_s + keys_g))
            fs = {e["key"]: e["freq"] for e in comparison["rule_level"][key]["synthetic_top"]}
            fg = {e["key"]: e["freq"] for e in comparison["rule_level"][key]["seed_top"]}
            x = np.arange(len(keys))
            ax.bar(x - 0.2, [fs.get(k, 0) for k in keys], width=0.4, label="Synthetic chains")
            ax.bar(x + 0.2, [fg.get(k, 0) for k in keys], width=0.4, label="Seed rules")
            ax.set_xticks(x)
            ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=7)
            ax.set_title(title, fontsize=9)
            ax.legend(fontsize=7)
        fig.suptitle("TARBench synthetic benign chains vs. seed rules (rule-level distributions)", fontsize=10)
        fig.tight_layout()
        FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_PATH, bbox_inches="tight")
        print(f"figure -> {FIG_PATH}")
    except Exception as exc:  # noqa: BLE001
        print(f"figure skipped: {exc}")


if __name__ == "__main__":
    main()
