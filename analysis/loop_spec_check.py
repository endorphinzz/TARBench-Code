"""Loop（论文中的 Endless Loop）标签审计：规格一致性 + 良性链上的同结构出现率。

1. §3.1 语义下的振荡判定：状态图有向环（置为当前值不触发）。仅对 500 个 Loop 片段做，规模小。
2. 结构规格：片段是逐条相连的依赖链，且最后一条规则的动作属性 == 第一条规则的触发属性。
3. 良性链：在"属性级触发→动作依赖图"上用 Tarjan 求强连通分量，统计含环的链数（线性时间）。
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from execution_semantics_audit import has_cycle, parse  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = HERE / "outputs" / "loop_spec_check.json"


def rules_of(c):
    keys = sorted((k for k in c if k.startswith("Rule_") and k[5:].isdigit()), key=lambda k: int(k[5:]))
    return [c[k] for k in keys]


def fragment(c):
    st, ln = int(c["abnormal_fragment_start_index"]), int(c["abnormal_fragment_length"])
    return [parse(c[f"Rule_{i}"]) for i in range(st, st + ln)]


def is_dependency_cycle(parsed):
    for a, b in zip(parsed, parsed[1:]):
        if a[1][0][0] != b[0][0][0]:
            return False
    return parsed[-1][1][0][0] == parsed[0][0][0][0]


def has_static_cycle(parsed):
    """属性依赖图（触发属性 -> 各动作属性）是否存在长度>=2 的环或自环。"""
    adj = {}
    for conds, acts in parsed:
        for c in conds:
            adj.setdefault(c[0], set()).update(a[0] for a in acts)
    index, low, on_stack, stack, counter, found = {}, {}, set(), [], [0], [False]

    def strongconnect(v):
        index[v] = low[v] = counter[0]; counter[0] += 1
        stack.append(v); on_stack.add(v)
        for w in adj.get(v, ()):
            if w not in index:
                strongconnect(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop(); on_stack.discard(w); comp.append(w)
                if w == v:
                    break
            if len(comp) > 1 or v in adj.get(v, ()):
                found[0] = True

    for v in list(adj):
        if v not in index:
            strongconnect(v)
    return found[0]


def main():
    abn = json.load(open(ROOT / "data/abnormal/abnormal_chains_gpt54.json", encoding="utf-8"))["chains"]
    ben = json.load(open(ROOT / "data/benign/benign_long_chains_gpt54.json", encoding="utf-8"))["chains"]
    loops = [c for c in abn if c.get("Type") == "Loop"]

    osc = term = unparsed = spec = 0
    for c in loops:
        parsed = fragment(c)
        if any(p is None for p in parsed):
            unparsed += 1
            continue
        spec += is_dependency_cycle(parsed)
        if has_cycle(parsed):
            osc += 1
        else:
            term += 1
    res = {"loop_fragments": len(loops), "oscillate_under_stated_semantics": osc, "terminate_after_one_pass": term,
           "unparsed": unparsed, "satisfy_structural_cycle_spec": spec}
    print(json.dumps(res), flush=True)

    ben_cycles = unparsed_ben = 0
    examples = []
    for c in ben:
        parsed = [parse(r) for r in rules_of(c)]
        if any(p is None for p in parsed):
            unparsed_ben += 1
            continue
        if has_static_cycle(parsed):
            ben_cycles += 1
            if len(examples) < 3:
                examples.append({"chain_id": c["chain_id"], "rules": rules_of(c)})
    res["benign_chains"] = len(ben)
    res["benign_with_static_dependency_cycle"] = ben_cycles
    res["benign_unparsed"] = unparsed_ben
    res["benign_examples"] = examples
    print(json.dumps({k: v for k, v in res.items() if k != "benign_examples"}, ensure_ascii=False), flush=True)
    for e in examples:
        print(f"  benign chain with a dependency cycle: {e['chain_id']}")
        for r in e["rules"]:
            print("     ", r)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
