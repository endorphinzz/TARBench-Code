"""按论文 §3.1 的执行语义模拟 Loop 片段，判断是否真的不终止。

论文语义：触发器 (d == v) 在 d 的状态"变为" v 时触发；动作 (d := v) 把 d 置为 v。
推论：把设备置为它当前的值不产生状态变化，不会再次触发。

判定：把片段视为状态图，节点是各设备取值的组合，边是"某规则触发且确实改变了状态"。
每条边都改变状态，所以一次不终止的振荡当且仅当可达状态图中存在有向环。
初始状态取片段内出现过的所有取值组合；不被任何规则写入的传感器视为外部常量并逐一枚举
（不等式条件额外枚举一个使之为真的取值）。为对标签有利，初始即为真的触发器也算触发。
无法用本脚本的三元组语法解析的片段记为 unparsed。

输出: analysis/outputs/execution_semantics_audit.json（含每个子类的计数与若干样例）
"""

import itertools
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "outputs" / "execution_semantics_audit.json"
RULE = re.compile(r"^IF\s+(.+?)\s+THEN\s+(.+)$")
TRIPLE = re.compile(r"([a-z0-9_]+\.[a-z0-9_]+)\s*(==|>=|<=|>|<|=)\s*([a-z0-9_.:-]+)")
TYPES = ("Loop", "Illegal State Rule", "Condition Block", "Condition Bypass")
LOOP_TYPE = "Loop"


def parse(rule):
    m = RULE.match(rule.strip())
    if not m:
        return None
    conds = [TRIPLE.findall(c) for c in re.split(r"\s+AND\s+", m.group(1))]
    acts = [t for t in TRIPLE.findall(m.group(2)) if t[1] == "="]
    if any(len(c) != 1 for c in conds) or not acts or any(t[1] == "=" for c in conds for t in c):
        return None
    return [c[0] for c in conds], acts


def holds(cond, state):
    key, op, val = cond
    cur = state.get(key)
    if cur is None:
        return False
    try:
        c, v = float(cur), float(val)
        return {"==": c == v, ">": c > v, "<": c < v, ">=": c >= v, "<=": c <= v}[op]
    except ValueError:
        return cur == val


def extra_values(cond):
    """让不等式成立的一个取值。"""
    key, op, val = cond
    try:
        v = float(val)
    except ValueError:
        return set()
    return {">": {v + 1}, "<": {v - 1}, ">=": {v}, "<=": {v}}.get(op, set())


def has_cycle(parsed):
    devices = sorted({k for conds, acts in parsed for k, _, _ in conds + acts})
    written = {k for _, acts in parsed for k, _, _ in acts}
    values = {k: set() for k in devices}
    for conds, acts in parsed:
        for c in conds:
            values[c[0]].add(c[2])
            if c[0] not in written:
                values[c[0]] |= {str(x) for x in extra_values(c)}
        for k, _, v in acts:
            values[k].add(v)
    external = [k for k in devices if k not in written]
    internal = [k for k in devices if k in written]
    ext_combos = list(itertools.product(*[sorted(values[k]) for k in external])) if external else [()]
    int_combos = list(itertools.product(*[sorted(values[k]) + [None] for k in internal]))

    def successors(state):
        out = []
        for conds, acts in parsed:
            if all(holds(c, state) for c in conds):
                new = dict(state)
                changed = False
                for k, _, v in acts:
                    if new.get(k) != v:
                        new[k] = v
                        changed = True
                if changed:
                    out.append(tuple(sorted(new.items())))
        return out

    for ext in ext_combos:
        seen, on_path = {}, set()

        def dfs(node):
            if node in on_path:
                return True
            if node in seen:
                return False
            seen[node] = True
            on_path.add(node)
            state = dict(node)
            for nxt in successors(state):
                if dfs(nxt):
                    return True
            on_path.discard(node)
            return False

        for init in int_combos:
            start = dict(zip(external, ext))
            start.update(zip(internal, init))
            if dfs(tuple(sorted(start.items()))):
                return True
    return False


def main():
    chains = json.load(open(ROOT / "data/abnormal/abnormal_chains_gpt54.json", encoding="utf-8"))["chains"]
    report, examples = {}, {}
    for want in TYPES:
        sel = [c for c in chains if c.get("Type") == want]
        row = {"n_chains": len(sel), "sub_categories": {}}
        examples[want] = []
        step = max(1, len(sel) // 5)
        for idx, c in enumerate(sel):
            st, ln = int(c["abnormal_fragment_start_index"]), int(c["abnormal_fragment_length"])
            frag = [c[f"Rule_{i}"] for i in range(st, st + ln)]
            sub = c.get("Sub_Category")
            row["sub_categories"].setdefault(sub, {"n": 0, "oscillates": 0, "terminates": 0, "unparsed": 0})
            r = row["sub_categories"][sub]
            r["n"] += 1
            if want == LOOP_TYPE:
                parsed = [parse(x) for x in frag]
                if any(p is None for p in parsed):
                    r["unparsed"] += 1
                elif has_cycle(parsed):
                    r["oscillates"] += 1
                else:
                    r["terminates"] += 1
            if idx % step == 0 and len(examples[want]) < 5:
                examples[want].append({"chain_id": c["chain_id"], "sub": sub, "rules": frag, "desc": c.get("Description")})
        report[want] = row
    report["examples"] = examples
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for t in TYPES:
        print(f"== {t}: n={report[t]['n_chains']}")
        for sub, r in sorted(report[t]["sub_categories"].items(), key=lambda kv: -kv[1]["n"]):
            print(f"   {str(sub)[:52]:52s} n={r['n']:4d} oscillates={r['oscillates']:4d} terminates={r['terminates']:4d} unparsed={r['unparsed']}")
    for t in ("Loop", "Illegal State Rule", "Condition Block", "Condition Bypass"):
        print(f"\n--- {t} examples ---")
        for e in examples[t][:3]:
            print(f"  [{e['sub']}] {e['desc']}")
            for r in e["rules"]:
                print("     ", r)


if __name__ == "__main__":
    main()
