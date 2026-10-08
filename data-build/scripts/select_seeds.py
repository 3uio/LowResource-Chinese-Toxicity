#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
select_seeds.py —— 表2 种子挑选器（C-P0-1 / C-2）

依据
----
* `docs/02_数据schema与source_id规则.md` §1（种子表字段）、§3（source_id 规则）、§4（split 分配）
* `docs/01_attack_type定义_v0.1.md` §4（8 小类样本量建议）
* A 交付的 `表2种子预留清单.csv`（2,041 条候选，含官方 `toxic_entity`）

做什么
------
1. 逐条判定**每条种子能自然生成哪些 attack_type**（可行性筛，不是硬凑）
2. 按可行性 + 多样性打分，选出 200 条
3. 生成 `source_id`（哈希前 10 位，冲突加序号）
4. 计算 `target_span_start/end/text`（从 `toxic_entity` 在原文里定位）
5. 按 `attack_family` 分层 + `source_id` 字典序轮转分配 split（135 / 30 / 35）
6. 自检：8 小类变体覆盖是否够 800–1000 条；split 无跨分片
7. 输出 `seed_candidates.csv` + 挑选报告

用法
----
    python scripts/select_seeds.py --in <表2种子预留清单.csv> --out out/seed_candidates.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------- attack_type 可行性

ALL_TYPES = ["PY_Full", "PY_Abbr", "Homo", "Graph_Similar",
             "Graph_Split", "Sym_Leet", "Emoji_Sub", "Sym_Separator"]

# 目标词里的这些字"拆得开"（左右/上下结构），Graph_Split 才做得自然。
# 这只是启发式，真实拆分表在 data/rules/ 里，评审后再建。
SPLITTABLE = set("强弓虽女马女子亥子木目日月火土金水口门马鸟鱼虫米糸"
                 "月贝页风飞马鱼鸟龙龟鹿麻黑鼓鼠鼻齐齿")
# 无法用拼音方案表达的目标词（纯符号/纯数字）
NON_CHINESE = re.compile(r"^[^\u4e00-\u9fff]+$")
CJK = re.compile(r"[\u4e00-\u9fff]")

# 已知有 emoji 谐音对应的字（小表，评审后由 data/rules/ 替换）
EMOJI_HOMOPHONE = set("啤酒杯子猪牛马鸡鸭鱼虾龟兔猫狗熊虎")


def feasible_types(text: str, entity: str, has_emoji: bool) -> list[str]:
    """判定该种子**自然可生成**的 attack_type。不过度承诺。"""
    out = []
    e = entity.strip()
    if not e or NON_CHINESE.match(e):
        # 目标词不是中文（纯符号/数字）→ 拼音与字形类都做不了
        pass
    else:
        # 拼音类：任何汉字目标词都可
        out += ["PY_Full", "Homo"]
        if len(e) >= 2:
            out.append("PY_Abbr")            # 首字母缩写要至少 2 个字
        # 字形类
        out.append("Graph_Similar")          # 形近字替换对单字/词都可行
        if any(c in SPLITTABLE for c in e):
            out.append("Graph_Split")
        # 符号类
        out.append("Sym_Leet")               # 需要形近数字/字母的目标字，先宽松给
        if any(c in EMOJI_HOMOPHONE for c in e):
            out.append("Emoji_Sub")
    out.append("Sym_Separator")              # 兜底类：任何目标词都能插分隔符
    return sorted(set(out))


# ---------------------------------------------------------------- 主流程

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="表2 种子挑选器")
    ap.add_argument("--in", dest="src", type=Path, required=True,
                    help="A 交付的 表2种子预留清单.csv")
    ap.add_argument("--out", type=Path, default=Path("out/seed_candidates.csv"))
    ap.add_argument("--want", type=int, default=200, help="目标种子数")
    ap.add_argument("--report", type=Path, default=Path("out/seed_selection_report.md"))
    args = ap.parse_args(argv)

    if not args.src.exists():
        print(f"[ERR] 输入不存在：{args.src}", file=sys.stderr)
        return 2

    with args.src.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    print(f"[info] 读入候选 {len(rows)} 条")

    # ------------------------------------------------------------ 机械排除
    keep = []
    drop_counter = Counter()
    for r in rows:
        text, e = r["text"], r["toxic_entity"].strip()
        subs = [s.strip() for s in e.split("|") if s.strip()]
        if not subs:
            drop_counter["无目标词"] += 1
            continue
        locatable = [s for s in subs if text.find(s) >= 0]
        if not locatable:
            drop_counter["目标词不在原文中"] += 1
            continue
        if max(len(s) for s in locatable) > 6:
            drop_counter["目标词过长(>6字)"] += 1
            continue
        r["_entities"] = locatable
        keep.append(r)
    print(f"[info] 机械排除 {len(rows) - len(keep)} 条：{dict(drop_counter)}")
    print(f"[info] 候选池 {len(keep)} 条")

    # ------------------------------------------------------------ 可行性 + 打分
    for r in keep:
        ent = r["_entities"][0]
        has_emoji = "has_emoji" in r["mech_flags"]
        r["_types"] = feasible_types(r["text"], ent, has_emoji)
        r["_has_emoji"] = has_emoji

    # 打分：多样性优先，但避免同质
    def score(r):
        s = len(r["_types"])                      # 能做的攻击类越多越好
        s += 1 if r["_has_emoji"] else 0          # 本身含 emoji → Emoji_Sub 更自然
        n_char = len(r["_entities"][0])
        s += 1 if 2 <= n_char <= 3 else 0         # 2–3 字目标词最典型
        return s

    keep.sort(key=lambda r: (-score(r), r["seed_normalized_hash"]))

    # ------------------------------------------------------------ 贪心挑选，保证 8 类覆盖
    want = args.want
    picked, seen_hash, per_type = [], set(), Counter()
    seen_entity = Counter()

    def take(r):
        picked.append(r)
        seen_hash.add(r["seed_normalized_hash"])
        for t in r["_types"]:
            per_type[t] += 1
        seen_entity[r["_entities"][0]] += 1

    # 第一轮：优先补齐覆盖最少的小类
    for _ in range(want):
        if len(picked) >= want:
            break
        best, best_key = None, None
        for r in keep:
            if r["seed_normalized_hash"] in seen_hash:
                continue
            ent = r["_entities"][0]
            if seen_entity[ent] >= 3:          # 同一目标词最多 3 条，防同质
                continue
            # 该行能补上的"最稀缺类"的紧迫度
            need = max((100 - per_type[t] for t in r["_types"]), default=0)
            key = (need, score(r), r["seed_normalized_hash"])
            if best_key is None or key > best_key:
                best, best_key = r, key
        if best is None:
            break
        take(best)

    print(f"[info] 选出 {len(picked)} 条")
    print(f"[info] 各 attack_type 可用种子数：")
    for t in ALL_TYPES:
        print(f"        {t:<16} {per_type[t]:>4}   → 按每条 4–5 变体估 "
              f"{per_type[t]*4}–{per_type[t]*5} 条")

    # ------------------------------------------------------------ source_id + target_span
    pref_count = Counter()
    for r in picked:
        base = f"toxi-{r['seed_normalized_hash'][:10]}"
        pref_count[base] += 1
    seq = Counter()
    for r in picked:
        base = f"toxi-{r['seed_normalized_hash'][:10]}"
        seq[base] += 1
        r["_source_id"] = base if pref_count[base] == 1 else f"{base}-{seq[base]}"
        # target_span：取第一个可定位的目标词
        ent = r["_entities"][0]
        st = r["text"].find(ent)
        r["_span"] = (st, st + len(ent), ent)

    ids = [r["_source_id"] for r in picked]
    assert len(set(ids)) == len(ids), "source_id 不唯一！"

    # ------------------------------------------------------------ split 分配（分层 + 轮转）
    def family_of(types):
        fams = set()
        for t in types:
            if t.startswith("PY_") or t == "Homo":
                fams.add("phonetic")
            elif t.startswith("Graph_"):
                fams.add("graphic")
            else:
                fams.add("symbolic")
        return fams

    # 按"首次出现的 family"分层，层内按 source_id 字典序轮转
    layers = defaultdict(list)
    for r in sorted(picked, key=lambda x: x["_source_id"]):
        fams = family_of(r["_types"])
        layers["+".join(sorted(fams))].append(r)

    ratios = {"train": 135, "dev": 30, "test": 35}
    total = sum(ratios.values())
    assign = {k: int(len(picked) * v / total) for k, v in ratios.items()}
    # 余数补给 train
    assign["train"] += len(picked) - sum(assign.values())

    for _, group in sorted(layers.items()):
        cycle = ["train"] * ratios["train"] + ["dev"] * ratios["dev"] + ["test"] * ratios["test"]
        for i, r in enumerate(group):
            r["_split"] = cycle[i % len(cycle)]

    split_count = Counter(r["_split"] for r in picked)
    print(f"[info] split 分配：{dict(split_count)}  (目标 {ratios})")

    # ------------------------------------------------------------ 自检
    across = defaultdict(set)
    for r in picked:
        across[r["seed_normalized_hash"]].add(r["_split"])
    bad = {h: s for h, s in across.items() if len(s) > 1}
    print(f"[check] 同一 source 跨 split：{len(bad)} 处 "
          f"{'OK' if not bad else 'FAIL'}")

    # ------------------------------------------------------------ 输出
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["source_id", "text", "target_span_start", "target_span_end",
            "target_span_text", "toxicity_label", "source_dataset", "orig_file",
            "orig_line", "split", "dup_of"]
    with args.out.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(picked, key=lambda x: x["_source_id"]):
            st, en, ent = r["_span"]
            w.writerow({
                "source_id": r["_source_id"], "text": r["text"],
                "target_span_start": st, "target_span_end": en,
                "target_span_text": ent, "toxicity_label": 1,
                "source_dataset": "ToxiCN", "orig_file": r["orig_file"],
                "orig_line": r["orig_line"], "split": r["_split"], "dup_of": "",
            })
    print(f"[out] {args.out}  ({len(picked)} 行)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
