#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
curate_homophone.py —— 从候选表分出「可直接用」与「需人工判断」两类

分类逻辑（三轴）
----------------
1. **声调 + 语料证据** → 脚本已在候选表里算好（confidence 列）
2. **生僻字** → 机械可判：候选里最生僻那个字的语料频次（min_char_freq 列）
3. **是否成词** → ⚠️ 需要人的语感，脚本判不了

输出两张表
----------
* `homophone_word.csv`        —— **可直接用**：A/C 级 且 无生僻字
* `_candidates/需人工确认_B级.csv` —— **待在词形**：B 级（音对，但不知道是不是词）

用法
----
    python scripts/curate_homophone.py
    python scripts/curate_homophone.py --min-char-freq 10
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

T = Path(__file__).resolve().parent.parent
CAND = T / "data" / "rules" / "_candidates" / "homophone_word_candidates.csv"
OUT_OK = T / "data" / "rules" / "homophone_word.csv"
OUT_REVIEW = T / "data" / "rules" / "_candidates" / "需人工确认_B级.csv"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="谐音词表筛选")
    ap.add_argument("--min-char-freq", type=int, default=10,
                    help="候选里每个字在语料中的最低频次（低于此视为生僻字）")
    ap.add_argument("--max-per-word", type=int, default=2,
                    help="每个目标词最多取几条（docs/01 §4：每词 1–2 条即可）")
    args = ap.parse_args(argv)

    rows = [r for r in csv.DictReader(CAND.open(encoding="utf-8-sig"))
            if r["candidate"]]
    print(f"[info] 候选 {len(rows)} 条，生僻字门槛 min_char_freq >= {args.min_char_freq}")

    ok, review, rejected = [], [], Counter()
    for r in rows:
        conf = r["confidence"]
        minf = int(r["min_char_freq"])
        if conf == "D_diff_tone":
            rejected["D 级（弱谐音）"] += 1
            continue
        if conf == "B_same_tone":
            review.append(r)
            continue
        # A / C 级
        if minf < args.min_char_freq:
            rejected["A/C 级但含生僻字"] += 1
            continue
        ok.append(r)

    # 每个目标词限条数，优先语料证据多的
    grouped = defaultdict(list)
    for r in ok:
        grouped[r["target_word"]].append(r)
    final = []
    for w, items in grouped.items():
        items.sort(key=lambda r: (-int(r["attested_in_corpus"]), r["candidate"]))
        final.extend(items[:args.max_per_word])
    final.sort(key=lambda r: (-int(r["attested_in_corpus"]), r["target_word"]))

    print(f"\n[结果]")
    print(f"  ✅ 可直接用（A/C 级 + 无生僻字 + 每词≤{args.max_per_word}）: {len(final)} 条，"
          f"覆盖 {len({r['target_word'] for r in final})} 个目标词")
    print(f"  ⚠️ 需人工确认（B 级，音对但不知是否成词）      : {len(review)} 条，"
          f"覆盖 {len({r['target_word'] for r in review})} 个目标词")
    for k, v in rejected.items():
        print(f"  ❌ 丢弃 {k:<22}: {v} 条")

    # ---- 成品表：生成器读取格式 char,replacements
    OUT_OK.parent.mkdir(parents=True, exist_ok=True)
    by_word = defaultdict(list)
    for r in final:
        by_word[r["target_word"]].append(r["candidate"])
    with OUT_OK.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["char", "replacements"])
        for word, cands in sorted(by_word.items()):
            w.writerow([word, "|".join(cands)])
    print(f"\n[out] 可直接用 → {OUT_OK.relative_to(T)}  ({len(by_word)} 行)")

    # ---- 待确认表：把 A/C 级被剔的也列出来，便于抽查是否误杀
    review_rows = []
    for r in review:
        review_rows.append({
            "reason": "B级_音对但需判是否成词",
            "target_word": r["target_word"], "candidate": r["candidate"],
            "pinyin": r["pinyin"], "same_tone": r["same_tone"],
            "attested_in_corpus": r["attested_in_corpus"],
            "min_char_freq": r["min_char_freq"],
            "accept": "", "note": "",
        })
    # A/C 级里因生僻字被剔的
    for r in rows:
        if r["confidence"] in ("A_same_tone_attested", "C_diff_tone_attested") \
                and int(r["min_char_freq"]) < args.min_char_freq:
            review_rows.append({
                "reason": "A/C级但被判生僻字_请抽查",
                "target_word": r["target_word"], "candidate": r["candidate"],
                "pinyin": r["pinyin"], "same_tone": r["same_tone"],
                "attested_in_corpus": r["attested_in_corpus"],
                "min_char_freq": r["min_char_freq"],
                "accept": "", "note": "",
            })
    review_rows.sort(key=lambda r: (r["reason"], r["target_word"], r["candidate"]))

    with OUT_REVIEW.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "reason", "target_word", "candidate", "pinyin", "same_tone",
            "attested_in_corpus", "min_char_freq", "accept", "note"])
        w.writeheader()
        w.writerows(review_rows)
    print(f"[out] 待人工确认 → {OUT_REVIEW.relative_to(T)}  ({len(review_rows)} 行)")

    print("\n=== 可直接用的表（全部）===")
    for word, cands in sorted(by_word.items()):
        ev = {r["candidate"]: r["attested_in_corpus"]
              for r in final if r["target_word"] == word}
        detail = "  ".join(f"{c}(语料{ev[c]})" for c in cands)
        print(f"  {word:<8} → {detail}")

    print("\n=== 需人工确认的样例（前 20 条，看'能不能成词'）===")
    print(f"  {'目标词':<10}{'候选':<10}{'最小字频':<8}")
    print("  " + "-" * 30)
    for r in review[:20]:
        print(f"  {r['target_word']:<10}{r['candidate']:<10}{r['min_char_freq']:<8}")
    print(f"\n  待确认表共 {len(review_rows)} 行，分两类：")
    print("    B级_音对但需判是否成词  → 看'是不是个词'")
    print("    A/C级但被判生僻字_请抽查 → 抽查我有没有误杀")
    print("  在 accept 列填 y/n 即可。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
