#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_seed_leakage.py —— 交付前查重：确认表2 种子不在表1 里

依据
----
* V2 第 4 节：表5 只读；同一 source_id 不跨 split
* `docs/02_§5.4`：跨表查重**必须用归一化**（精确匹配会漏 `Sym_Separator` 类）
* A 实测：按来源排除时必须连带处理**跨来源孪生副本**

做三件事
--------
1. **种子的归一化哈希** 是否出现在 A 的 `表1_归一化哈希清单.csv` 里
   （命中 = 该种子被 DAPT 读过 = 泄漏）
2. **表5 排除清单** 是否命中（若已填）
3. 报告的哈希一律用与 Track A 一致的**五步**归一化（含 zhconv）

用法
----
    python scripts/verify_seed_leakage.py \
        --seeds out/seed_candidates.csv \
        --table1-hashes data/raw/seeds/A交付_2026-02-05/表1_归一化哈希清单.csv \
        --exclusion data/templates/table5_exclusion_list.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_schema import HAS_ZHCONV, norm, sha  # noqa: E402


def read_csv(p: Path):
    with p.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="表2 种子泄漏查重")
    ap.add_argument("--seeds", type=Path, required=True)
    ap.add_argument("--table1-hashes", type=Path,
                    help="A 交付的表1 归一化哈希清单")
    ap.add_argument("--exclusion", type=Path, help="表5 排除清单")
    args = ap.parse_args(argv)

    problems = 0

    if not HAS_ZHCONV:
        print("[FAIL] 缺 zhconv —— 归一化不完整，查重结论不可信。")
        print("       先跑 .\\install_zhconv.ps1")
        return 1

    if not args.seeds.exists():
        print(f"[ERR] 种子表不存在：{args.seeds}")
        return 2
    seeds = read_csv(args.seeds)
    print(f"[info] 种子 {len(seeds)} 条")

    # ---------------------------------------------------------- 1. 表1 交叉核对
    if args.table1_hashes:
        if not args.table1_hashes.exists():
            print(f"[ERR] 表1 哈希清单不存在：{args.table1_hashes}")
            return 2
        t1 = {r["normalized_hash"] for r in read_csv(args.table1_hashes)}
        print(f"[info] 表1 哈希 {len(t1)} 条")
        hit = [r for r in seeds if sha(norm(r["text"])) in t1]
        print(f"\n[check] 种子 ∩ 表1（归一化）：{len(hit)} 条")
        for r in hit[:10]:
            print(f"    ✗ {r['source_id']}  {r['text'][:40]}")
        if hit:
            print("    ❌ 命中的种子已被 DAPT 读过，必须剔除或与 A 核对")
            problems += len(hit)
        else:
            print("    ✅ 零命中 —— 无种子落入表1")
    else:
        print("\n[skip] 未提供表1 哈希清单，跳过交叉核对")

    # ---------------------------------------------------------- 2. 表5 排除清单
    if args.exclusion and args.exclusion.exists():
        ex = read_csv(args.exclusion)
        rows = [r for r in ex if (r.get("text") or "").strip()
                or (r.get("text_sha256") or "").strip()
                or (r.get("text_norm_sha256") or "").strip()]
        if not rows:
            print("\n[warn] 表5 排除清单为空 —— 这一步实际上没有生效")
        else:
            ex_exact = {r["text_sha256"] for r in rows if r.get("text_sha256")}
            ex_norm = {r["text_norm_sha256"] for r in rows if r.get("text_norm_sha256")}
            ex_text = {r["text"] for r in rows if r.get("text")}
            hit = [r for r in seeds
                   if sha(r["text"]) in ex_exact
                   or sha(norm(r["text"])) in ex_norm
                   or r["text"] in ex_text]
            print(f"\n[check] 种子 ∩ 表5：{len(hit)} 条")
            for r in hit[:10]:
                print(f"    ✗ {r['source_id']}  {r['text'][:40]}")
            if hit:
                problems += len(hit)
            else:
                print("    ✅ 零命中")
    else:
        print("\n[skip] 未提供表5 排除清单")

    # ---------------------------------------------------------- 3. 结构自检
    bad_split = [r for r in seeds if r.get("split") not in {"train", "dev", "test"}]
    ids = [r["source_id"] for r in seeds]
    dup = {i for i in ids if ids.count(i) > 1}
    spans_bad = []
    for r in seeds:
        try:
            s, e = int(r["target_span_start"]), int(r["target_span_end"])
        except (KeyError, ValueError):
            spans_bad.append(r["source_id"])
            continue
        if r["text"][s:e] != r["target_span_text"]:
            spans_bad.append(r["source_id"])

    print(f"\n[check] split 非法：{len(bad_split)} 条")
    print(f"[check] source_id 重复：{len(dup)} 处")
    print(f"[check] target_span 与原文不符：{len(spans_bad)} 条")
    for sid in spans_bad[:5]:
        print(f"    ✗ {sid}")
    problems += len(bad_split) + len(dup) + len(spans_bad)

    print("\n" + "=" * 50)
    if problems == 0:
        print("结论：✅ 全部通过，种子集可用于生成表2 变体。")
        return 0
    print(f"结论：❌ 共 {problems} 处问题，需修正后再交付。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
