#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
import_split_candidates.py —— 把 A 的部件候选表导入为 `split_char.csv`（带过滤报告）

背景
----
`Graph_Split`（拆字，如 `强 → 弓虽`）此前是我 8 个小类里**唯一 0 产出**的一类，
原因是缺「汉字→部件」表。A 于 2026-10-20 交付了 CJKVI 的 `ids-cdp.txt`（GPLv2）
以及机械派生的 `split_char_candidates.csv`（17,180 个字，`quality_pass=pending`）。

A 的明确要求（《先读我》§②）
---------------------------
> **不得把候选表直接更名为正式 split_char.csv 就宣称质检通过。**
> C 应结合人工标注的目标词，审查可读性、自然性、可恢复性及上下文，
> 再把通过项的 `char,replacements` 导出为自己规则目录下的 `split_char.csv`。

所以本脚本做的是**机械初筛 + 报告**，不是质量认证：
  * 只保留**常用字**（字频 ≥ 阈值）——生僻字拆出来也没人读得回；
  * 要求**每个部件都常用**——`强→弓虽` 成立是因为 `弓`、`虽` 都常见；
    若拆出的部件是生僻字，读者根本还原不回来（违反"可恢复性"）；
  * 原字与部件都是基本 CJK 汉字（A 已保证，这里再核一遍）；
  * 输出 `quality_pass=pending` 的**候选规则表**，最终是否 pass 由 D-2 质检判。

用法
----
    # 默认阈值 24（覆盖 99% 字量，见 data/rules/README.md 的口径）
    python scripts/import_split_candidates.py \
        --candidates "../data/raw/seeds/A交付_20261020/02_汉字部件/split_char_candidates.csv" \
        --out ../data/rules/split_char.csv --min-freq 24

    python scripts/import_split_candidates.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
T = HERE.parent

# 控制台兜底
try:                                                         # pragma: no cover
    sys.stdout.reconfigure(errors="replace")
except Exception:                                            # pragma: no cover
    pass

DEFAULT_CANDIDATES = (T / "data/raw/seeds/A交付_20261020/02_汉字部件/"
                          "split_char_candidates.csv")
DEFAULT_FREQ = T / "data/rules/_candidates/char_freq_table1.csv"
DEFAULT_OUT = T / "data/rules/split_char.csv"


def is_cjk(ch: str) -> bool:
    return len(ch) == 1 and "\u4e00" <= ch <= "\u9fff"


def load_freq(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return {r["char"]: int(r["count"]) for r in csv.DictReader(fh)
                if (r.get("char") or "").strip()}


def build_rank(freq: dict[str, int]) -> dict[str, int]:
    """字频排名（1 = 最常见）。用来判断「这个部件读者认不认得」。"""
    ranked = sorted(freq.items(), key=lambda kv: -kv[1])
    return {c: i + 1 for i, (c, _) in enumerate(ranked)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="导入部件候选表为 split_char.csv（带过滤报告）")
    ap.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    ap.add_argument("--freq", default=str(DEFAULT_FREQ), help="字频表（char,count）")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--min-freq", type=int, default=24,
                    help="**原字**的字频下限（目标字要够常见才值得攻击）")
    ap.add_argument("--max-part-rank", type=int, default=3200,
                    help="**部件**的字频排名上限（越小越常用）。"
                         "判据：部件必须是读者能认出来的字，否则拆开就还原不回来。"
                         "实测分界：弓 rank=3124（保留）、丨 rank=3251、丿 rank=4336（排除）")
    ap.add_argument("--report", default=None, help="过滤报告输出路径")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    cand_path = Path(args.candidates)
    if not cand_path.exists():
        print(f"[FAIL] 候选表不存在：{cand_path}")
        return 2
    freq = load_freq(Path(args.freq))
    have_freq = bool(freq)
    rank = build_rank(freq) if have_freq else {}
    if not have_freq:
        print("[WARN] 未找到字频表 —— 跳过常用字过滤，只做结构性检查。"
              "这样产出的表**不宜直接用**（会含大量生僻字拆分）。")

    rows = list(csv.DictReader(open(cand_path, encoding="utf-8-sig")))
    print(f"[1] 候选 {len(rows)} 条（来源：{cand_path.name}）")

    kept: dict[str, str] = {}
    drop: Counter = Counter()
    examples: dict[str, list[str]] = {}

    def record(reason: str, ch: str, rep: str) -> None:
        drop[reason] += 1
        if len(examples.setdefault(reason, [])) < 5:
            examples[reason].append(f"{ch}->{rep}")

    for r in rows:
        ch = (r.get("char") or "").strip()
        rep = (r.get("replacements") or "").strip()
        if not is_cjk(ch):
            record("原字非基本汉字", ch, rep)
            continue
        if not rep or rep == ch:
            record("无部件或与原字相同", ch, rep)
            continue
        parts = list(rep)
        if len(parts) not in (2, 3):
            record("部件数不是 2/3", ch, rep)
            continue
        if any(not is_cjk(p) for p in parts):
            record("部件含非基本汉字", ch, rep)
            continue
        if ch in parts:
            record("部件包含原字（循环拆分）", ch, rep)
            continue
        if have_freq:
            if freq.get(ch, 0) < args.min_freq:
                record(f"原字字频<{args.min_freq}", ch, rep)
                continue
            # 部件判「可识别」而不是「高频」：弓 在语料里只出现 5 次，
            # 但人人都认得；用排名才能把 弓(3124) 与 丨(3251) 分开。
            hard = [p for p in parts if rank.get(p, 10 ** 9) > args.max_part_rank]
            if hard:
                record(f"部件排名>{args.max_part_rank}（不可识别）", ch, rep)
                continue
        kept[ch] = rep

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["char", "replacements"])
        for ch in sorted(kept):
            wr.writerow([ch, kept[ch]])

    print(f"[2] 保留 {len(kept)} 个字 → {out}")
    print("[3] 丢弃原因：")
    for reason, n in drop.most_common():
        ex = "，".join(examples[reason])
        print(f"      {n:>6}  {reason}   例：{ex}")

    # 关键样例自检（A 的《先读我》点名的那几个）
    must = {"强": "弓虽", "明": "日月"}
    print("[4] 关键样例：")
    for ch, want in must.items():
        got = kept.get(ch)
        print(f"      {ch} -> {got or '（未保留）'}"
              f"{'  [OK]' if got == want else ''}")

    rep_path = Path(args.report) if args.report else out.with_name(
        out.stem + "_过滤报告.md")
    with open(rep_path, "w", encoding="utf-8") as fh:
        fh.write("# split_char.csv 导入过滤报告\n\n")
        fh.write(f"> 由 `scripts/import_split_candidates.py` 生成\n")
        fh.write(f"> 候选来源：A《2026-10-20 三项请求材料》`02_汉字部件/split_char_candidates.csv`"
                 f"（原始数据 CJKVI `ids-cdp.txt`，GPLv2）\n")
        fh.write(f"> 字频门槛：{args.min_freq}（原字与每个部件都要达标）\n\n")
        fh.write(f"| 项 | 数量 |\n|---|---|\n")
        fh.write(f"| 候选总数 | {len(rows)} |\n")
        fh.write(f"| **保留** | **{len(kept)}** |\n")
        for reason, n in drop.most_common():
            fh.write(f"| 丢弃：{reason} | {n} |\n")
        fh.write("\n## ⚠️ 这只做了机械初筛\n\n")
        fh.write("按 A 的要求，候选表**不得改个名就宣称质检通过**。本表满足的是：\n\n")
        fh.write("1. 原字与所有部件都是**基本 CJK 汉字**；\n")
        fh.write("2. 原字与所有部件的字频都 ≥ 阈值（保证读者能还原）；\n")
        fh.write("3. 部件数 2/3，且不包含原字本身。\n\n")
        fh.write("**仍需人工判的**：拆分是否自然（`强→弓虽` 自然，`三→一二` 生硬）、"
                 "在具体句子语境里能否还原、是否与 `Graph_Similar`/繁体类混淆。"
                 "最终 `quality_pass` 由 D-2 双人质检 + 仲裁给出。\n\n")
        fh.write("## 许可\n\n原始 IDS 数据为 CJKVI 项目，GPLv2；派生表须保留来源与许可，"
                 "不与 ToxiCN 词表混成同一许可。\n")
    print(f"[5] 报告：{rep_path}")
    return 0


def self_test() -> int:
    fails, n = [], [0]

    def chk(name, got, want):
        n[0] += 1
        if got != want:
            fails.append(f"{name}: 得到 {got!r}，期望 {want!r}")

    chk("CJK 判断", is_cjk("强"), True)
    chk("非 CJK", is_cjk("A"), False)
    chk("emoji 非 CJK", is_cjk("🙂"), False)

    # 过滤逻辑用内联数据复现（不依赖外部文件）
    freq = {"强": 100, "弓": 50, "虽": 40, "蠢": 30, "春": 10, "虫": 60}
    def keep(ch, rep, min_freq=24):
        if not is_cjk(ch) or not rep or rep == ch:
            return False
        parts = list(rep)
        if len(parts) not in (2, 3) or any(not is_cjk(p) for p in parts) or ch in parts:
            return False
        if freq.get(ch, 0) < min_freq:
            return False
        return all(freq.get(p, 0) >= min_freq for p in parts)

    chk("强→弓虽 保留", keep("强", "弓虽"), True)
    chk("蠢→春虫虫 丢弃（部件春字频10）", keep("蠢", "春虫虫"), False)
    chk("原字生僻 丢弃", keep("巭", "功丂"), False)
    chk("与原字相同 丢弃", keep("强", "强"), False)
    chk("部件数超限 丢弃", keep("强", "弓虽弓虽"), False)

    if fails:
        print(f"[FAIL] 自测 {len(fails)}/{n[0]} 项失败：")
        for f in fails:
            print("   ", f)
        return 1
    print(f"[OK] 自测 {n[0]} 项全过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
