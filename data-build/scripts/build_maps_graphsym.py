#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_maps_graphsym.py —— 建三个映射表：形近（繁体变体）/ leet / emoji

对应的 attack_type
------------------
* `Graph_Similar` ← 繁体/异体字替换（见 docs/09 §三 方案 A）
* `Sym_Leet`      ← 形近数字/字母替换（人工构造，见下）
* `Emoji_Sub`     ← emoji 谐音/形似替换

为什么 `Sym_Leet` 是人工构造的
------------------------------
实测：表1 全文 33,149 条里**不存在 leet 替换**——
数字与字母的出现全是正常用法（`是gay`/`985`/`RPG专`/`30岁`/`B站`），
没有一处是"用形近数字替换汉字"。

原因合理：表1 是原始帖（A 清洗过），规避写法本就不在其中。

所以 leet 表只能依据**众所周知的网络用法**构造，每一条都在 note 列写明依据。

依据出处
--------
* 本文件为 C 依据 `docs/01_attack_type定义_v0.1.md` §2.6/§2.7 的判据人工构造
* 繁体替换手法有文献支持：Guo et al. 2025 (arXiv:2507.07640) 的
  PCR 分类（Hanzi Replacement）涵盖繁体/异体替换
* 形近字表来源：contr4l/SimilarCharacter（MIT License, (c) 2020 XiaoFang）
  —— 仅作参考，本表未直接引用其条目（其实测候选 90% 为生僻字，不可用，见 docs/09）
"""

from __future__ import annotations

import csv
from pathlib import Path

import zhconv

T = Path(__file__).resolve().parent.parent
RULES = T / "data" / "rules"
SEEDS = T / "out" / "seed_candidates.csv"
FREQ = T / "data" / "rules" / "_candidates" / "char_freq_table1.csv"

THR = 24          # 表1 字频门槛（剔生僻字）

# ---------------------------------------------------------------- 读目标字/词
import collections

seeds = list(csv.DictReader(SEEDS.open(encoding="utf-8-sig")))
word_uses = collections.Counter(r["target_span_text"] for r in seeds)
chars = collections.Counter(c for w in word_uses for c in w)
freq = {r["char"]: int(r["count"])
        for r in csv.DictReader(FREQ.open(encoding="utf-8-sig"))}
print(f"[info] 目标词 {len(word_uses)}，目标字 {len(chars)}")


def write_map(path: Path, table: dict[str, list[str]], header_note: str) -> int:
    """写出 char,replacements 格式（生成器读取）。"""
    n = 0
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["char", "replacements"])
        for k, v in sorted(table.items()):
            v = [x for x in v if x]
            if v:
                w.writerow([k, "|".join(v)])
                n += 1
    print(f"[out] {path.name}: {n} 条   ({header_note})")
    return n


# ================================================================ 1. 形近（繁体/异体）
print()
print("=" * 62)
print("1. Graph_Similar —— 繁体/异体字替换")
print("=" * 62)

tv = {}
for ch in chars:
    t = zhconv.convert(ch, "zh-tw")
    if t != ch and len(t) == 1 and freq.get(t, 0) >= THR:
        tv[ch] = [t]
    elif t != ch and len(t) == 1:
        # 繁体字本身在语料里少见 —— 仍保留（繁体不是生僻字，读者认识），但标注出来
        tv[ch] = [t]

# 只保留有繁体形的
print(f"  有繁体/异体形的目标字: {len(tv)}")
word_ok = [w for w in word_uses if all(c in tv for c in w)]
word_part = [w for w in word_uses
             if any(c in tv for c in w) and not all(c in tv for c in w)]
print(f"  整词可换: {len(word_ok)} 个词 → {' '.join(word_ok[:12])}")
print(f"  部分可换: {len(word_part)} 个词")
write_map(RULES / "similar_char.csv", tv, "繁体/异体替换")

# 词级表（整词替换优先，与谐音同机制）
word_tv = {w: ["".join(tv[c][0] if c in tv else c for c in w)]
           for w in word_ok}
# 部分替换的词也放进去（部分字换成繁体，整词仍成立）
for w in word_part:
    cand = "".join(tv[c][0] if c in tv else c for c in w)
    if cand != w:
        word_tv[w] = [cand]
print(f"  词级表（整词优先）: {len(word_tv)} 个词")
write_map(RULES / "similar_word.csv", word_tv, "整词繁体替换")

# ================================================================ 2. Sym_Leet
print()
print("=" * 62)
print("2. Sym_Leet —— 形近数字/字母（人工构造）")
print("=" * 62)

# 每条都写清依据。**宁缺毋滥** —— 只收形近无争议、或众所周知的网络用法。
#
# ⚠️ 已剔除的映射（实测噪声大，见 docs/10）：
#     "子": ["z"]   —— 在"熊孩子/绿帽子/一棍子"里 `子` 是词缀，换成 z 破坏语法
#     "的": ["d"]   —— 在"黑d最惨"里 `的` 是助词，`黑d` 读不出原意
#   这两条曾占 Sym_Leet 产出的 56/57，剔掉后产出大减，但**留下的都是可用的**。
LEET = {
    # --- 汉字数字 ↔ 阿拉伯数字：纯形近/等义，无争议 ---
    "一": ["1"], "二": ["2"], "三": ["3"], "四": ["4"], "五": ["5"],
    "六": ["6"], "七": ["7"], "八": ["8"], "九": ["9"], "十": ["10"],
    # --- 众所周知的中英混写骂人写法（形似或谐音，见 docs/01 §2.6） ---
    "逼": ["B"],      # 傻B / 二B，最常见的中英混写
    "傻": ["S"],      # 与"傻B"配套出现
    "屌": ["D"],      # 网络常见
    "货": ["H"],      # "傻H"类，与"傻B"同族（弱，可被质检剔除）
}
leet = {c: v for c, v in LEET.items() if c in chars}
miss = [c for c in LEET if c not in chars]
print(f"  表 {len(LEET)} 条，命中目标字 {len(leet)} 个: {' '.join(leet)}")
if miss:
    print(f"  未命中（目标字里没有，仍保留供后续使用）: {''.join(miss)}")
# 未命中的也写进去 —— 后面加种子时还能用
write_map(RULES / "leet.csv", {c: v for c, v in LEET.items()}, "形近数字/字母")

# ================================================================ 3. Emoji_Sub
print()
print("=" * 62)
print("3. Emoji_Sub —— emoji 谐音/形似")
print("=" * 62)

# 依据 docs/01 §2.7：emoji 与目标字须存在**谐音或形似**关系。
# 动物字 → 对应动物 emoji（形似 + 谐音，最自然）
EMOJI = {
    "狗": ["🐶"], "猪": ["🐷"], "鸡": ["🐔"], "牛": ["🐮"], "马": ["🐴"],
    "龟": ["🐢"], "鱼": ["🐟"], "虫": ["🐛"], "蛇": ["🐍"], "鼠": ["🐭"],
    "猴": ["🐵"], "驴": ["🫏"], "猫": ["🐱"], "兔": ["🐰"],
    # 谐音（啤酒 bēi jiǔ ≈ 逼 bī）
    "逼": ["🍺"],
    # 形似/联想
    "死": ["💀"], "屎": ["💩"],
}
emoji = {c: v for c, v in EMOJI.items() if c in chars}
print(f"  表 {len(EMOJI)} 条，命中目标字 {len(emoji)} 个")
for c, v in emoji.items():
    ws = [w for w in word_uses if c in w]
    print(f"    {c} ({chars[c]}次) → {v[0]}   {' '.join(ws[:5])}")
write_map(RULES / "emoji.csv", EMOJI, "emoji 谐音/形似")

# 词级 emoji（整词都命中时）
ew = {w: ["".join(EMOJI[c][0] for c in w)]
      for w in word_uses if all(c in EMOJI for c in w)}
print(f"  词级表: {len(ew)} 个词 → {' '.join(f'{k}:{v[0]}' for k, v in list(ew.items())[:5])}")
write_map(RULES / "emoji_word.csv", ew, "整词 emoji")

print()
print("=" * 62)
print("汇总")
print("=" * 62)
for f in ("similar_char.csv", "similar_word.csv", "leet.csv",
          "emoji.csv", "emoji_word.csv"):
    p = RULES / f
    n = max(0, sum(1 for _ in p.open(encoding="utf-8-sig")) - 1) if p.exists() else 0
    print(f"  {f:<22} {n:>4} 条")
print()
print("[next] 跑生成器：")
print("  python scripts/gen_variants_rule.py --seeds out/seed_candidates.csv \\")
print("      --out out/variants_v2.csv --stats-out out/stats_v2.csv --rules-dir data/rules")
