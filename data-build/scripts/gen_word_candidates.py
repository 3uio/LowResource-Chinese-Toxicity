#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_word_candidates.py —— 整词级同音候选生成（v3：分级）

依据
----
* `docs/07_替换单位决策记录.md`（组长已批准：**整词为主 + 字级兜底**）
* `docs/01_attack_type定义_v0.1.md` §2.3：谐音变体要"贴近真实网络用法，
  不要用生僻同音字硬凑"

版本演进（每一版都是被实测逼出来的）
------------------------------------
v1  只从语料里**挖**同音片段 → 157 词只有 41 个有候选。覆盖率太低。
v2  改成"生成 + 验证"，但要求**声调完全一致** → 只有 27 词拿到 high/medium。
v3  实测定位到两个根因，都不是"表不够大"：
    (a) **声调是主要约束**，不是字频。同一字频门槛下，只放宽声调：38% → 97%。
    (b) **真实谐音变体只替换目标词的一部分字**（`女拳` → `女权` 只换了 `拳`）。
        v2 要求"每个字都替换"，于是 `女拳` 只能生成 `钕权`/`籹全` 这类生僻字组合，
        真正的 `女权` 反而生成不出来 —— 因为 `女` 被排除在它自己的候选池外了。
    故 v3 改为：**每位可选「保留原字」或「换成同音节字」**，并用"最小改动"打分。

置信度分级
----------
| 级别 | 含义 | 用途 |
|---|---|---|
| `A_same_tone_attested` | 声调全同 **且** 语料里真出现过该写法 | 最接近真实网络用法 |
| `B_same_tone` | 声调全同，语料未出现 | 标准同音词 |
| `C_diff_tone_attested` | 声调不同但语料出现过 | 真实口语谐音 |
| `D_diff_tone` | 仅声母韵母相同 | 弱谐音，慎用 |

**"常用"判据**：用语料自身（2,041 条真实 ToxiCN 文本）统计字符频次，
精确回答"这个字在**我们的数据域**里真实出现过吗"。

输出
----
`data/rules/_candidates/homophone_word_candidates.csv`
⚠️ **是候选表，不是成品表**（全部 `need_review=YES`）。筛选标准见 `data/rules/README.md`。

用法
----
    python scripts/gen_word_candidates.py
"""

from __future__ import annotations

import csv
import itertools
import sys
from collections import Counter, defaultdict
from pathlib import Path

T = Path(__file__).resolve().parent.parent          # TrackC/
POOL = T / "data" / "raw" / "seeds" / "A交付_2026-02-05" / "表2种子预留清单.csv"
SEEDS = T / "out" / "seed_candidates.csv"
OUTDIR = T / "data" / "rules" / "_candidates"

MAX_CAND_PER_WORD = 8
MIN_CHAR_FREQ = 2           # 候选字的语料最低频次
MAX_ALT_PER_POS = 4         # 每个位置最多几个同音节替代字（不含"保留原字"）

try:
    from pypinyin import pinyin, Style
except ImportError:
    print("[FAIL] 需要 pypinyin：python -m pip install pypinyin")
    raise SystemExit(1)


def syl(text: str, tone: bool = False) -> list[str]:
    return [x[0] for x in pinyin(text,
                                 style=Style.TONE3 if tone else Style.NORMAL,
                                 errors="ignore")]


CONF_ORDER = {"A_same_tone_attested": 0, "B_same_tone": 1,
              "C_diff_tone_attested": 2, "D_diff_tone": 3}


def main() -> int:
    # ---------------------------------------------------------- 语料频次 + 证据
    pool = list(csv.DictReader(POOL.open(encoding="utf-8-sig")))
    freq = Counter()
    for r in pool:
        for ch in r["text"]:
            if "\u4e00" <= ch <= "\u9fff":
                freq[ch] += 1
    common = {ch for ch, n in freq.items() if n >= MIN_CHAR_FREQ}
    print(f"[info] 语料 {len(pool)} 条 → {len(freq)} 个汉字，"
          f"常用(≥{MIN_CHAR_FREQ}次) {len(common)} 个")

    # 不带调索引（主）与带调索引（用于判声调）
    idx_plain: dict[str, set[str]] = defaultdict(set)
    tone_of: dict[str, str] = {}
    for ch in common:
        p = syl(ch)
        t = syl(ch, tone=True)
        if p and t:
            idx_plain[p[0]].add(ch)
            tone_of[ch] = t[0]
    print(f"[info] 不带调音节 {len(idx_plain)} 个，平均 "
          f"{sum(len(v) for v in idx_plain.values())/len(idx_plain):.1f} 字/音节")

    # 语料连续片段（作为"真实出现过"的证据）
    evidence: Counter = Counter()
    for r in pool:
        run, runs = [], []
        for ch in r["text"]:
            if "\u4e00" <= ch <= "\u9fff":
                run.append(ch)
            else:
                if run:
                    runs.append("".join(run))
                run = []
        if run:
            runs.append("".join(run))
        for seg in runs:
            for n in (2, 3, 4):
                for i in range(len(seg) - n + 1):
                    evidence[seg[i:i + n]] += 1

    # ---------------------------------------------------------- 目标词
    seeds = list(csv.DictReader(SEEDS.open(encoding="utf-8-sig")))
    words = Counter(r["target_span_text"] for r in seeds)
    print(f"[info] 目标词 {len(words)} 个，覆盖 {sum(words.values())} 条种子")

    rows, stat = [], Counter()
    words_with = set()
    for word, _uses in words.most_common():
        if not all("\u4e00" <= c <= "\u9fff" for c in word):
            stat["跳过:非纯汉字目标"] += 1
            continue
        wp = syl(word)
        if len(wp) != len(word):
            stat["跳过:音节切分异常"] += 1
            continue

        # ---- 生成：每位可选「保留原字」或「换成同音节常用字」
        # 这是 v3 的关键修正：真实谐音变体常常只换一部分字（女拳 → 女权）。
        # 若强制每位都换，`女` 的候选池被排除自身后只剩生僻字，反而生成不出 女权。
        pools = []
        for s, c in zip(wp, word):
            alts = sorted(idx_plain.get(s, set()) - {c},
                          key=lambda x: -freq.get(x, 0))[:MAX_ALT_PER_POS]
            pools.append([c] + alts)        # [原字, alt1, alt2, ...]

        scored = []
        for combo in itertools.product(*pools):
            cand = "".join(combo)
            if cand == word:
                continue
            # 至少要改一个字，否则不是变体
            changed = sum(1 for a, b in zip(cand, word) if a != b)
            if changed == 0:
                continue
            same_tone = all(tone_of.get(a) == tone_of.get(b)
                            for a, b in zip(cand, word))
            ev = evidence.get(cand, 0)
            if same_tone and ev >= 1:
                conf = "A_same_tone_attested"
            elif same_tone:
                conf = "B_same_tone"
            elif ev >= 1:
                conf = "C_diff_tone_attested"
            else:
                conf = "D_diff_tone"
            minf = min(freq.get(c, 0) for c in cand)
            # 打分优先级：置信度 → 改动越少越好 → 语料证据越多越好 → 字越常见越好
            scored.append((CONF_ORDER[conf], changed, -ev, -minf, cand,
                           same_tone, ev, minf, conf))

        if not scored:
            stat["无候选"] += 1
            continue
        scored.sort(key=lambda x: x[:5])
        for _o, _ch, _e, _m, cand, same_tone, ev, minf, conf in scored[:MAX_CAND_PER_WORD]:
            rows.append({
                "target_word": word, "pinyin": " ".join(wp),
                "candidate": cand, "candidate_pinyin": " ".join(syl(cand)),
                "same_tone": "Y" if same_tone else "N",
                "attested_in_corpus": ev,
                "min_char_freq": minf, "confidence": conf,
                "need_review": "YES", "note": "",
            })
            stat[f"候选:{conf}"] += 1
        words_with.add(word)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / "homophone_word_candidates.csv"
    cols = ["target_word", "pinyin", "candidate", "candidate_pinyin",
            "same_tone", "attested_in_corpus", "min_char_freq",
            "confidence", "need_review", "note"]
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    pure = sum(1 for w in words if all("\u4e00" <= c <= "\u9fff" for c in w))
    print(f"\n[ok] 候选表 → {out}")
    for k in ["跳过:非纯汉字目标", "跳过:音节切分异常", "无同音字可选", "无候选"]:
        if stat[k]:
            print(f"       {k:<20} {stat[k]}")
    print(f"       有候选的词            {len(words_with)}/{pure}")
    for k in sorted(CONF_ORDER, key=lambda x: CONF_ORDER[x]):
        if stat[f"候选:{k}"]:
            print(f"       候选:{k:<24} {stat[f'候选:{k}']}")

    print("\n=== A 级候选（声调全同 + 语料出现过）===")
    a = [r for r in rows if r["confidence"] == "A_same_tone_attested"]
    a.sort(key=lambda r: -r["attested_in_corpus"])
    for r in a[:25]:
        print(f"  {r['target_word']:<8} → {r['candidate']:<8} "
              f"[{r['pinyin']}]  语料 {r['attested_in_corpus']:>3} 次")

    print("\n⚠️ 候选表，不是成品表（全部 need_review=YES）。")
    print("   筛选标准待写在 data/rules/README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
