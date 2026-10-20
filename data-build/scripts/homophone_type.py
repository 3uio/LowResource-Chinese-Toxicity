#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
homophone_type.py —— 谐音变体的同音程度判定（D 要求）

依据
----
Guo et al. 2025, *Lost in Pronunciation: Detecting Chinese Offensive Language
Disguised by Phonetic Cloaking Replacement*（EMNLP 2025 Industry Track,
arXiv:2507.07640）的 PCR 分类：

  * **完美同音** perfect homophone：声母、韵母、**声调全部相同**
  * **近似同音** near-homophone：声调偏移，或声母/韵母是**声学相邻音**
      - 声调偏移：`sǐ → sì`
      - 声学相邻：`z → zh`、`n → l`（汉语方言常见混淆）
  * 原文指出：**真实用户以 near-homophone 为主，而这是现有审核模型的主要盲区**

本模块把该分类落地成可复现的判据，供 `gen_variants_rule.py` 与
`validate_schema.py` 共用（**只此一份实现，不许各写一遍**）。

三档取值
--------
| 值 | 含义 | 判据 |
|---|---|---|
| `perfect` | 完美同音 | 每个替换字的声母+韵母+声调全同 |
| `near` | 近似同音 | 声调偏移，或声母/韵母为声学相邻音 |
| `far` | 非谐音 | 有一处声母或韵母差异较大 |

`far` 的变体不应计入 `Homo` 类（读者读不回原词，伪装不成立）。
"""

from __future__ import annotations

import re

# 控制台兜底（本脚本未 import sys，用局部导入）：Windows GBK 控制台遇到 emoji 会崩，
# 统一降级为"打不出的字符显示成 ?"。
try:                                                     # pragma: no cover
    import sys as _sys
    _sys.stdout.reconfigure(errors="replace")
except Exception:                                        # pragma: no cover
    pass

try:
    from pypinyin import Style, pinyin
    HAS_PYPINYIN = True
except ImportError:                                   # pragma: no cover
    HAS_PYPINYIN = False

# ---------------------------------------------------------------- 声学相邻音
# 声母：按发音部位/方式相近分组，组内视为"声学相邻"
#   平舌/翘舌（z/zh, c/ch, s/sh）、鼻音/边音（n/l）、唇齿/舌根（f/h）、
#   送气/不送气（b/p, d/t, g/k）、j/q/x 与 z/c/s
INITIAL_GROUPS = [
    frozenset("zzh"), frozenset("cch"), frozenset("ssh"),
    frozenset("nl"), frozenset("rl"),
    frozenset("fh"),
    frozenset("bp"), frozenset("dt"), frozenset("gk"),
    frozenset("jqx"), frozenset("zcs"),
    frozenset("m"), frozenset("ng"), frozenset("yw"),
]
# 韵母：前后鼻音、o/e、ui/ei 等常见混淆
FINAL_GROUPS = [
    frozenset({"an", "ang"}), frozenset({"en", "eng"}),
    frozenset({"in", "ing"}), frozenset({"ian", "iang"}),
    frozenset({"uan", "uang"}), frozenset({"un", "ong"}),
    frozenset({"o", "e", "ou", "uo"}), frozenset({"i", "u", "ü", "v"}),
    frozenset({"ie", "ue"}), frozenset({"ai", "ei"}), frozenset({"ua", "a"}),
]

_INITIALS = ("zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l",
             "g", "k", "h", "j", "q", "x", "r", "z", "c", "s", "y", "w")


def parse_char(ch: str):
    """把一个汉字拆成 (声母, 韵母, 声调)。

    返回 None 表示取不到拼音（非汉字、生僻字等）。
    声调：1–4，轻声记 5；取不到时记 '?'。
    """
    if not HAS_PYPINYIN:
        return None
    try:
        t = pinyin(ch, style=Style.TONE3, errors="ignore")
    except Exception:
        return None
    if not t or not t[0]:
        return None
    s = t[0][0]
    m = re.match(r"^([a-zü]+?)(\d?)$", s)
    if not m:
        return None
    body, tone = m.group(1), (m.group(2) or "5")
    ini = ""
    for cand in _INITIALS:
        if body.startswith(cand):
            ini = cand
            break
    return ini, body[len(ini):], tone


def _adjacent(a: str, b: str, groups) -> bool:
    if a == b:
        return True
    return any(a in g and b in g for g in groups)


def classify_char(a: str, b: str) -> str:
    """单字对的同音程度：perfect / near / far。"""
    if a == b:
        return "perfect"
    pa, pb = parse_char(a), parse_char(b)
    if not pa or not pb:
        return "far"
    ia, fa, ta = pa
    ib, fb, tb = pb
    if ia == ib and fa == fb:
        return "perfect" if ta == tb else "near"   # 声母韵母同、声调不同 → near
    if ia == ib and _adjacent(fa, fb, FINAL_GROUPS):
        return "near"                              # 韵母声学相邻
    if fa == fb and _adjacent(ia, ib, INITIAL_GROUPS):
        return "near"                              # 声母声学相邻
    return "far"


def classify(original: str, candidate: str) -> str:
    """整词对的同音程度。

    取**最差**的那一对作整词判定：只要有一处 far，整词就是 far
    （读者读不回原词，伪装不成立）。
    """
    if not original or not candidate:
        return "far"
    if len(original) != len(candidate):
        return "far"
    kinds = [classify_char(a, b) for a, b in zip(original, candidate)]
    if "far" in kinds:
        return "far"
    if all(k == "perfect" for k in kinds):
        return "perfect"
    return "near"


# ---------------------------------------------------------------- 自检样例
SAMPLES = [
    # (原文, 变体, 期望, 说明)
    ("狗", "苟", "perfect", "gǒu/gǒu 声母韵母声调全同"),
    ("狗", "够", "near", "gǒu/gòu 声调偏移（3→4）—— 完美同音要求声调也同"),
    ("狗", "沟", "near", "gǒu/gōu 声调偏移（3→1）"),
    ("女", "钕", "perfect", "nǚ/nǚ 完美同音（但字生僻，另一道闸会拦）"),
    ("强", "墙", "perfect", "qiáng/qiáng 完美同音"),
    ("强", "抢", "near", "qiáng/qiǎng 声调偏移（2→3）"),
    ("黑", "嘿", "perfect", "hēi/hēi 完美同音"),
    ("河南", "和南", "perfect", "hé nán / hé nán 逐字完美同音"),
    ("河南", "河男", "perfect", "同上"),
    ("河南", "荷兰", "near", "hé nán / hé lán 声母 n→l 声学相邻"),
    ("黑", "恩", "far", "hēi / ēn 声母韵母都不同"),
    ("地域", "地育", "perfect", "yù/yù 完美同音"),
    # 整词替换的两个例子：一个 near、一个 perfect
    ("母狗", "幕狗", "near", "mǔ(3) gǒu(3) / mù(4) gǒu(3) —— 母→幕 声调偏移"),
    ("母狗", "姆狗", "perfect", "mǔ(3) gǒu(3) / mǔ(3) gǒu(3) —— 姆与母同音同调"),
]


def selftest() -> int:
    print("=" * 66)
    print("homophone_type 自检（判据见模块 docstring）")
    print("=" * 66)
    bad = 0
    for orig, cand, want, note in SAMPLES:
        got = classify(orig, cand)
        ok = got == want
        bad += (not ok)
        print(f"  {'OK' if ok else '!!'}  {orig} → {cand:<6} "
              f"判 {got:<8} 期望 {want:<8} {note}")
    print()
    print(f"  {len(SAMPLES) - bad}/{len(SAMPLES)} 通过"
          if bad == 0 else f"  ❌ {bad} 条不符")
    return bad


if __name__ == "__main__":
    raise SystemExit(1 if selftest() else 0)
