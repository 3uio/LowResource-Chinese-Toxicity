#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_variants_rule.py —— Track C 规则型对抗变体生成器（骨架版 v0.1）

设计要点
--------
1. **只替换 target_span 内的字**，句子的其余部分原样保留。
   这是为什么种子表必须有 target_span_start / target_span_end 字段
   （见 docs/02_数据schema与source_id规则.md §1.1）。
2. 每条规则只产出一个 primary attack_type，天然满足
   docs/01_attack_type定义_v0.1.md §3 的"单 primary"要求。
3. **不猜测、不硬凑**：字符不在映射表里就跳过并计数，最后打印覆盖率报告。
   覆盖率低说明映射表不够，而不是样本"生成不出来"。
4. pypinyin 是**可选依赖**。未安装时 PY_Full / PY_Abbr 退化为内置极小映射表。

用法
----
    python scripts/gen_variants_rule.py \
        --seeds examples/demo_seeds.csv \
        --out out/variants.csv \
        --stats-out out/generation_stats.csv

    # 只生成部分类别
    python scripts/gen_variants_rule.py --seeds X.csv --out Y.csv \
        --attack-types Homo,Sym_Separator

    # 限定种子数量（前 N 条），用于小批量试跑
    python scripts/gen_variants_rule.py --seeds X.csv --out Y.csv --limit 20

输出
----
- --out         变体表（列与 data/templates/variants_template.csv 一致）
- --stats-out   每条规则的生成数 / 跳过数与跳过原因

注意
----
quality_pass 一律写 `pending`；semantic_score 留空。
**它们必须由人工质检与独立的语义打分步骤填写，本脚本不做判断。**
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------- 可选依赖
try:
    from pypinyin import lazy_pinyin  # type: ignore

    HAS_PYPINYIN = True
except ImportError:  # pragma: no cover
    HAS_PYPINYIN = False


# ---------------------------------------------------------------- 常量
FAMILY_OF = {
    "PY_Full": "phonetic",
    "PY_Abbr": "phonetic",
    "Homo": "phonetic",
    "Graph_Similar": "graphic",
    "Graph_Split": "graphic",
    "Sym_Leet": "symbolic",
    "Emoji_Sub": "symbolic",
    "Sym_Separator": "symbolic",
}

# 优先级：数字越小越优先（见定义文档 §3）
PRIORITY = [
    "Homo", "PY_Full", "PY_Abbr",
    "Graph_Similar", "Graph_Split",
    "Emoji_Sub", "Sym_Leet", "Sym_Separator",
]

# ---------------------------------------------------------------- 内置占位映射表
# !! 这些只是"能让脚本跑起来"的极小占位表。
# !! 正式数据必须换成经过人工筛选的映射表（见 README「下一步」）。
# !! 映射表文件格式：CSV，两列 char,replacements（多项用 | 分隔）

PLACEHOLDER_PINYIN = {
    "傻": "sha", "逼": "bi", "蠢": "chun", "货": "huo", "滚": "gun",
    "开": "kai", "恶": "e", "心": "xin", "废": "fei", "物": "wu",
    "猪": "zhu", "子": "zi", "讨": "tao", "厌": "yan",
}

PLACEHOLDER_HOMOPHONE = {
    "傻": ["沙", "煞"],
    "子": ["紫"],
    "蠢": ["春"],
    "货": ["祸"],
    "滚": ["棍"],
    "恶": ["饿"],
    "心": ["新"],
    "废": ["肺"],
    "物": ["务"],
    "猪": ["珠"],
}

# 形近字 / 异体字。注意：是否允许使用异体字与生僻字是定义文档 §5 的开放问题，
# 这里放进去只为演示，评审后可能被禁用。
PLACEHOLDER_SIMILAR = {
    "货": ["贷"],
    "废": ["廃"],
    "恶": ["悪"],
    "猪": ["豬"],
}

# 部件拆分（真·部件，不是符号）
PLACEHOLDER_SPLIT = {
    "蠢": "春虫虫",
    "货": "化贝",
    "物": "牛勿",
    "猪": "犭者",
}

# 数字 / 拉丁字母混写
PLACEHOLDER_LEET = {
    "逼": ["B", "13"],
    "货": ["H"],
    "子": ["2"],
}

# 表情符号替换：emoji 必须与目标字**谐音**（定义文档 §2.7）
PLACEHOLDER_EMOJI = {
    "逼": ["🍺"],
}

SEPARATORS = ["*", ".", " ", "—", "\u200b"]  # 最后一个是零宽空格


# ---------------------------------------------------------------- 映射表加载
def load_map(path: Path | None) -> dict[str, list[str]]:
    """读取 char,replacements 两列的映射表；多项用 | 分隔。"""
    if path is None:
        return {}
    if not path.exists():
        print(f"[warn] 映射表不存在，忽略：{path}", file=sys.stderr)
        return {}
    out: dict[str, list[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            ch = (row.get("char") or "").strip()
            repl = (row.get("replacements") or "").strip()
            if ch and repl:
                out[ch].extend(x for x in repl.split("|") if x)
    return dict(out)


def load_str_map(path: Path | None) -> dict[str, str]:
    """读取 char,replacements 两列的映射表，每个字只取第一个替换项，值为字符串。

    拼音表必须用这个加载器：值必须是 str，不能是 list，
    否则 rule_py_full / rule_py_abbr 会拿到 list 而崩掉（或静默产出错结果）。
    """
    if path is None or not path.exists():
        return {}
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            ch = (row.get("char") or "").strip()
            repl = (row.get("replacements") or "").strip()
            if ch and repl:
                out[ch] = repl.split("|")[0]
    return out


def load_split_map(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            ch = (row.get("char") or "").strip()
            repl = (row.get("replacements") or "").strip()
            if ch and repl:
                out[ch] = repl.split("|")[0]
    return out


def load_word_map(path: Path | None) -> dict[str, list[str]]:
    """整词级映射表（键是**词**，不是单字）。

    文件格式与 load_map 相同（`char,replacements`），只是 `char` 列填的是词。
    列名保持 `char` 是为了复用同一套表结构。

    依据 `docs/07_替换单位决策记录.md`：整词替换为主，字级兜底。
    """
    if path is None or not path.exists():
        return {}
    out: dict[str, list[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            word = (row.get("char") or "").strip()
            repl = (row.get("replacements") or "").strip()
            if word and repl:
                out[word].extend(x.strip() for x in repl.split("|") if x.strip())
    return dict(out)


# ---------------------------------------------------------------- 规则实现
# 每个规则接收 (span, rng, maps) 返回候选字符串列表（可能为空列表）

def rule_py_full(span, rng, maps):
    if HAS_PYPINYIN:
        parts = lazy_pinyin(span)
    else:
        parts = [maps["pinyin"].get(c) for c in span]
    if any(p is None or p == "" for p in parts):
        return []
    return [("".join(parts), "char"), (" ".join(parts), "char")]


def rule_py_abbr(span, rng, maps):
    if HAS_PYPINYIN:
        parts = lazy_pinyin(span)
    else:
        parts = [maps["pinyin"].get(c) for c in span]
    if any(p is None or p == "" for p in parts):
        return []
    return [("".join(p[0] for p in parts), "char")]


def _charwise(span, rng, table, joiner="", allow_partial=False):
    """逐字查表替换。

    allow_partial=False（默认，严格模式）
        任一字查不到就整体放弃。适合**谐音/形近**这类"整词都要换"的规则，
        避免半替换产出伪样本。

    allow_partial=True（部分替换）
        只替换表中有的字，其余原样保留。适合**emoji / leet** 这类
        "一个符号换一个字"的规则 —— 例如 `母狗` 里只有 `狗` 有 emoji 时，
        应产出 `母🐶`，而不是因为 `母` 不在表里就整体放弃。

    ⚠️ 这个区别是实测逼出来的：v1 只有严格模式，导致 `Emoji_Sub`
       在 200 条种子上只产出 5 条（正确应为数十条）。
    """
    if allow_partial:
        results = []
        for i in range(2):          # 最多产出 2 条，避免组合爆炸
            chars = []
            for ch in span:
                cands = table.get(ch)
                if not cands:
                    chars.append(ch)                     # 原样保留
                else:
                    chars.append(cands[min(i, len(cands) - 1)])
            out = joiner.join(chars)
            if out and out != span and out not in results:
                results.append(out)
        return results

    outs = []
    for ch in span:
        cands = table.get(ch)
        if not cands:
            return []
        outs.append(cands)
    # 取笛卡尔积会爆炸，这里只做"每个位置取第 idx 个"，idx 由 rng 决定
    n = min(len(c) for c in outs)
    results = []
    for i in range(min(n, 2)):  # 每个位置最多取 2 种组合，避免组合爆炸
        pick = rng.randrange(len(outs[0])) if len(outs[0]) > 1 else 0
        results.append(joiner.join(c[min(pick + i, len(c) - 1)] for c in outs))
    return results


def rule_homo(span, rng, maps):
    """同音／近音替换 —— **整词优先，字级兜底**。

    依据 `docs/07_替换单位决策记录.md`（组长 2026-02-05 批准）：
    实测 98% 的目标词是多字的，而逐字替换对多字目标词会产出伪样本
    （`母狗` 逐字换 → `目狗`）。真实网络用法是整词级的（`母狗` → `姆狗`）。

    所以：
        ① 先查整词表 `homophone_word.csv`（键为目标词，如 `母狗`）
        ② 未命中再查字级表 `homophone.csv`（键为单字，逐字替换）

    返回值统一为 [(候选, 单位)]，单位取值 `word` / `char`，写入
    `substitution_unit` 字段供"字级 vs 词级"消融分析。
    """
    hits = maps["homo_word"].get(span)
    if hits:
        return [(h, "word") for h in hits]
    return [(c, "char") for c in _charwise(span, rng, maps["homo"])]


def _word_first(span, rng, word_map, char_table, allow_partial=False):
    """整词优先、字级兜底的通用替换器（与 rule_homo 同机制）。

    依据 docs/07_替换单位决策记录.md：逐字替换对多字目标词会产出伪样本
    （`母狗` 逐字换 → `目狗`），真实用法是整词级的。
    所以所有"替换型"规则都先查词表，未命中再逐字兜底。

    allow_partial 透传给 _charwise：emoji/leet 允许部分替换（`母狗`→`母🐶`），
    谐音/形近不允许（避免 `母够` 这类半替换废品）。
    """
    hits = word_map.get(span)
    if hits:
        return [(h, "word") for h in hits]
    return [(c, "char")
            for c in _charwise(span, rng, char_table, allow_partial=allow_partial)]


def rule_graph_similar(span, rng, maps):
    """形近替换：整词（繁体/异体）优先，字级兜底。"""
    return _word_first(span, rng, maps["similar_word"], maps["similar"])


def rule_graph_split(span, rng, maps):
    table = maps["split"]
    if any(ch not in table for ch in span):
        return []
    return [("".join(table[ch] for ch in span), "char")]


def rule_sym_leet(span, rng, maps):
    """数字/字母混写：整词优先，字级兜底。允许部分替换（`傻13` 只换一部分）。"""
    return _word_first(span, rng, maps["leet_word"], maps["leet"],
                       allow_partial=True)


def rule_emoji_sub(span, rng, maps):
    """emoji 替换：整词优先，字级兜底。允许部分替换（`母狗` → `母🐶`）。"""
    return _word_first(span, rng, maps["emoji_word"], maps["emoji"],
                       allow_partial=True)


def rule_sym_separator(span, rng, maps):
    if len(span) < 2:
        return []
    return [(sep.join(span), "char") for sep in SEPARATORS]


RULES = {
    "PY_Full": rule_py_full,
    "PY_Abbr": rule_py_abbr,
    "Homo": rule_homo,
    "Graph_Similar": rule_graph_similar,
    "Graph_Split": rule_graph_split,
    "Sym_Leet": rule_sym_leet,
    "Emoji_Sub": rule_emoji_sub,
    "Sym_Separator": rule_sym_separator,
}


# ---------------------------------------------------------------- 主流程
OUT_COLS = [
    "source_id", "variant_id", "text", "attack_family", "attack_type",
    "secondary_attack_types", "is_adversarial", "generation_method",
    "generation_rule_id", "recoverability", "semantic_score_raw",
    "semantic_score_normalized", "quality_pass", "qc_reviewer_1",
    "qc_reviewer_2", "qc_arbiter", "split", "substitution_unit",
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Track C 规则型对抗变体生成器")
    ap.add_argument("--seeds", required=True, type=Path)
    ap.add_argument("--out", default=Path("out/variants.csv"), type=Path)
    ap.add_argument("--stats-out", default=Path("out/generation_stats.csv"), type=Path)
    ap.add_argument("--attack-types", default=",".join(PRIORITY))
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条种子（0=全部）")
    ap.add_argument("--seed", type=int, default=20260101, help="随机种子，保证可复现")
    ap.add_argument("--rules-dir", default=None, type=Path,
                    help="映射表目录，含 pinyin.csv/homophone.csv/similar_char.csv/"
                         "split_char.csv/leet.csv/emoji.csv")
    ap.add_argument("--per-type-quota", default="Sym_Separator=1",
                    help="按 attack_type 限制**每种子**最多产出几条，"
                         "用逗号分隔的 type=N，如 "
                         "'Sym_Separator=1,PY_Full=2'。为空则不限。"
                         "用途：防止「最容易生成的类」把变体总量撑爆 —— "
                         "实测 Sym_Separator 不设限时每种子产 5 条，"
                         "占全部变体的一半，会让攻击类型分布严重偏斜。")
    args = ap.parse_args(argv)

    quota = {}
    if args.per_type_quota.strip():
        for part in args.per_type_quota.split(","):
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            k = k.strip()
            try:
                quota[k] = int(v.strip())
            except ValueError:
                print(f"[error] --per-type-quota 格式错误：{part}", file=sys.stderr)
                return 2
        bad_q = [k for k in quota if k not in RULES]
        if bad_q:
            print(f"[error] --per-type-quota 里有未知 attack_type：{bad_q}",
                  file=sys.stderr)
            return 2

    rng = random.Random(args.seed)

    rd = args.rules_dir
    maps = {
        "pinyin": load_str_map(rd / "pinyin.csv") if rd else {},
        "homo": load_map(rd / "homophone.csv") if rd else {},
        "homo_word": load_word_map(rd / "homophone_word.csv") if rd else {},
        "similar": load_map(rd / "similar_char.csv") if rd else {},
        "similar_word": load_word_map(rd / "similar_word.csv") if rd else {},
        "split": load_split_map(rd / "split_char.csv") if rd else {},
        "leet": load_map(rd / "leet.csv") if rd else {},
        "leet_word": load_word_map(rd / "leet_word.csv") if rd else {},
        "emoji": load_map(rd / "emoji.csv") if rd else {},
        "emoji_word": load_word_map(rd / "emoji_word.csv") if rd else {},
    }
    for k in ("homo_word", "similar_word", "leet_word", "emoji_word"):
        if maps[k]:
            print(f"[info] 整词表 {k}: {len(maps[k])} 个目标词")
    # 内置占位表兜底
    for k, v in PLACEHOLDER_PINYIN.items():          # 拼音表：str -> str
        maps["pinyin"].setdefault(k, v)
    for key, fallback in (
        ("homo", PLACEHOLDER_HOMOPHONE),
        ("similar", PLACEHOLDER_SIMILAR),
        ("leet", PLACEHOLDER_LEET),
        ("emoji", PLACEHOLDER_EMOJI),
    ):
        for k, v in fallback.items():
            maps[key].setdefault(k, v if isinstance(v, list) else [v])
    for k, v in PLACEHOLDER_SPLIT.items():
        maps["split"].setdefault(k, v)

    enabled = [t.strip() for t in args.attack_types.split(",") if t.strip()]
    bad = [t for t in enabled if t not in RULES]
    if bad:
        print(f"[error] 未知 attack_type：{bad}", file=sys.stderr)
        print(f"        可选：{list(RULES)}", file=sys.stderr)
        return 2

    if not args.seeds.exists():
        print(f"[error] 种子文件不存在：{args.seeds}", file=sys.stderr)
        return 2

    # ---- 读取种子
    seeds = []
    with args.seeds.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"source_id", "text", "target_span_start", "target_span_end"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            print(f"[error] 种子表缺少列：{sorted(missing)}", file=sys.stderr)
            return 2
        for i, row in enumerate(reader):
            if args.limit and i >= args.limit:
                break
            seeds.append(row)

    print(f"[info] 读入种子 {len(seeds)} 条；启用规则 {len(enabled)} 个")
    if quota:
        print("[info] 每种子配额：" +
              "，".join(f"{k}≤{v}" for k, v in quota.items()))
    if not HAS_PYPINYIN:
        print("[warn] 未安装 pypinyin，PY_Full / PY_Abbr 仅对内置占位表的字生效。")
        print("       安装：python -m pip install pypinyin")

    rows = []
    stats: Counter = Counter()
    skip_reason: Counter = Counter()
    coverage: dict[str, Counter] = defaultdict(Counter)
    seen_texts: set[str] = set()
    counter: dict[tuple[str, str], int] = defaultdict(int)

    for seed in seeds:
        sid = seed["source_id"].strip()
        text = seed["text"]
        try:
            s = int(seed["target_span_start"])
            e = int(seed["target_span_end"])
        except (TypeError, ValueError):
            stats["种子跳过:span非法"] += 1
            continue
        span = text[s:e]
        declared = (seed.get("target_span_text") or "").strip()
        if declared and declared != span:
            print(f"[warn] {sid}: target_span_text='{declared}' 与切片 '{span}' 不一致，"
                  f"以偏移为准", file=sys.stderr)
        if not span:
            stats["种子跳过:span为空"] += 1
            continue

        for at in enabled:
            coverage[at]["种子数"] += 1
            try:
                cands = RULES[at](span, rng, maps)
            except Exception as exc:  # 单条规则失败不应中断整批
                stats[f"规则异常:{at}"] += 1
                skip_reason[f"{at}:异常 {exc!r}"] += 1
                continue

            if not cands:
                stats[f"空产出:{at}"] += 1
                skip_reason[f"{at}:映射表缺字（span={span}）"] += 1
                coverage[at]["空产出"] += 1
                continue

            produced = 0
            cap = quota.get(at)          # 该类的每种子配额（None = 不限）
            for item in cands:
                if cap is not None and produced >= cap:
                    skip_reason[f"{at}:达每种子配额({cap})"] += 1
                    continue
                # 规则统一返回 (候选文本, 替换单位)；单位写入 substitution_unit
                cand, unit = item if isinstance(item, tuple) else (item, "")
                cand = cand.strip()
                if not cand or cand == span:
                    skip_reason[f"{at}:与原词相同"] += 1
                    continue
                new_text = text[:s] + cand + text[e:]
                if new_text in seen_texts:
                    skip_reason[f"{at}:批内重复"] += 1
                    continue
                seen_texts.add(new_text)
                counter[(sid, at)] += 1
                rows.append({
                    "source_id": sid,
                    "variant_id": f"r-{sid}-{at}-{counter[(sid, at)]:02d}",
                    "text": new_text,
                    "attack_family": FAMILY_OF[at],
                    "attack_type": at,
                    "secondary_attack_types": "",
                    "is_adversarial": 1,
                    "generation_method": "rule",
                    "generation_rule_id": f"{at.upper()}-001",
                    "recoverability": "",          # 人工/后续填写
                    "semantic_score_raw": "",      # 待语义打分
                    "semantic_score_normalized": "",
                    "quality_pass": "pending",     # 必须人工质检
                    "qc_reviewer_1": "", "qc_reviewer_2": "", "qc_arbiter": "",
                    "split": seed.get("split", ""),
                    "substitution_unit": unit,     # char / word（见 docs/07）
                })
                produced += 1
                coverage[at][f"单位:{unit}"] += 1
            coverage[at]["产出"] += produced
            stats[f"产出:{at}"] += produced

    # ---- 写出
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=OUT_COLS)
        w.writeheader()
        w.writerows(rows)

    args.stats_out.parent.mkdir(parents=True, exist_ok=True)
    with args.stats_out.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["attack_type", "family", "种子输入数", "产出变体数", "空产出种子数", "覆盖率"])
        for at in enabled:
            c = coverage[at]
            n_in, n_out, n_empty = c["种子数"], c["产出"], c["空产出"]
            rate = f"{n_out / n_in:.2f} 变体/种子" if n_in else "-"
            w.writerow([at, FAMILY_OF[at], n_in, n_out, n_empty, rate])

    # ---- 报告
    print(f"\n[ok] 写出变体 {len(rows)} 条 → {args.out}")
    print(f"[ok] 统计表 → {args.stats_out}\n")
    print("按类别：")
    for at in enabled:
        extra = ""
        units = {k.split(":", 1)[1]: v for k, v in coverage[at].items()
                 if k.startswith("单位:")}
        if units:
            extra = "  单位 " + " / ".join(f"{k}:{v}" for k, v in sorted(units.items()))
        print(f"  {at:<16} {stats[f'产出:{at}']:>5} 条  "
              f"(空产出种子 {coverage[at]['空产出']}){extra}")
    if skip_reason:
        print("\n跳过原因 Top10：")
        for reason, n in skip_reason.most_common(10):
            print(f"  {n:>5}  {reason}")
    print("\n[next] 下一步：填写 semantic_score → 人工双人质检 → quality_pass"
          "（见 docs/02_数据schema与source_id规则.md §7）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
