#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pilot_log.py —— C-P0-3 手工变体试跑的**记录、计时与统计**工具（v0.1）

目的
----
V2 绿灯项要求"小规模标注试跑（30-50 条），实测工时"。这一步有三个用途
（见 `Track C 执行方案` §C-P0-3）：
    ① 实测单条工时  → 校正 C 线的 90-134h 总工时估算
    ② 暴露规则难点  → 找出 8 小类定义里说不清的地方（`docs/01` 的开放问题）
    ③ 验证 schema   → 确认 `docs/02` 的字段够不够用

本脚本**不生成变体**（手工试跑的价值就在"人做一遍"），只负责：
    记录 → 校验 → 计时 → 统计 → 出报告

为什么要有 `blocker` 标记
-------------------------
试跑最重要的产出不是 30 条变体，而是**"哪条规则的定义不够用"**。用
`--blocker` 标出来，最后 `report` 会专门列一节，直接作为 `docs/01` 评审会
的输入。没有这个标记，卡点信息会散在对话里丢掉。

⚠️ 工时是**手工填写**的，脚本不做自动计时
----------------------------------------
自动计时会把"我去接了杯水"算成工时。实测值要用于估算 90-134h，宁可用
手填的整数分钟。建议用手机秒表，做完一条记一次。

用法
----
    # 1) 初始化（会创建 pilot_log.csv、占位种子表与试跑变体表）
    python scripts/pilot_log.py init

    # 2) 记一条（手工做完一条变体后立刻记）
    python scripts/pilot_log.py add --seed trcn-000001 --attack-type Homo `
        --variant "你这个沙比真讨厌" --minutes 4 --difficulty 2 `
        --rule HOMO-001 --note "沙比是真实网络用法，可复现"

    # 3) 卡点标记（定义说不清的时候，标出来而不是硬凑）
    python scripts/pilot_log.py add --seed trcn-000002 --attack-type Emoji_Sub `
        --variant "蠢🐷一个" --minutes 9 --difficulty 5 `
        --blocker "emoji 与目标字只有形似没有谐音，docs/01 §2.7 判据说必须谐音，判不了"

    # 4) 看进度
    python scripts/pilot_log.py stats

    # 5) 出报告（markdown，直接贴评审会）
    python scripts/pilot_log.py report --out out/pilot_report.md

退出码
------
0 = 成功；1 = 校验失败（数据没写进去）；2 = 用法错误
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

# 与 docs/01_attack_type定义_v0.1.md §1 保持一致的 8 小类
FAMILY_OF = {
    "PY_Full": "phonetic", "PY_Abbr": "phonetic", "Homo": "phonetic",
    "Graph_Similar": "graphic", "Graph_Split": "graphic",
    "Sym_Leet": "symbolic", "Emoji_Sub": "symbolic", "Sym_Separator": "symbolic",
}
ATTACK_TYPES = list(FAMILY_OF)
DEFAULT_LOG = Path("data/templates/pilot_log.csv")
DEFAULT_VARIANTS = Path("out/pilot_variants.csv")
DEFAULT_SEEDS = Path("examples/pilot_targets.csv")

LOG_COLS = [
    "date", "operator", "seed_source_id", "seed_text", "target_span_text",
    "attack_type", "attack_family", "variant_text", "minutes_spent",
    "difficulty_1to5", "rule_candidate", "blocker", "note", "seed_not_found",
]

# 与 validate_schema.py 的 VARIANT_COLS 对齐（末尾多一个 is_pilot 用于隔离试跑产物）
VARIANT_COLS = [
    "source_id", "variant_id", "text", "attack_family", "attack_type",
    "secondary_attack_types", "is_adversarial", "generation_method",
    "generation_rule_id", "recoverability", "semantic_score_raw",
    "semantic_score_normalized", "quality_pass", "qc_reviewer_1",
    "qc_reviewer_2", "qc_arbiter", "split", "is_pilot",
]

# 试跑用占位种子：**全部是虚构的**，只用于测工时，不可作任何实验数据
# 末位是 split：给非空值是为了让 validate_schema.py 能跑通（它会校验 split 合法性）
PILOT_TARGETS = [
    ("trcn-000001", "你这个傻子真讨厌", 3, 5, "傻子", "train"),
    ("trcn-000002", "蠢货一个，别说话了", 0, 2, "蠢货", "train"),
    ("trcn-000003", "滚开吧你", 0, 2, "滚开", "dev"),
    ("trcn-000004", "这个人真恶心", 4, 6, "恶心", "train"),
    ("trcn-000005", "他就是个废物", 4, 6, "废物", "test"),
    ("trcn-000006", "你这个傻逼真让人无语", 3, 5, "傻逼", "train"),
    ("trcn-000007", "别在这儿丢人现眼了", 4, 6, "丢人", "train"),
    ("trcn-000008", "一群脑残粉丝", 2, 4, "脑残", "train"),
    ("trcn-000009", "闭嘴吧你", 0, 2, "闭嘴", "dev"),
    ("trcn-000010", "真是个二百五", 3, 6, "二百五", "train"),
    ("trcn-000011", "你脑子有病吧", 1, 5, "脑子有病", "train"),
    ("trcn-000012", "这种垃圾言论少说", 2, 4, "垃圾", "test"),
]
SEED_COLS = ["source_id", "text", "target_span_start", "target_span_end",
             "target_span_text", "toxicity_label", "source_dataset",
             "orig_file", "orig_line", "split", "dup_of"]


# --------------------------------------------------------------------------
# 读写
# --------------------------------------------------------------------------

def read_csv(path: Path) -> list:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, cols, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def load_seeds(path: Path) -> dict:
    """返回 {source_id: 行}。文件不存在时返回空 dict（调用方自行决定是否报错）。"""
    return {r["source_id"]: r for r in read_csv(path) if r.get("source_id")}


# --------------------------------------------------------------------------
# init
# --------------------------------------------------------------------------

def cmd_init(args) -> int:
    made = []

    if not args.log.exists() or args.force:
        write_csv(args.log, LOG_COLS, [])
        made.append(str(args.log))
    else:
        print(f"[跳过] 已存在：{args.log}")

    if not args.seeds_file.exists() or args.force:
        rows = [{
            "source_id": sid, "text": t,
            "target_span_start": s, "target_span_end": e,
            "target_span_text": span, "toxicity_label": 1,
            "source_dataset": "PILOT_PLACEHOLDER",
            "orig_file": "pilot_placeholder.txt",
            "orig_line": i + 1, "split": sp, "dup_of": "",
        } for i, (sid, t, s, e, span, sp) in enumerate(PILOT_TARGETS)]
        write_csv(args.seeds_file, SEED_COLS, rows)
        made.append(str(args.seeds_file))
    else:
        print(f"[跳过] 已存在：{args.seeds_file}")

    if not args.variants.exists() or args.force:
        write_csv(args.variants, VARIANT_COLS, [])
        made.append(str(args.variants))
    else:
        print(f"[跳过] 已存在：{args.variants}")

    print("\n已创建：")
    for m in made:
        print(f"  + {m}")

    print(
        "\n⚠️ 提醒："
        f"\n   {args.seeds_file} 里是**虚构的占位种子**，只用于测工时，"
        "\n   不得作为表2 的任何输入，也不得进入任何实验。"
        "\n   真实种子要等 A 的预留清单。"
    )
    print("\n下一步：")
    print("  python scripts/pilot_log.py add --seed trcn-000001 --attack-type Homo "
          "--variant \"你这个沙比真讨厌\" --minutes 4 --difficulty 2")
    return 0


# --------------------------------------------------------------------------
# add
# --------------------------------------------------------------------------

def cmd_add(args) -> int:
    errors = []

    # --- 字段校验 ---------------------------------------------------------
    if args.attack_type not in FAMILY_OF:
        errors.append(
            f"attack_type 非法：'{args.attack_type}'。"
            f"只接受 8 小类：{' / '.join(ATTACK_TYPES)}"
        )
    if not (0.5 <= args.minutes <= 120):
        errors.append(f"--minutes 应在 0.5–120 之间，收到 {args.minutes}")
    if not (1 <= args.difficulty <= 5):
        errors.append(f"--difficulty 应在 1–5 之间，收到 {args.difficulty}")

    seed = None
    if args.seed:
        seeds = load_seeds(args.seeds_file)
        seed = seeds.get(args.seed)
        if args.seeds_file.exists() and seed is None:
            errors.append(f"种子表里没有 source_id='{args.seed}'（{args.seeds_file}）")
        if not args.seeds_file.exists():
            print(f"[提示] 种子表 {args.seeds_file} 不存在，跳过种子关联校验")
    else:
        print("[警告] 没有给 --seed：这条记录无法与种子关联，只能算工时")

    if errors:
        print("❌ 校验失败，本条**未写入**：")
        for e in errors:
            print(f"   - {e}")
        return 1

    # --- 组装记录 ---------------------------------------------------------
    if seed:
        seed_text = seed.get("text", "")
        span = seed.get("target_span_text", "")
        if span and span not in seed_text:
            print(f"[警告] 种子表自检异常：target_span_text='{span}' 不在 text 里")
        if args.variant.strip() == seed_text.strip():
            print("[提示] 变体与原句完全相同——这条是基线，不是对抗变体（仍会记下）")
    else:
        seed_text, span = "", ""

    split = seed.get("split", "") if seed else ""
    fam = FAMILY_OF.get(args.attack_type, "")
    seq = _next_seq(args.variants, args.seed or "noseed", args.attack_type)
    variant_id = f"r-{args.seed or 'noseed'}-{args.attack_type}-{seq:02d}"

    log_row = {
        "date": args.date or date.today().isoformat(),
        "operator": args.operator,
        "seed_source_id": args.seed or "",
        "seed_text": seed_text,
        "target_span_text": span,
        "attack_type": args.attack_type,
        "attack_family": fam,
        "variant_text": args.variant,
        "minutes_spent": _fmt_minutes(args.minutes),
        "difficulty_1to5": args.difficulty,
        "rule_candidate": args.rule or "",
        "blocker": args.blocker or "",
        "note": args.note or "",
        "seed_not_found": "0",
    }

    rows = [r for r in read_csv(args.log) if r.get("date")]
    rows.append(log_row)
    write_csv(args.log, LOG_COLS, rows)

    # --- 同步写入 schema 形状的变体表（供后续 validate_schema.py 试跑） ----
    var_row = {
        "source_id": args.seed or "",
        "variant_id": variant_id,
        "text": args.variant,
        "attack_family": fam,
        "attack_type": args.attack_type,
        "secondary_attack_types": args.secondary or "",
        "is_adversarial": 1,
        "generation_method": "manual",       # ← 手工试跑，区别于 rule / llm
        "generation_rule_id": args.rule or "PILOT-MANUAL",
        "recoverability": args.recoverability or "",
        "semantic_score_raw": "",             # 未与 D 对齐前一律留空
        "semantic_score_normalized": "",
        "quality_pass": "pending",            # 与生成器一致：不自动判定
        "qc_reviewer_1": "", "qc_reviewer_2": "", "qc_arbiter": "",
        "split": split,
        "is_pilot": 1,
    }
    vrows = [r for r in read_csv(args.variants) if r.get("variant_id")]
    vrows.append(var_row)
    write_csv(args.variants, VARIANT_COLS, vrows)

    total = len(rows)
    print(f"✅ 已记录第 {total} 条：{args.seed or '(无种子)'} / {args.attack_type} "
          f"/ {_fmt_minutes(args.minutes)} 分钟 / 难度 {args.difficulty}")
    print(f"   variant_id = {variant_id}")
    if args.blocker:
        print(f"   ⚠️ 卡点已标记：{args.blocker}")
    if total < 20:
        print(f"   进度：{total}/20（最低目标）　剩余 {20 - total} 条")
    elif total < 50:
        print(f"   进度：{total}/50（理想目标）　剩余 {50 - total} 条")
    else:
        print("   进度：已达 30–50 条目标，可以出报告了 → "
              "python scripts/pilot_log.py report")
    return 0


def _next_seq(variants_path: Path, source_id: str, attack_type: str) -> int:
    """同一 (source_id, attack_type) 内从 01 递增（对齐 docs/02 §3.2）。"""
    n = 0
    for r in read_csv(variants_path):
        if r.get("source_id") == source_id and r.get("attack_type") == attack_type:
            n += 1
    return n + 1


def _fmt_minutes(m: float) -> str:
    return str(int(m)) if float(m).is_integer() else f"{m:g}"


# --------------------------------------------------------------------------
# stats / report
# --------------------------------------------------------------------------

def _collect(args):
    rows = [r for r in read_csv(args.log) if r.get("date")]
    mins = [float(r["minutes_spent"]) for r in rows if r.get("minutes_spent")]
    diffs = [int(r["difficulty_1to5"]) for r in rows
             if str(r.get("difficulty_1to5", "")).isdigit()]

    by_type = defaultdict(list)
    for r in rows:
        if r.get("minutes_spent"):
            by_type[r.get("attack_type", "?")].append(float(r["minutes_spent"]))

    blockers = [r for r in rows if (r.get("blocker") or "").strip()]
    metrics = {
        "count": len(rows),
        "total_minutes": sum(mins),
        "mean_minutes": statistics.mean(mins) if mins else 0.0,
        "median_minutes": statistics.median(mins) if mins else 0.0,
        "min_minutes": min(mins) if mins else 0.0,
        "max_minutes": max(mins) if mins else 0.0,
        "mean_difficulty": statistics.mean(diffs) if diffs else 0.0,
        "by_type": {k: {"n": len(v), "mean": statistics.mean(v)}
                    for k, v in sorted(by_type.items())},
        "missing_types": [t for t in ATTACK_TYPES if t not in by_type],
        "blocker_count": len(blockers),
        "notes_count": len([r for r in rows if (r.get("note") or "").strip()]),
    }
    return rows, metrics


def cmd_stats(args) -> int:
    rows, m = _collect(args)
    if not rows:
        print(f"还没有任何记录（{args.log}）。先跑 init 再用 add 记一条。")
        return 0

    print(f"=== 试跑进度（{args.log}）===")
    print(f"已记录条目        ：{m['count']} 条")
    print(f"累计工时          ：{m['total_minutes'] / 60:.2f} h"
          f"（{m['total_minutes']:g} 分钟）")
    print(f"单条工时 均值/中位：{m['mean_minutes']:.1f} / {m['median_minutes']:.1f} 分钟"
          f"（区间 {m['min_minutes']:g}–{m['max_minutes']:g}）")
    print(f"平均难度（1–5）   ：{m['mean_difficulty']:.2f}")
    print(f"卡点条数          ：{m['blocker_count']}")
    print()

    print("--- 按 attack_type ---")
    print(f"{'attack_type':<16}{'条数':>5}{'平均工时':>10}")
    for t in ATTACK_TYPES:
        v = m["by_type"].get(t)
        if v:
            print(f"{t:<16}{v['n']:>5}{v['mean']:>9.1f}m")
        else:
            print(f"{t:<16}{'-':>5}{'-':>10}   ← 未覆盖")
    print()

    if m["missing_types"]:
        print(f"⚠️ 还没覆盖的小类（{len(m['missing_types'])}/8）："
              f"{' / '.join(m['missing_types'])}")
        print("   8 小类都要至少手写 2–3 条，否则测不出类间工时差异，")
        print("   也验证不了定义在每类上是否够用。")
    else:
        print("✅ 8 小类已全覆盖")

    if m["blocker_count"]:
        print(f"\n⚠️ 有 {m['blocker_count']} 条标了卡点 → 出报告时会被单列，"
              "直接作为 docs/01 评审会的输入。")
    return 0


def cmd_report(args) -> int:
    rows, m = _collect(args)
    if not rows:
        print("没有记录，无法出报告。")
        return 1

    L = []
    L.append("# C-P0-3 手工变体试跑报告（实测工时）")
    L.append("")
    L.append(f"> 生成时间：{date.today().isoformat()}　记录条数：**{m['count']}**"
             f"　数据源：`{args.log}`")
    L.append("> 用途：V2 绿灯项「小规模标注试跑（30-50 条），实测工时」的交付证据；")
    L.append("> 同时作为 `docs/01_attack_type定义_v0.1.md` 评审会的输入。")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 1. 工时实测（用于校正 C 线的 90–134h 估算）")
    L.append("")
    L.append("| 指标 | 实测值 |")
    L.append("|---|---|")
    L.append(f"| 记录条数 | {m['count']} 条 |")
    L.append(f"| 累计工时 | {m['total_minutes'] / 60:.2f} h |")
    L.append(f"| 单条工时 均值 | **{m['mean_minutes']:.1f} 分钟** |")
    L.append(f"| 单条工时 中位数 | {m['median_minutes']:.1f} 分钟 |")
    L.append(f"| 单条工时 区间 | {m['min_minutes']:g} – {m['max_minutes']:g} 分钟 |")
    L.append(f"| 平均难度（1–5） | {m['mean_difficulty']:.2f} |")
    L.append("")
    L.append("### 按 attack_type 拆分")
    L.append("")
    L.append("| attack_type | 条数 | 平均工时 | 备注 |")
    L.append("|---|---|---|---|")
    for t in ATTACK_TYPES:
        v = m["by_type"].get(t)
        if v:
            L.append(f"| `{t}` | {v['n']} | {v['mean']:.1f} 分钟 | |")
        else:
            L.append(f"| `{t}` | 0 | — | ⚠️ **本次未覆盖** |")
    L.append("")

    if m["mean_minutes"] > 0:
        lo = m["mean_minutes"] * 800 / 60
        hi = m["mean_minutes"] * 1000 / 60
        L.append("### 外推：批量生成 800–1000 条的手工成本")
        L.append("")
        L.append(f"按实测均值 {m['mean_minutes']:.1f} 分钟/条外推：")
        L.append("")
        L.append(f"- 800 条 ≈ **{lo:.0f} h**")
        L.append(f"- 1000 条 ≈ **{hi:.0f} h**")
        L.append("")
        L.append("> ⚠️ 这个数字**不能直接当成工时预算**：批量阶段由脚本生成、"
                 "人工只做质检，与『从零手写』不是同一件事。")
        L.append("> 它的正确用途是**反证『纯手工不可行』**——如果外推值远超 "
                 "`Track C 执行方案` §7 估的 20–30h（脚本开发与调优），")
        L.append("> 就说明规则脚本是必需的，不是可选的。")
        L.append("")

    L.append("---")
    L.append("")
    L.append(f"## 2. 规则难点与卡点（{m['blocker_count']} 条）")
    L.append("")
    if m["blocker_count"] == 0:
        L.append("本次试跑**没有标记任何卡点**。")
        L.append("")
        L.append("> ⚠️ 这本身值得怀疑：8 小类的定义是 v0.1 草案，正常应该会在"
                 "边界案例上暴露问题。")
        L.append("> 请回看是否有『当时觉得说不清、但随手凑过去了』的情况——"
                 "那正是需要标出来的。")
    else:
        L.append("每条都是 `docs/01` 评审会必须解决的**具体**问题（不是抽象讨论）：")
        L.append("")
        for i, r in enumerate([x for x in rows if (x.get("blocker") or "").strip()], 1):
            L.append(f"**B{i}. `{r['attack_type']}`**（种子 `{r['seed_source_id']}`，"
                     f"耗时 {r['minutes_spent']} 分钟，难度 {r['difficulty_1to5']}/5）")
            L.append("")
            L.append(f"- 原文：`{r['seed_text']}`")
            L.append(f"- 变体：`{r['variant_text']}`")
            L.append(f"- 卡点：**{r['blocker']}**")
            if (r.get("rule_candidate") or "").strip():
                L.append(f"- 候选规则：`{r['rule_candidate']}`")
            L.append("")
    L.append("---")
    L.append("")
    L.append("## 3. schema 验证结论")
    L.append("")
    L.append("试跑同时用于验证 `docs/02` 的字段是否够用。请逐条确认：")
    L.append("")
    L.append("| 字段 | 试跑中是否真的用到了 | 结论 |")
    L.append("|---|---|---|")
    L.append("| `target_span_start/end/text` | 是（每条都要定位目标词） | "
             "确认**必需**，需补进 V2 第 4 节 |")
    L.append("| `recoverability` | 试跑中是否判过？ | 待确认 |")
    L.append("| `secondary_attack_types` | 是否有复合攻击？ | 待确认 |")
    L.append("| `generation_rule_id` | 是（`rule_candidate` 列） | 确认必需 |")
    L.append("| `blocker` / 难度 / 工时 | 是（本次新增） | "
             "**建议作为试跑专用列，不进正式 schema** |")
    L.append("")
    L.append("> `semantic_score_raw/normalized` 在本次试跑中**一律留空**——"
             "口径尚未与 D 对齐（见 `对接文档/C给D/C给D_对接确认单（接口约定）.md` D-1）。")
    L.append("")

    if m["notes_count"]:
        L.append("---")
        L.append("")
        L.append(f"## 4. 观察记录（{m['notes_count']} 条 note）")
        L.append("")
        for r in rows:
            note = (r.get("note") or "").strip()
            if note:
                L.append(f"- `{r['seed_source_id']}` / `{r['attack_type']}`：{note}")
        L.append("")

    L.append("---")
    L.append("")
    L.append("## 5. 下一步")
    L.append("")
    L.append("1. 把 §2 的卡点逐条带上 `docs/01` 评审会 → 冻结为 v1.0（解除边界①）")
    L.append("2. 把 §1 的实测工时回填 `Track C 执行方案` §7 的估算表")
    L.append("3. 把 §3 的 schema 结论回填 `docs/02`")
    L.append("4. ⚠️ 试跑产物 `out/pilot_variants.csv` **带 `is_pilot=1` 标记**，"
             "**不得混入正式表2**")
    L.append("")

    text = "\n".join(L)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"✅ 报告已写入：{args.out}")
    else:
        print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(m, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        print(f"✅ 指标 JSON 已写入：{args.json}")
    return 0


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="C-P0-3 手工变体试跑：记录 / 计时 / 统计 / 出报告",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--log", type=Path, default=DEFAULT_LOG,
                       help=f"试跑日志（默认 {DEFAULT_LOG}）")
        p.add_argument("--variants", type=Path, default=DEFAULT_VARIANTS,
                       help=f"试跑变体表（默认 {DEFAULT_VARIANTS}）")
        p.add_argument("--seeds-file", type=Path, default=DEFAULT_SEEDS,
                       help=f"种子表（默认 {DEFAULT_SEEDS}）")

    p = sub.add_parser("init", help="初始化日志、占位种子表与试跑变体表")
    common(p)
    p.add_argument("--force", action="store_true", help="覆盖已存在的文件")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("add", help="记录一条手工变体")
    common(p)
    p.add_argument("--seed", help="种子 source_id（如 trcn-000001）")
    p.add_argument("--attack-type", required=True, choices=ATTACK_TYPES)
    p.add_argument("--variant", required=True, help="变体文本")
    p.add_argument("--minutes", type=float, required=True,
                   help="本条实际耗时（分钟，手工秒表计时，不要用自动计时）")
    p.add_argument("--difficulty", type=int, required=True, choices=[1, 2, 3, 4, 5])
    p.add_argument("--operator", default="C", help="操作人（默认 C）")
    p.add_argument("--date", help="日期 YYYY-MM-DD（默认今天）")
    p.add_argument("--rule", help="候选规则 ID（如 HOMO-001）")
    p.add_argument("--blocker", help="⚠️ 定义说不清的地方——这是试跑最有价值的产出")
    p.add_argument("--note", help="观察记录")
    p.add_argument("--secondary", help="secondary_attack_types（分号分隔）")
    p.add_argument("--recoverability", choices=["recoverable", "unrecoverable"],
                   help="可恢复性（可选）")
    p.set_defaults(func=cmd_add)

    p = sub.add_parser("stats", help="看进度与工时统计")
    common(p)
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("report", help="出 markdown 报告")
    common(p)
    p.add_argument("--out", type=Path, help="输出路径（默认打印到屏幕）")
    p.add_argument("--json", type=Path, help="同时输出指标 JSON")
    p.set_defaults(func=cmd_report)

    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
