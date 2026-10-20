#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seed_annotation.py —— 种子候选分层、标注表、标注校验、双人一致性（C-P0-1 / Q2 / D-2）

依据
----
* `docs/02_数据schema与source_id规则.md` §1（种子表字段）、§3.1（source_id 规则）
* `docs/11_20261011材料核对与整改清单.md` §5 Q2（`target_span` 判据待评审）
* D-2 决议：双人独立质检 + 仲裁，**须报告一致性**

为什么需要这个脚本
------------------
A 的隔离修复后，候选池 3,137 条的 `toxic_entity` **全为空**，
`mech_flags` 全部是 `needs_target_span` —— 目标词必须 100% 人工标。
手工挑 + 手工标 200 条会出三类问题：挑得不分层（全挑最长的或全挑最短的）、
span 偏移对不上原文、两个人标完无法算一致性。这个脚本把四步固化成流程。

子命令
------
    stratify   候选池 → 分层短名单（**不落 `target_span`**，Q2 未定也能先做）
    sheet      短名单 → 标注表（每个标注者一份，互相看不见，顺序一致）
    check      标注结果合法性校验（span 必须能对上原文）
    kappa      双人独立标注 → Cohen's κ + 分歧清单（供仲裁）

用法
----
    # 1) 分层短名单（默认只取 ≤20 字，配额按分层比例分配）
    python scripts/seed_annotation.py stratify \
        --pool "../data/raw/seeds/A交付_20261010/表2种子预留清单.csv" \
        --max-len 20 --target 200

    # 2) 生成两份互不可见的标注表
    python scripts/seed_annotation.py sheet --shortlist out/seed_shortlist.csv \
        --annotators 1,2

    # 3) 标完校验（span 对不上原文直接报错）
    python scripts/seed_annotation.py check \
        --annotations out/annotation_1.csv --shortlist out/seed_shortlist.csv

    # 4) 一致性 + 分歧清单（仲裁用）
    python scripts/seed_annotation.py kappa \
        --a1 out/annotation_1.csv --a2 out/annotation_2.csv

    # 5) 自测（不需要任何外部文件）
    python scripts/seed_annotation.py --self-test

退出码
------
0 = 通过；1 = 发现问题；2 = 用法错误
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

T = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 归一化实现只允许有一份 —— 直接复用校验器里的，禁止在这里另写一遍。
# （两边写法一旦分叉，source_id 与 A 的 seed_normalized_hash 就对不上）
import validate_schema as V                                   # noqa: E402

# Windows 控制台默认 GBK，遇到 emoji / 生僻符号会 UnicodeEncodeError。
# 统一降级为"打不出的字符显示成 ?"，避免因为**数据里**的字符把脚本崩掉。
try:                                                     # pragma: no cover
    sys.stdout.reconfigure(errors="replace")
except Exception:                                        # pragma: no cover
    pass

DATASET_CODE = {
    "ToxiCN": "toxi",
    "COLD": "cold",
    "CNTP": "cntp",
    "ToxiRewriteCN": "trcn",
}

LABELS = {"0", "1", "toxic", "non_toxic", "nontoxic"}
LABEL_NORM = {"0": "0", "1": "1", "toxic": "1", "non_toxic": "0", "nontoxic": "0"}

SHORTLIST_COLS = [
    "rank", "quota_stratum", "in_quota", "source_id", "len_chars", "len_bucket",
    "toxic_type", "topic", "platform", "source_dataset", "orig_file", "orig_line",
    "seed_normalized_hash", "family_size", "text",
    # 以下是留空的标注列，由人工填写
    "target_span_start", "target_span_end", "target_span_text",
    "toxicity_label", "attack_candidates", "notes",
]

ANNOTATION_COLS = [
    "source_id", "annotator", "guideline_version", "text",
    "target_span_start", "target_span_end", "target_span_text",
    "toxicity_label", "confidence", "notes",
]


# ---------------------------------------------------------------- 基础工具

def len_bucket(n: int) -> str:
    """长度分桶。短的更好标目标词，也更容易被模型判错，所以按桶分层。"""
    if n <= 10:
        return "1-10"
    if n <= 15:
        return "11-15"
    if n <= 20:
        return "16-20"
    return "21+"


def make_source_id(dataset: str, full_hash: str, reg: dict[str, str]) -> str:
    """`<dataset_code>-<哈希前缀>[-序号]`，见 docs/02 §3.1。

    三条规则（按优先级）：
      1. 前缀唯一 → 直接用前 10 位；
      2. 前缀相同但**整串哈希不同** → **加长前缀**（10 → 12 → …），不加序号。
         因为序号会破坏「纯内容派生」这个性质，且可能跨数据集撞号；
      3. **整串哈希相同**（归一化文本完全一致）→ 加长前缀没有意义，此时才加 `-2`/`-3`。

    `reg` 是整个批次共享的注册表：`id -> 完整归一化哈希`。
    """
    code = DATASET_CODE.get(dataset, dataset.lower()[:4])
    n = 10
    while n <= len(full_hash):
        cand = f"{code}-{full_hash[:n]}"
        owner = reg.get(cand)
        if owner is None:
            reg[cand] = full_hash
            return cand
        if owner == full_hash:
            k = 2
            while f"{cand}-{k}" in reg:
                k += 1
            sid = f"{cand}-{k}"
            reg[sid] = full_hash
            return sid
        n += 2                                        # 前缀撞了 → 加长，不是加序号
    raise ValueError(f"哈希前缀加到全长仍冲突：{full_hash}")


def read_csv(path: Path) -> tuple[list[dict], list[str]]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rd = csv.DictReader(fh)
        rows = list(rd)
        cols = list(rd.fieldnames or [])
    return rows, cols


def write_csv(path: Path, cols: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow({c: r.get(c, "") for c in cols})


def alloc_quota(counts: dict[str, int], target: int) -> dict[str, int]:
    """按分层大小**比例**分配配额，最大余数法补足；每层至少 1（若配额够）。"""
    strata = {k: v for k, v in counts.items() if v > 0}
    total = sum(strata.values())
    if total == 0 or target <= 0:
        return {k: 0 for k in strata}
    target = min(target, total)
    raw = {k: v * target / total for k, v in strata.items()}
    floor = {k: int(v) for k, v in raw.items()}
    # 每层至少 1，但要保证总和不超 target
    if target >= len(strata):
        for k in floor:
            if floor[k] == 0:
                floor[k] = 1
    rest = target - sum(floor.values())
    if rest > 0:
        order = sorted(strata, key=lambda k: (-(raw[k] - int(raw[k])), k))
        for i in range(rest):
            floor[order[i % len(order)]] += 1
    elif rest < 0:
        order = sorted(strata, key=lambda k: (floor[k] - raw[k], k), reverse=True)
        for i in range(-rest):
            k = order[i % len(order)]
            if floor[k] > 1:
                floor[k] -= 1
    return floor


def iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    """两个 span 的字符级 IoU。用于 judge 两个人标得"差不多"还是"差很远"。"""
    if a[0] >= a[1] or b[0] >= b[1]:
        return 0.0
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def cohen_kappa(pairs: list[tuple[str, str]]) -> tuple[float, float]:
    """Cohen's κ 与观察一致率 Po。pairs = [(rater1, rater2), ...]。"""
    if not pairs:
        return float("nan"), float("nan")
    n = len(pairs)
    po = sum(1 for a, b in pairs if a == b) / n
    ca = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)
    pe = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    if pe >= 1.0:
        return 1.0, po
    return (po - pe) / (1.0 - pe), po


def _is_blank(v: str | None) -> bool:
    return v is None or str(v).strip() == ""


# ---------------------------------------------------------------- stratify

def cmd_stratify(args: argparse.Namespace) -> int:
    pool_path = Path(args.pool)
    if not pool_path.exists():
        print(f"[FAIL] 候选池不存在：{pool_path}")
        return 2
    rows, cols = read_csv(pool_path)
    if not rows:
        print(f"[FAIL] 候选池为空：{pool_path}")
        return 2
    if "text" not in cols:
        print(f"[FAIL] 候选池缺 text 列，实际列：{','.join(cols)}")
        return 2

    print(f"[1] 候选池 {len(rows)} 条，列：{','.join(cols)}")

    # ---- 逐条重算哈希（不盲信上游给的列），并核对是否一致
    seen: dict[str, int] = {}
    hash_mismatch = 0
    short = []
    for r in rows:
        text = r.get("text") or ""
        h = V.sha(V.norm(text))
        if "seed_normalized_hash" in cols and r.get("seed_normalized_hash", "").strip():
            if r["seed_normalized_hash"].strip() != h:
                hash_mismatch += 1
        n = len(text)
        if n > args.max_len:
            continue
        short.append({
            "source_id": make_source_id(r.get("source_dataset", ""), h, seen),
            "len_chars": n,
            "len_bucket": len_bucket(n),
            "toxic_type": (r.get("toxic_type") or "").strip(),
            "topic": (r.get("topic") or "").strip(),
            "platform": (r.get("platform") or "").strip(),
            "source_dataset": (r.get("source_dataset") or "").strip(),
            "orig_file": (r.get("orig_file") or "").strip(),
            "orig_line": (r.get("orig_line") or "").strip(),
            "seed_normalized_hash": h,
            "text": text,
        })
    print(f"[2] 重算归一化哈希：与上游列不一致 {hash_mismatch} 条"
          f"{'（口径一致）' if hash_mismatch == 0 else ' ⚠️ 两边口径可能不同！'}")

    if not short:
        print(f"[FAIL] 没有长度 ≤{args.max_len} 的候选，放宽 --max-len 再试")
        return 1

    # ---- 家族大小（同哈希即同家族；split 与统计都要按家族算）
    fam = Counter(s["seed_normalized_hash"] for s in short)
    for s in short:
        s["family_size"] = fam[s["seed_normalized_hash"]]

    # ---- 分层 = (toxic_type, len_bucket)，按比例分配配额
    strata: dict[str, list[dict]] = defaultdict(list)
    for s in short:
        key = f"ttype={s['toxic_type'] or 'NA'}|len={s['len_bucket']}"
        s["quota_stratum"] = key
        strata[key].append(s)
    quota = alloc_quota({k: len(v) for k, v in strata.items()}, args.target)

    print(f"[3] ≤{args.max_len} 字候选 {len(short)} 条，分 {len(strata)} 层，"
          f"目标配额 {args.target} 条：")
    for k in sorted(strata):
        print(f"      {k:32s} 池 {len(strata[k]):4d} → 配额 {quota[k]:3d}")

    # ---- 排序：先按配额层，层内按长度升序 + 哈希定序（可复现，不带主观偏好）
    ordered: list[dict] = []
    for k in sorted(strata):
        group = sorted(strata[k], key=lambda s: (s["len_chars"], s["seed_normalized_hash"]))
        for i, s in enumerate(group):
            s["_in_quota"] = i < quota[k]
            ordered.append(s)
    # 配额内的排前面，方便标注者先标够数
    ordered.sort(key=lambda s: (not s["_in_quota"], s["quota_stratum"], s["len_chars"],
                                s["seed_normalized_hash"]))
    for i, s in enumerate(ordered, 1):
        s["rank"] = i
        s["in_quota"] = "1" if s["_in_quota"] else "0"

    out = Path(args.out)
    write_csv(out, SHORTLIST_COLS, ordered)
    print(f"[4] 写出 {out}（{len(ordered)} 条，其中配额内 "
          f"{sum(1 for s in ordered if s['in_quota'] == '1')} 条）")

    # ---- 报告
    rep = out.with_name(out.stem + "_report.md")
    with open(rep, "w", encoding="utf-8") as fh:
        fh.write("# 种子候选分层短名单报告\n\n")
        fh.write(f"> 由 `scripts/seed_annotation.py stratify` 生成\n")
        fh.write(f"> 输入：`{pool_path.name}`（{len(rows)} 条）\n")
        fh.write(f"> 输出：`{out.name}`（{len(ordered)} 条）\n\n")
        fh.write("## 一、筛选与分层\n\n")
        fh.write(f"- 长度 ≤{args.max_len} 字：**{len(short)}** 条 / 共 {len(rows)} 条\n")
        fh.write(f"- 分层维度：`toxic_type` × 长度桶 → **{len(strata)}** 层\n")
        fh.write(f"- 目标配额：**{args.target}** 条（按层大小比例分配，最大余数法）\n")
        fh.write(f"- 归一化哈希与上游 `seed_normalized_hash` 不一致：**{hash_mismatch}** 条\n\n")
        fh.write("## 二、分层明细\n\n")
        fh.write("| 分层 | 池内条数 | 配额 |\n|---|---|---|\n")
        for k in sorted(strata):
            fh.write(f"| `{k}` | {len(strata[k])} | {quota[k]} |\n")
        fh.write(f"\n## 三、长度分布\n\n| 长度桶 | 条数 |\n|---|---|\n")
        lb = Counter(s["len_bucket"] for s in short)
        for k in ["1-10", "11-15", "16-20", "21+"]:
            if lb.get(k):
                fh.write(f"| {k} | {lb[k]} |\n")
        fh.write("\n## 四、⚠️ 本文件**没有** `target_span`\n\n")
        fh.write("`target_span` 需要人工标注，且**判据待评审（Q2）**：\n\n")
        fh.write("1. 一句里多个可攻击词（如 `河南新疆回族人`）标哪个？\n")
        fh.write("2. 整句攻击、没有可替换目标词（如 `不能呼吸是对的`）收不收？\n")
        fh.write("3. 标**毒性实体**（骂人的词）还是标**攻击对象**（被骂的群体）？\n\n")
        fh.write("Q2 定案前**不要**开始标，否则判据一变就要返工。\n")
    print(f"[5] 报告：{rep}")
    return 0


# ---------------------------------------------------------------- sheet

def cmd_sheet(args: argparse.Namespace) -> int:
    short_path = Path(args.shortlist)
    if not short_path.exists():
        print(f"[FAIL] 短名单不存在：{short_path}")
        return 2
    rows, cols = read_csv(short_path)
    if "source_id" not in cols or "text" not in cols:
        print(f"[FAIL] 短名单缺 source_id/text 列，实际：{','.join(cols)}")
        return 2

    # 默认只发配额内的行；--all 则全发
    if not args.all and "in_quota" in cols:
        quota_rows = [r for r in rows if (r.get("in_quota") or "1").strip() != "0"]
        if quota_rows:
            rows = quota_rows
    if args.top and args.top > 0:
        rows = rows[: args.top]

    # 两个标注者拿到**完全相同顺序**，但顺序本身被打乱（避免从头到尾的疲劳偏差）
    if args.shuffle_seed:
        rows = rows[:]
        random.Random(args.shuffle_seed).shuffle(rows)

    tags = [t.strip() for t in args.annotators.split(",") if t.strip()]
    if len(tags) < 2:
        print("[WARN] 少于 2 个标注者，后续算不了一致性（D-2 要求双人独立）")

    outdir = Path(args.outdir)
    written = []
    for tag in tags:
        out = outdir / f"annotation_{tag}.csv"
        data = [{
            "source_id": r["source_id"],
            "annotator": tag,
            "guideline_version": args.guideline_version,
            "text": r["text"],
            # 以下留空由人工填
            "target_span_start": "", "target_span_end": "", "target_span_text": "",
            "toxicity_label": "", "confidence": "", "notes": "",
        } for r in rows]
        write_csv(out, ANNOTATION_COLS, data)
        written.append((out, len(data)))

    print(f"[1] 标注表 {len(rows)} 条 × {len(tags)} 人")
    for out, n in written:
        print(f"      {out}  ({n} 行)")
    print("[2] [WARN] 两份表必须**独立**填写，禁止互看；分歧交给仲裁裁定（D-2）")
    print("[3] 填写要求：")
    print("      target_span_start/end 用**半开区间** [start, end)，0 起计，针对 text 原文")
    print("      target_span_text 必须与 text[start:end] 完全一致（check 会逐条比对）")
    print("      toxicity_label 填 1/0（或 toxic/non_toxic）")
    print("      整句攻击、无可替换目标词：span 留空并在 notes 写「整句攻击」")
    return 0


# ---------------------------------------------------------------- check

def cmd_check(args: argparse.Namespace) -> int:
    ann_path = Path(args.annotations)
    if not ann_path.exists():
        print(f"[FAIL] 标注文件不存在：{ann_path}")
        return 2
    ann, cols = read_csv(ann_path)
    if not ann:
        print(f"[FAIL] 标注文件为空：{ann_path}")
        return 2
    need = {"source_id", "text", "target_span_start", "target_span_end",
            "target_span_text", "toxicity_label"}
    missing = need - set(cols)
    if missing:
        print(f"[FAIL] 缺列：{','.join(sorted(missing))}")
        return 2

    text_of: dict[str, str] = {}
    if args.shortlist and Path(args.shortlist).exists():
        srows, _ = read_csv(Path(args.shortlist))
        text_of = {r["source_id"]: r["text"] for r in srows if r.get("source_id")}

    errors: list[str] = []
    warns: list[str] = []
    done = 0
    pending = 0
    empty_span = 0
    seen: set[str] = set()

    for i, r in enumerate(ann, 2):                    # 2 = CSV 行号（含表头）
        sid = (r.get("source_id") or "").strip()
        if not sid:
            errors.append(f"行 {i}: source_id 为空")
            continue
        if sid in seen:
            errors.append(f"行 {i}: source_id 重复 {sid}")
        seen.add(sid)

        text = r.get("text") or ""
        if not text and sid in text_of:
            text = text_of[sid]
        if not text:
            errors.append(f"行 {i}: {sid} 的 text 为空，且短名单里找不到")
            continue
        if sid in text_of and text_of[sid].strip() != text.strip():
            errors.append(f"行 {i}: {sid} 的 text 与短名单不一致（可能被改动）")
            continue

        s_raw = (r.get("target_span_start") or "").strip()
        e_raw = (r.get("target_span_end") or "").strip()
        span_raw = (r.get("target_span_text") or "").strip()
        lab = (r.get("toxicity_label") or "").strip()

        all_blank = _is_blank(s_raw) and _is_blank(e_raw) and _is_blank(lab)
        if all_blank:
            pending += 1
            continue

        if lab:
            if lab.lower() not in LABELS:
                errors.append(f"行 {i}: {sid} toxicity_label 非法 `{lab}`（只接受 1/0/toxic/non_toxic）")
            elif LABEL_NORM[lab.lower()] != "1":
                errors.append(f"行 {i}: {sid} toxicity_label={lab}，但表2 种子必须是**毒性正例**")

        if _is_blank(s_raw) and _is_blank(e_raw):
            empty_span += 1
            notes = (r.get("notes") or "").strip()
            if not args.allow_empty_span:
                errors.append(
                    f"行 {i}: {sid} 没有 target_span。若确属「整句攻击、无可替换目标词」，"
                    f"请加 --allow-empty-span 并在 notes 写明")
            elif not notes:
                errors.append(f"行 {i}: {sid} 空 span 必须在 notes 里写明原因")
            done += 1
            continue

        try:
            s, e = int(s_raw), int(e_raw)
        except ValueError:
            errors.append(f"行 {i}: {sid} span 起止不是整数（`{s_raw}` / `{e_raw}`）")
            continue
        if not (0 <= s < e <= len(text)):
            errors.append(f"行 {i}: {sid} span [{s},{e}) 越界（text 长度 {len(text)}）")
            continue
        actual = text[s:e]
        if span_raw and span_raw != actual:
            errors.append(f"行 {i}: {sid} target_span_text 与原文不符\n"
                          f"        填的是 `{span_raw}`\n"
                          f"        实际是 `{actual}`")
            continue
        if not span_raw:
            warns.append(f"行 {i}: {sid} target_span_text 为空（可接受，但建议补上便于人工复核）")
        done += 1

    print(f"[1] 标注 {len(ann)} 行：已完成 {done} / 待标 {pending}")
    print(f"      其中空 span（整句攻击类）{empty_span} 行")
    if warns:
        print(f"[2] 警告 {len(warns)} 条：")
        for w in warns[:10]:
            print(f"      {w}")
        if len(warns) > 10:
            print(f"      … 另 {len(warns) - 10} 条")
    if errors:
        print(f"[3] [FAIL] 错误 {len(errors)} 条：")
        for e in errors[:20]:
            print(f"      {e}")
        if len(errors) > 20:
            print(f"      … 另 {len(errors) - 20} 条")
        return 1
    print("[3] [OK] 全部通过")
    return 0


# ---------------------------------------------------------------- kappa

def _load_ann(path: Path) -> dict[str, dict]:
    rows, cols = read_csv(path)
    if "source_id" not in cols:
        raise SystemExit(f"[FAIL] {path} 缺 source_id 列")
    return {r["source_id"]: r for r in rows if r.get("source_id")}


def _span_of(r: dict) -> tuple[int, int] | None:
    s, e = (r.get("target_span_start") or "").strip(), (r.get("target_span_end") or "").strip()
    if not s or not e:
        return None
    try:
        s_i, e_i = int(s), int(e)
    except ValueError:
        return None
    return (s_i, e_i)


def cmd_kappa(args: argparse.Namespace) -> int:
    a1_path, a2_path = Path(args.a1), Path(args.a2)
    for p in (a1_path, a2_path):
        if not p.exists():
            print(f"[FAIL] 不存在：{p}")
            return 2
    a1, a2 = _load_ann(a1_path), _load_ann(a2_path)
    common = sorted(set(a1) & set(a2))
    if not common:
        print("[FAIL] 两份标注没有共同的 source_id，检查是否同一份短名单")
        return 2

    label_pairs: list[tuple[str, str]] = []
    exact = overlap = total_span = 0
    ious: list[float] = []
    adjud: list[dict] = []
    only_one = 0

    for sid in common:
        r1, r2 = a1[sid], a2[sid]
        l1 = LABEL_NORM.get((r1.get("toxicity_label") or "").strip().lower())
        l2 = LABEL_NORM.get((r2.get("toxicity_label") or "").strip().lower())
        if l1 and l2:
            label_pairs.append((l1, l2))

        sp1, sp2 = _span_of(r1), _span_of(r2)
        text = r1.get("text") or r2.get("text") or ""
        if sp1 and sp2:
            total_span += 1
            if sp1 == sp2:
                exact += 1
                overlap += 1
            else:
                v = iou(sp1, sp2)
                ious.append(v)
                if v >= 0.5:
                    overlap += 1
        elif sp1 or sp2:
            only_one += 1

        if l1 != l2 or sp1 != sp2:
            if l1 or l2 or sp1 or sp2:
                adjud.append({
                    "source_id": sid,
                    "text": text,
                    "label_a1": l1 or "", "label_a2": l2 or "",
                    "span_a1": f"[{sp1[0]},{sp1[1]})" if sp1 else "",
                    "span_text_a1": text[sp1[0]:sp1[1]] if sp1 else "",
                    "span_a2": f"[{sp2[0]},{sp2[1]})" if sp2 else "",
                    "span_text_a2": text[sp2[0]:sp2[1]] if sp2 else "",
                    "iou": f"{iou(sp1, sp2):.2f}" if (sp1 and sp2) else "",
                    "verdict": "", "adjudicator": "",
                })

    kappa_label, po_label = cohen_kappa(label_pairs)

    print(f"[1] 共同条数 {len(common)}")
    print(f"[2] 毒性标签：可比 {len(label_pairs)} 条，一致率 {_fmt(po_label)}，"
          f"Cohen's κ = {_fmt(kappa_label)}")
    if total_span:
        print(f"[3] 目标词 span：可比 {total_span} 条")
        print(f"      完全一致      {exact}/{total_span} = {_pct(exact, total_span)}")
        print(f"      重叠 IoU≥0.5  {overlap}/{total_span} = {_pct(overlap, total_span)}")
        if ious:
            print(f"      不一致项平均 IoU = {sum(ious)/len(ious):.2f}")
    if only_one:
        print(f"[4] 只有一人给出 span 的 {only_one} 条（计入分歧）")
    print(f"[5] 分歧 {len(adjud)} 条 → 仲裁需要逐条裁定")

    outdir = a1_path.parent
    adj_path = Path(args.out) if args.out else outdir / "adjudication.csv"
    write_csv(adj_path, ["source_id", "text", "label_a1", "label_a2",
                         "span_a1", "span_text_a1", "span_a2", "span_text_a2",
                         "iou", "verdict", "adjudicator"], adjud)

    rep_path = adj_path.with_name("一致性报告.md")
    with open(rep_path, "w", encoding="utf-8") as fh:
        fh.write("# 双人独立标注一致性报告（D-2）\n\n")
        fh.write(f"> 由 `scripts/seed_annotation.py kappa` 生成\n")
        fh.write(f"> A1：`{a1_path.name}`｜A2：`{a2_path.name}`\n\n")
        fh.write("| 指标 | 数值 |\n|---|---|\n")
        fh.write(f"| 共同条数 | {len(common)} |\n")
        fh.write(f"| 毒性标签 可比条数 | {len(label_pairs)} |\n")
        fh.write(f"| 毒性标签 一致率 Po | {_fmt(po_label)} |\n")
        fh.write(f"| **毒性标签 Cohen's κ** | **{_fmt(kappa_label)}** |\n")
        fh.write(f"| span 可比条数 | {total_span} |\n")
        fh.write(f"| span 完全一致率 | {_pct(exact, total_span)} |\n")
        fh.write(f"| span 重叠率（IoU≥0.5） | {_pct(overlap, total_span)} |\n")
        if ious:
            fh.write(f"| 不一致项平均 IoU | {sum(ious)/len(ious):.2f} |\n")
        fh.write(f"| 仅一人给出 span | {only_one} |\n")
        fh.write(f"| 需仲裁条数 | {len(adjud)} |\n\n")
        fh.write("## 判读\n\n")
        fh.write("- κ < 0.40：判据没讲清，**先别标剩下的**，回去把 Q2 判据补成可操作的规则；\n")
        fh.write("- 0.40 ≤ κ < 0.61：可接受但需仲裁比例高，建议每条都过一遍仲裁；\n")
        fh.write("- κ ≥ 0.61：判据可用，分歧按 `adjudication.csv` 逐条裁定即可。\n\n")
        fh.write("> span 指标不是 κ（它测的是连续区间，不是分类），所以用一致率 + IoU 表示。\n")
    print(f"[6] 分歧清单：{adj_path}")
    print(f"[7] 报告：{rep_path}")
    return 0


def _fmt(v: float) -> str:
    return "n/a" if v != v else f"{v:.3f}"          # v != v 判 NaN


def _pct(a: int, b: int) -> str:
    return "n/a" if b == 0 else f"{a / b * 100:.1f}%"


# ---------------------------------------------------------------- 自测

def self_test() -> int:
    fails = []
    checks = [0]

    def chk(name: str, got, want) -> None:
        checks[0] += 1
        if got != want:
            fails.append(f"{name}: 得到 {got!r}，期望 {want!r}")

    # 1) Cohen's κ 构造例：N=50，A 判 20 正 / 30 负，B 判 25 正 / 25 负，
    #    在 20 个正例 + 25 个负例上一致 → Po=0.9，Pe=0.5，κ=0.8
    pairs = [("1", "1")] * 20 + [("0", "0")] * 25 + [("0", "1")] * 5
    k, po = cohen_kappa(pairs)
    chk("κ 构造例 Po", round(po, 4), 0.9)
    chk("κ 构造例", round(k, 4), 0.8)

    # 2) 完全一致 → κ=1
    k2, _ = cohen_kappa([("1", "1"), ("0", "0")])
    chk("κ 完全一致", round(k2, 4), 1.0)

    # 3) 两人都只用一个类别且相同 → pe=1，定义为 1
    k3, _ = cohen_kappa([("1", "1"), ("1", "1")])
    chk("κ 单一类别", k3, 1.0)

    # 4) 完全随机反向 → κ 为负
    k4, _ = cohen_kappa([("1", "0"), ("0", "1")])
    chk("κ 反向", k4 < 0, True)

    # 5) 长度分桶边界
    for n, want in [(1, "1-10"), (10, "1-10"), (11, "11-15"), (15, "11-15"),
                    (16, "16-20"), (20, "16-20"), (21, "21+"), (507, "21+")]:
        chk(f"len_bucket({n})", len_bucket(n), want)

    # 6) IoU
    chk("iou 相等", iou((0, 3), (0, 3)), 1.0)
    chk("iou 相交一半", iou((0, 3), (1, 4)), 0.5)
    chk("iou 不相交", iou((0, 2), (5, 7)), 0.0)
    chk("iou 空 span", iou((2, 2), (0, 3)), 0.0)

    # 7) 配额分配：3 层各 10，目标 9 → 各 3
    chk("配额 9/3 层", alloc_quota({"a": 10, "b": 10, "c": 10}, 9), {"a": 3, "b": 3, "c": 3})
    q = alloc_quota({"a": 10, "b": 10, "c": 10}, 10)
    chk("配额 10 总和", sum(q.values()), 10)
    # 8) 配额不超过池子
    q2 = alloc_quota({"a": 2, "b": 1}, 200)
    chk("配额不超池", sum(q2.values()), 3)
    # 9) 小层也至少 1（目标够大时）
    q3 = alloc_quota({"a": 1000, "b": 1}, 10)
    chk("小层保底", q3["b"] >= 1, True)

    # 10) source_id：前缀唯一→10 位；前缀撞但整串不同→加长；整串相同→加序号
    reg: dict[str, str] = {}
    h1 = "a" * 10 + "bb" + "c" * 52
    h2 = "a" * 10 + "dd" + "c" * 52               # 前 10 位相同，第 11-12 位不同
    chk("source_id 基本", make_source_id("ToxiCN", h1, reg), "toxi-aaaaaaaaaa")
    chk("source_id 前缀撞→加长前缀", make_source_id("ToxiCN", h2, reg), "toxi-aaaaaaaaaadd")
    chk("source_id 整串相同→加序号", make_source_id("ToxiCN", h1, reg), "toxi-aaaaaaaaaa-2")
    chk("source_id 整串相同→序号递增", make_source_id("ToxiCN", h1, reg), "toxi-aaaaaaaaaa-3")
    chk("source_id COLD", make_source_id("COLD", "c" * 64, {}), "cold-cccccccccc")
    chk("source_id 加长后也登记", reg.get("toxi-aaaaaaaaaadd"), h2)

    # 11) 标签归一
    chk("label toxic", LABEL_NORM["toxic"], "1")
    chk("label non_toxic", LABEL_NORM["non_toxic"], "0")

    if fails:
        print(f"[FAIL] 自测 {len(fails)}/{checks[0]} 项失败：")
        for f in fails:
            print("   ", f)
        return 1
    print(f"[OK] 自测 {checks[0]} 项全过")
    return 0


# ---------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="种子候选分层 / 标注表 / 标注校验 / 双人一致性",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法\n----")[1] if "用法\n----" in __doc__ else None)
    ap.add_argument("--self-test", action="store_true", help="跑内置自测，不需要外部文件")
    sub = ap.add_subparsers(dest="cmd")

    p1 = sub.add_parser("stratify", help="候选池 → 分层短名单（不落 target_span）")
    p1.add_argument("--pool", default=str(T / "data/raw/seeds/A交付_20261010/表2种子预留清单.csv"),
                    help="A 交付的候选池 CSV")
    p1.add_argument("--max-len", type=int, default=20, help="只取长度不超过该值的候选")
    p1.add_argument("--target", type=int, default=200, help="目标配额条数")
    p1.add_argument("--out", default=str(T / "out/seed_shortlist.csv"), help="输出短名单")
    p1.set_defaults(func=cmd_stratify)

    p2 = sub.add_parser("sheet", help="短名单 → 标注表（每标注者一份）")
    p2.add_argument("--shortlist", default=str(T / "out/seed_shortlist.csv"))
    p2.add_argument("--annotators", default="1,2", help="标注者标签，逗号分隔，如 1,2")
    p2.add_argument("--outdir", default=str(T / "out"))
    p2.add_argument("--top", type=int, default=0, help="只取前 N 条（0 = 全部）")
    p2.add_argument("--all", action="store_true", help="连配额外的行也发（默认只发配额内）")
    p2.add_argument("--shuffle-seed", type=int, default=7,
                    help="打乱顺序的随机种子（两人顺序一致；给 0 表示不 shuffle）")
    p2.add_argument("--guideline-version", default="v0.1",
                    help="标注指南版本号，跟着结果一起进交付清单")
    p2.set_defaults(func=cmd_sheet)

    p3 = sub.add_parser("check", help="标注结果合法性校验")
    p3.add_argument("--annotations", required=True)
    p3.add_argument("--shortlist", default=str(T / "out/seed_shortlist.csv"))
    p3.add_argument("--allow-empty-span", action="store_true",
                    help="允许空 span（仅限「整句攻击、无可替换目标词」，需在 notes 写明）")
    p3.set_defaults(func=cmd_check)

    p4 = sub.add_parser("kappa", help="双人一致性 + 分歧清单")
    p4.add_argument("--a1", required=True)
    p4.add_argument("--a2", required=True)
    p4.add_argument("--out", default=None, help="分歧清单输出路径（默认 adjudication.csv）")
    p4.set_defaults(func=cmd_kappa)

    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    if not getattr(args, "cmd", None):
        ap.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
