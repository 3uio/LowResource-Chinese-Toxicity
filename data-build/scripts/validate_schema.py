#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_schema.py —— Track C 数据校验 + 表5 泄漏查重（v0.1）

检查项
------
1. 必需列是否齐全
2. attack_family / attack_type 是否在词表内，二者是否自洽
3. is_adversarial / generation_method / quality_pass 取值是否合法
4. variant_id 是否唯一、是否与命名规则一致（r-/l- 前缀）
5. 外键：变体的 source_id 必须存在于种子表
6. **split 一致性**：同一 source_id 的所有变体必须同属一个 split
   （V2 第 4 节硬约束：禁止跨 split 拆分）
7. **表5 泄漏查重**：精确匹配 + 归一化匹配（去空白标点、全角转半角、
   去零宽字符、小写）。任一命中 → 报错并非零退出。

用法
----
    python scripts/validate_schema.py \
        --seeds data/raw/seeds/seed_candidates.csv \
        --variants out/variants.csv \
        --exclusion data/templates/table5_exclusion_list.csv

    # 只校验种子表
    python scripts/validate_schema.py --seeds X.csv

退出码
------
0 = 全部通过；1 = 发现错误；2 = 用法错误
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

FAMILY_OF = {
    "PY_Full": "phonetic", "PY_Abbr": "phonetic", "Homo": "phonetic",
    "Graph_Similar": "graphic", "Graph_Split": "graphic",
    "Sym_Leet": "symbolic", "Emoji_Sub": "symbolic", "Sym_Separator": "symbolic",
}

SEED_COLS = ["source_id", "text", "target_span_start", "target_span_end",
             "toxicity_label", "source_dataset", "split"]
VARIANT_COLS = ["source_id", "variant_id", "text", "attack_family", "attack_type",
                "is_adversarial", "generation_method", "quality_pass", "split"]

VALID_SPLITS = {"train", "dev", "test"}
VALID_METHODS = {"rule", "llm", "manual"}
VALID_QPASS = {"pass", "fail", "pending"}
# 替换单位：char=字级，word=整词级（见 docs/07_替换单位决策记录.md）
VALID_UNITS = {"char", "word"}
ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff"), None)

# 繁转简：与 A 侧（Track A 的清洗脚本）保持一致，两边必须用同一个实现。
# 缺这个库时**不能静默降级** —— 归一化哈希会与 A 不一致，跨表查重等于没查。
try:
    from zhconv import convert as _zh_convert
    HAS_ZHCONV = True
except ImportError:                                    # pragma: no cover
    _zh_convert = None
    HAS_ZHCONV = False

NORM_STEPS = "NFKC → 去零宽与BOM → 去空白与标点 → 小写 → 繁转简(zhconv) → SHA-256"


def norm(text: str) -> str:
    """归一化（五步，与 Track A 的实现严格一致）。

    顺序固定，**不得改动** —— 顺序不同则哈希不同，两边查重结果对不上：
        NFKC（全/半角、兼容字符统一）
        → 去零宽字符（U+200B/200C/200D/FEFF）与 BOM
        → 去全部空白与标点
        → 转小写
        → 繁转简（zhconv，缺库时为空操作并已在 main() 报警）
    """
    t = unicodedata.normalize("NFKC", text)
    t = t.translate(ZERO_WIDTH)
    t = "".join(ch for ch in t
                if not ch.isspace() and not unicodedata.category(ch).startswith("P"))
    t = t.lower()
    if HAS_ZHCONV:
        t = _zh_convert(t, "zh-cn")
    return t


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        r = csv.DictReader(fh)
        return list(r), (r.fieldnames or [])


class Report:
    def __init__(self):
        self.errors: list[str] = []
        self.warns: list[str] = []
        self.info: list[str] = []

    def err(self, msg): self.errors.append(msg)
    def warn(self, msg): self.warns.append(msg)
    def note(self, msg): self.info.append(msg)

    def dump(self) -> int:
        for m in self.info:
            print(f"[info] {m}")
        for m in self.warns:
            print(f"[warn] {m}")
        for m in self.errors:
            print(f"[ERROR] {m}")
        print(f"\n结果：{len(self.errors)} 错误 / {len(self.warns)} 警告 / {len(self.info)} 提示")
        return 1 if self.errors else 0


def check_columns(rep: Report, name: str, cols: list[str], need: list[str]):
    missing = [c for c in need if c not in cols]
    if missing:
        rep.err(f"{name}: 缺少必需列 {missing}")
    return not missing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Track C schema + 泄漏校验")
    ap.add_argument("--seeds", type=Path)
    ap.add_argument("--variants", type=Path)
    ap.add_argument("--safe-control", type=Path)
    ap.add_argument("--exclusion", type=Path,
                    help="表5 排除清单（table5_exclusion_list.csv）")
    args = ap.parse_args(argv)

    if not any([args.seeds, args.variants, args.safe_control]):
        ap.print_help()
        return 2

    # 繁转简是归一化的第 5 步，缺库会让本脚本算出的哈希与 Track A 不一致，
    # 跨表查重就会静默漏报。这里必须显式报警，不能装作没事。
    if not HAS_ZHCONV:
        print("[WARN] 未安装 zhconv —— 归一化缺第 5 步「繁转简」，"
              "本脚本算出的哈希与 Track A 不一致！")
        print("       跨表查重结果不可信（含繁体字的文本会漏报）。")
        print("       请先安装：python -m pip install zhconv")
        print(f"       应有口径：{NORM_STEPS}")
        print("       （继续执行，但泄漏检查结论不得用于交付）\n")

    rep = Report()
    seed_rows, seed_ids = [], set()

    # ---------------------------------------------------------- 种子表
    if args.seeds:
        if not args.seeds.exists():
            rep.err(f"种子表不存在：{args.seeds}")
        else:
            rows, cols = read_csv(args.seeds)
            if check_columns(rep, "种子表", cols, SEED_COLS):
                for i, r in enumerate(rows, start=2):
                    sid = (r.get("source_id") or "").strip()
                    if not sid:
                        rep.err(f"种子表 行{i}: source_id 为空")
                        continue
                    if sid in seed_ids:
                        rep.err(f"种子表 行{i}: source_id 重复：{sid}")
                    seed_ids.add(sid)
                    if (r.get("split") or "").strip() not in VALID_SPLITS:
                        rep.err(f"种子表 行{i} ({sid}): split 非法 "
                                f"'{r.get('split')}'，应为 {sorted(VALID_SPLITS)}")
                    txt = r.get("text") or ""
                    try:
                        s, e = int(r["target_span_start"]), int(r["target_span_end"])
                        if not (0 <= s < e <= len(txt)):
                            rep.err(f"种子表 行{i} ({sid}): span [{s},{e}) 越界"
                                    f"（文本长度 {len(txt)}）")
                        else:
                            declared = (r.get("target_span_text") or "").strip()
                            if declared and declared != txt[s:e]:
                                rep.err(f"种子表 行{i} ({sid}): target_span_text="
                                        f"'{declared}' 与 text[{s}:{e}]='{txt[s:e]}' 不一致")
                    except (TypeError, ValueError, KeyError):
                        rep.err(f"种子表 行{i} ({sid}): target_span_start/end 不是整数")
            rep.note(f"种子表：{len(rows)} 条")

    # ---------------------------------------------------------- 变体表
    split_of_seed: dict[str, str] = {}
    variant_ids: set[str] = set()
    if args.variants:
        if not args.variants.exists():
            rep.err(f"变体表不存在：{args.variants}")
        else:
            rows, cols = read_csv(args.variants)
            if check_columns(rep, "变体表", cols, VARIANT_COLS):
                seen_split: dict[str, set[str]] = defaultdict(set)
                for i, r in enumerate(rows, start=2):
                    vid = (r.get("variant_id") or "").strip()
                    sid = (r.get("source_id") or "").strip()
                    at = (r.get("attack_type") or "").strip()
                    fam = (r.get("attack_family") or "").strip()
                    sp = (r.get("split") or "").strip()

                    if not vid:
                        rep.err(f"变体表 行{i}: variant_id 为空")
                    elif vid in variant_ids:
                        rep.err(f"变体表 行{i}: variant_id 重复：{vid}")
                    else:
                        variant_ids.add(vid)
                        if not (vid.startswith("r-") or vid.startswith("l-")):
                            rep.err(f"变体表 行{i} ({vid}): variant_id 前缀必须是 r- 或 l-")
                        elif f"-{sid}-" not in vid:
                            rep.err(f"变体表 行{i} ({vid}): variant_id 与 source_id "
                                    f"'{sid}' 不一致")

                    if at not in FAMILY_OF:
                        rep.err(f"变体表 行{i} ({vid}): attack_type 非法 '{at}'")
                    elif fam != FAMILY_OF[at]:
                        rep.err(f"变体表 行{i} ({vid}): attack_family='{fam}' 与 "
                                f"attack_type='{at}'（应属 {FAMILY_OF[at]}）不自洽")

                    if (r.get("is_adversarial") or "").strip() not in {"1", "0"}:
                        rep.err(f"变体表 行{i} ({vid}): is_adversarial 必须为 0/1")
                    if (r.get("generation_method") or "").strip() not in VALID_METHODS:
                        rep.err(f"变体表 行{i} ({vid}): generation_method 非法")
                    if (r.get("quality_pass") or "").strip() not in VALID_QPASS:
                        rep.err(f"变体表 行{i} ({vid}): quality_pass 非法")

                    # substitution_unit：可选列（旧产物没有），有则必须是 char/word
                    # 依据 docs/07_替换单位决策记录.md（整词为主 + 字级兜底）
                    unit = (r.get("substitution_unit") or "").strip()
                    if unit and unit not in VALID_UNITS:
                        rep.err(f"变体表 行{i} ({vid}): substitution_unit "
                                f"非法 '{unit}'，应为 {sorted(VALID_UNITS)} 或留空")

                    if sp not in VALID_SPLITS:
                        rep.err(f"变体表 行{i} ({vid}): split 非法 '{sp}'")
                    if sid:
                        seen_split[sid].add(sp)
                        if seed_ids and sid not in seed_ids:
                            rep.err(f"变体表 行{i} ({vid}): source_id '{sid}' "
                                    f"在种子表中不存在")

                for sid, sps in seen_split.items():
                    if len(sps) > 1:
                        rep.err(f"变体表: source_id '{sid}' 跨 split {sorted(sps)}"
                                f" —— 违反 V2 第 4 节硬约束（禁止跨 split 拆分）")
            rep.note(f"变体表：{len(rows)} 条，{len(variant_ids)} 个唯一 variant_id")

    # ---------------------------------------------------------- 表5 泄漏查重
    if args.exclusion:
        if not args.exclusion.exists():
            rep.err(f"排除清单不存在：{args.exclusion}")
        else:
            ex_rows, ex_cols = read_csv(args.exclusion)
            if not ex_rows:
                rep.warn("排除清单为空 —— 表5 泄漏检查实际上没有生效。"
                         "拉取到表5 数据后必须回填（可只填哈希列）。")
            else:
                ex_sha, ex_norm = set(), set()
                for r in ex_rows:
                    h = (r.get("text_sha256") or "").strip()
                    hn = (r.get("text_norm_sha256") or "").strip()
                    t = (r.get("text") or "").strip()
                    if h:
                        ex_sha.add(h)
                    if hn:
                        ex_norm.add(hn)
                    if t:
                        ex_sha.add(sha(t))
                        ex_norm.add(sha(norm(t)))
                rep.note(f"排除清单：{len(ex_rows)} 条（哈希 {len(ex_sha)} / 归一化 {len(ex_norm)}）")

                targets = []
                if args.seeds and args.seeds.exists():
                    targets.append(("种子表", read_csv(args.seeds)[0]))
                if args.variants and args.variants.exists():
                    targets.append(("变体表", read_csv(args.variants)[0]))
                if args.safe_control and args.safe_control.exists():
                    targets.append(("表3", read_csv(args.safe_control)[0]))
                elif args.safe_control:
                    rep.err(f"表3 不存在：{args.safe_control}")

                hits = 0
                for name, rows in targets:
                    for i, r in enumerate(rows, start=2):
                        t = r.get("text") or ""
                        if not t:
                            continue
                        if sha(t) in ex_sha:
                            rep.err(f"表5 泄漏（精确匹配）：{name} 行{i} → {t[:30]}...")
                            hits += 1
                        elif sha(norm(t)) in ex_norm:
                            rep.err(f"表5 泄漏（归一化匹配）：{name} 行{i} → {t[:30]}...")
                            hits += 1
                if hits == 0:
                    rep.note("表5 泄漏检查：未发现命中")

    return rep.dump()


if __name__ == "__main__":
    raise SystemExit(main())
