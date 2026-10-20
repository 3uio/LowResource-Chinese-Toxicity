#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_manifest.py —— 交付清单生成器（B 于 2026-10-11 明确要求的字段）

依据
----
* B 的交付要求（2026-10-11）：**最终交付清单须含**
  文件名 / 版本 / SHA-256 / 行数 / split 计数 / 标签分布 / 攻击类型分布 / 质检状态
* D-2：双人独立质检 + 仲裁，**须报告一致性**
* `docs/02_数据schema与source_id规则.md` §4（split 分配）

为什么需要它
------------
B 明确说过："现有 1175 条变体先视为**待质检数据**，不能把 schema 校验通过等同于数据质量验收。"
交付时如果还是一句"已交付 1175 条"，对方无法判断质量状态，
也无法在论文里写清数据版本。**这个脚本把交付清单固化成可机读 + 可人读两份**，
避免每次交付靠手写、字段漏项、SHA-256 忘记更新。

产出
----
* `out/delivery_manifest.csv` —— 机读，一行一个交付文件
* `out/交付清单.md`        —— 人读，含质检状态交叉表与复现命令

用法
----
    python scripts/make_manifest.py \
        --seeds out/seed_candidates.csv \
        --variants out/variants_v3.csv \
        --rules-dir data/rules \
        --version v0.3-pending-qc \
        --review-round 0 \
        --kappa 0.68 \
        --outdir out

    python scripts/make_manifest.py --self-test

退出码
------
0 = 正常；1 = 发现需要处理的问题（不阻断出清单）；2 = 用法错误
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# Windows 控制台默认 GBK，遇到 emoji / 生僻符号会 UnicodeEncodeError。
# 统一降级为"打不出的字符显示成 ?"，避免因为**数据里**的字符把脚本崩掉。
try:                                                     # pragma: no cover
    sys.stdout.reconfigure(errors="replace")
except Exception:                                        # pragma: no cover
    pass

T = Path(__file__).resolve().parent.parent

MANIFEST_COLS = [
    "file", "role", "version", "sha256", "bytes", "rows",
    "split_train", "split_dev", "split_test", "split_other",
    "label_toxic", "label_non_toxic", "label_missing",
    "qc_pass", "qc_fail", "qc_pending",
    "attack_types", "review_round", "kappa", "disagreement_rate",
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> tuple[list[dict], list[str]]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rd = csv.DictReader(fh)
        rows = list(rd)
        return rows, list(rd.fieldnames or [])


def _norm_label(v: str) -> str:
    v = (v or "").strip().lower()
    if v in {"1", "toxic", "true", "yes"}:
        return "1"
    if v in {"0", "non_toxic", "nontoxic", "false", "no"}:
        return "0"
    return ""


def summarize(rows: list[dict], cols: list[str]) -> dict:
    """统计一个表的 split / 标签 / 质检状态 / 攻击类型分布。

    同时记录**哪些列存在** —— 「没有该列」和「数值为 0」必须区分，
    否则变体表（标签继承自种子、本身没有 toxicity_label 列）会被显示成"0 条有标签"。
    """
    out = {
        "rows": len(rows),
        "split": Counter(), "label": Counter(), "qc": Counter(), "attack": Counter(),
        "has": set(cols),
    }
    has = lambda c: c in cols                                    # noqa: E731
    for r in rows:
        if has("split"):
            s = (r.get("split") or "").strip().lower()
            out["split"][s if s in {"train", "dev", "test"} else "other"] += 1
        if has("toxicity_label"):
            lb = _norm_label(r.get("toxicity_label", ""))
            out["label"][lb or "missing"] += 1
        if has("quality_pass"):
            q = (r.get("quality_pass") or "").strip().lower()
            out["qc"][q if q in {"pass", "fail", "pending"} else "pending"] += 1
        if has("attack_type"):
            a = (r.get("attack_type") or "").strip()
            if a:
                out["attack"][a] += 1
    return out


def crosstab(rows: list[dict], cols: list[str], row_key: str, col_key: str) -> dict:
    """二维交叉表：如 攻击类型 × 质检状态、split × 质检状态。"""
    table: dict[tuple[str, str], int] = defaultdict(int)
    for r in rows:
        if row_key not in cols or col_key not in cols:
            continue
        a = (r.get(row_key) or "").strip() or "NA"
        b = (r.get(col_key) or "").strip().lower() or "NA"
        table[(a, b)] += 1
    return table


def build(args: argparse.Namespace) -> int:
    files: list[tuple[Path, str]] = []
    if args.seeds:
        files.append((Path(args.seeds), "表2 种子集"))
    if args.variants:
        files.append((Path(args.variants), "表2 变体集"))
    rules_dir = Path(args.rules_dir) if args.rules_dir else None
    if rules_dir and rules_dir.is_dir():
        for p in sorted(rules_dir.glob("*.csv")):
            files.append((p, "映射表"))

    missing = [str(p) for p, _ in files if not p.exists()]
    if missing:
        print("[FAIL] 以下文件不存在，清单不能凭空生成：")
        for m in missing:
            print("      ", m)
        return 2
    if not files:
        print("[FAIL] 至少要给 --seeds 或 --variants 之一")
        return 2

    problems: list[str] = []
    manifest: list[dict] = []

    for path, role in files:
        rows, cols = read_csv(path) if path.suffix == ".csv" else ([], [])
        s = summarize(rows, cols) if rows else {"rows": 0, "split": Counter(),
                                                "label": Counter(), "qc": Counter(),
                                                "attack": Counter(), "has": set()}
        rec = {
            "file": path.name,
            "role": role,
            "version": args.version,
            "sha256": sha256_of(path),
            "bytes": path.stat().st_size,
            "rows": s["rows"],
            "split_train": s["split"].get("train", 0),
            "split_dev": s["split"].get("dev", 0),
            "split_test": s["split"].get("test", 0),
            "split_other": s["split"].get("other", 0),
            "label_toxic": s["label"].get("1", 0),
            "label_non_toxic": s["label"].get("0", 0),
            "label_missing": s["label"].get("missing", 0),
            "qc_pass": s["qc"].get("pass", 0),
            "qc_fail": s["qc"].get("fail", 0),
            "qc_pending": s["qc"].get("pending", 0),
            "attack_types": ";".join(f"{k}={v}" for k, v in sorted(s["attack"].items())),
            "review_round": args.review_round,
            "kappa": "" if args.kappa is None else f"{args.kappa:.3f}",
            "disagreement_rate": ("" if args.disagreement_rate is None
                                  else f"{args.disagreement_rate:.3f}"),
        }
        # 「列不存在」与「数值为 0」要分开，md 里据此换措辞
        rec["_has_label"] = "toxicity_label" in cols
        rec["_has_qc"] = "quality_pass" in cols
        rec["_has_attack"] = "attack_type" in cols
        rec["_has_split"] = "split" in cols
        manifest.append(rec)

        # ---- 自检：这些是交付时最容易"看起来没问题其实有问题"的点
        if role == "表2 变体集":
            if s["qc"].get("pass", 0) == 0 and s["qc"].get("pending", 0) == 0:
                problems.append(f"{path.name}: 没有 quality_pass 列或全为空，"
                                f"交付时必须写 quality_pass=pending")
            if s["qc"].get("pending", 0) and s["qc"].get("pass", 0) == 0:
                problems.append(f"{path.name}: 全部 pending —— 交付时可以这样，"
                                f"但**正文必须写『待质检数据』**，不得写『验收通过』")
            small = [(k, v) for k, v in s["attack"].items() if v < args.min_per_class]
            for k, v in small:
                problems.append(f"{path.name}: `{k}` 只有 {v} 条（< {args.min_per_class}），"
                                f"该类的统计结论只能标为探索性")
            if s["split"].get("test", 0) == 0:
                problems.append(f"{path.name}: test split 为 0")
        if role == "表2 种子集":
            if s["label"].get("0", 0):
                problems.append(f"{path.name}: 有 {s['label']['0']} 条非毒性种子 —— "
                                f"表2 种子应为毒性正例（安全对照另表）")
            if s["label"].get("missing", 0):
                problems.append(f"{path.name}: {s['label']['missing']} 条缺 toxicity_label")
        if path.suffix != ".csv":
            problems.append(f"{path.name}: 不是 CSV，未能统计行数与分布（只给了哈希）")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    man_path = outdir / "delivery_manifest.csv"
    md_path = outdir / "交付清单.md"

    with open(man_path, "w", encoding="utf-8-sig", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=MANIFEST_COLS)
        wr.writeheader()
        for r in manifest:
            wr.writerow({c: r.get(c, "") for c in MANIFEST_COLS})

    # ---------------- 人读清单
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("# Track C 交付清单\n\n")
        fh.write(f"> 版本：`{args.version}`　质检轮次：{args.review_round}")
        if args.kappa is not None:
            fh.write(f"　双人一致性 κ：{args.kappa:.3f}")
        fh.write("\n")
        fh.write(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}"
                 f"（时间仅作记录，不影响内容哈希）\n")
        fh.write("> 由 `scripts/make_manifest.py` 生成，**机读版**：`delivery_manifest.csv`\n\n")

        fh.write("## ⚠️ 质检状态声明\n\n")
        tot_pending = sum(r["qc_pending"] for r in manifest)
        tot_pass = sum(r["qc_pass"] for r in manifest)
        tot_fail = sum(r["qc_fail"] for r in manifest)
        if tot_pass == 0:
            fh.write("**本批数据为「待质检数据」（`quality_pass=pending`），"
                     "不是「验收通过」。**\n\n")
            fh.write("schema 校验通过只说明**字段与格式合法**，不代表质量已验收。"
                     "最终质量结论由 D 组织的双人独立质检 + 仲裁给出（D-2）。\n\n")
        else:
            fh.write(f"本批质检状态：pass {tot_pass} / fail {tot_fail} / pending {tot_pending}\n\n")

        fh.write("## 一、文件清单\n\n")
        fh.write("| 文件 | 角色 | 行数 | SHA-256（前 16 位） | 字节 |\n|---|---|---|---|---|\n")
        for r in manifest:
            fh.write(f"| `{r['file']}` | {r['role']} | {r['rows']} | "
                     f"`{r['sha256'][:16]}` | {r['bytes']:,} |\n")

        fh.write("\n## 二、split 分布\n\n")
        fh.write("| 文件 | train | dev | test | 其他 |\n|---|---|---|---|---|\n")
        for r in manifest:
            if not r["_has_split"]:
                fh.write(f"| `{r['file']}` | — | — | — | 本表无 split 列 |\n")
                continue
            fh.write(f"| `{r['file']}` | {r['split_train']} | {r['split_dev']} | "
                     f"{r['split_test']} | {r['split_other']} |\n")

        fh.write("\n## 三、标签分布\n\n")
        fh.write("| 文件 | 毒性 | 非毒性/安全对照 | 缺标签 |\n|---|---|---|---|\n")
        for r in manifest:
            if not r["_has_label"]:
                fh.write(f"| `{r['file']}` | — | — | 本表无 `toxicity_label` 列"
                         f"（标签继承自种子） |\n")
                continue
            fh.write(f"| `{r['file']}` | {r['label_toxic']} | {r['label_non_toxic']} | "
                     f"{r['label_missing']} |\n")

        fh.write("\n## 四、攻击类型分布\n\n")
        for r in manifest:
            if not r["attack_types"]:
                continue
            fh.write(f"**`{r['file']}`**\n\n| attack_type | 条数 |\n|---|---|\n")
            for kv in r["attack_types"].split(";"):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    fh.write(f"| `{k}` | {v} |\n")
            fh.write("\n")

        # 攻击类型 × 质检状态 / split
        vpath = Path(args.variants) if args.variants else None
        if vpath and vpath.exists():
            vrows, vcols = read_csv(vpath)
            t_qc = crosstab(vrows, vcols, "attack_type", "quality_pass")
            t_sp = crosstab(vrows, vcols, "attack_type", "split")
            if t_qc:
                statuses = sorted({b for _, b in t_qc})
                fh.write("## 五、攻击类型 × 质检状态\n\n")
                fh.write("| attack_type | " + " | ".join(statuses) + " |\n")
                fh.write("|---|" + "---|" * len(statuses) + "\n")
                for a in sorted({a for a, _ in t_qc}):
                    cells = [str(t_qc.get((a, s), 0)) for s in statuses]
                    fh.write(f"| `{a}` | " + " | ".join(cells) + " |\n")
                fh.write("\n")
            if t_sp:
                splits = [s for s in ("train", "dev", "test") if any(b == s for _, b in t_sp)]
                splits += sorted({b for _, b in t_sp} - set(splits))
                fh.write("## 六、攻击类型 × split\n\n")
                fh.write("| attack_type | " + " | ".join(splits) + " |\n")
                fh.write("|---|" + "---|" * len(splits) + "\n")
                for a in sorted({a for a, _ in t_sp}):
                    cells = [str(t_sp.get((a, s), 0)) for s in splits]
                    fh.write(f"| `{a}` | " + " | ".join(cells) + " |\n")
                fh.write("\n")

        fh.write("## 七、质检与一致性\n\n")
        fh.write("| 项 | 值 |\n|---|---|\n")
        fh.write(f"| 质检轮次 | {args.review_round} |\n")
        fh.write(f"| 双人一致性 κ | {'n/a' if args.kappa is None else f'{args.kappa:.3f}'} |\n")
        fh.write(f"| 分歧率 | {'n/a' if args.disagreement_rate is None else f'{args.disagreement_rate:.3f}'} |\n")
        fh.write("| 标注指南版本 | 见标注表 `guideline_version` 列（须一致） |\n")
        fh.write("| 一致性报告 | 由 `scripts/seed_annotation.py kappa` 生成 |\n\n")

        fh.write("## 八、复现命令\n\n```bash\n")
        fh.write("# 校验（正式交付一律带 --require-exclusion）\n")
        fh.write("python scripts/validate_schema.py \\\n")
        if args.seeds:
            fh.write(f"    --seeds {args.seeds} \\\n")
        if args.variants:
            fh.write(f"    --variants {args.variants} \\\n")
        fh.write("    --exclusion data/templates/table5_exclusion_list.csv \\\n")
        fh.write("    --require-exclusion\n")
        fh.write("```\n\n")
        fh.write("> 上面这行是本批数据的**唯一正式校验命令**。"
                 "不带 `--require-exclusion` 的运行只算本地联调，不构成交付验证。\n")

        if problems:
            fh.write("\n## 九、⚠️ 本清单指出的问题\n\n")
            for p in problems:
                fh.write(f"- {p}\n")

    print(f"[1] 机读清单：{man_path}")
    print(f"[2] 人读清单：{md_path}")
    print(f"[3] 文件 {len(manifest)} 个：")
    for r in manifest:
        print(f"      {r['file']:28s} {r['rows']:>6} 行  "
              f"sha256={r['sha256'][:12]}…  pending={r['qc_pending']}")
    if problems:
        print(f"[4] [WARN] {len(problems)} 条需要注意：")
        for p in problems:
            print("      -", p)
    else:
        print("[4] [OK] 无异常")
    return 0


# ---------------------------------------------------------------- 自测

def self_test() -> int:
    import tempfile
    fails, n = [], [0]

    def chk(name, got, want):
        n[0] += 1
        if got != want:
            fails.append(f"{name}: 得到 {got!r}，期望 {want!r}")

    # sha256 已知值（"abc"）
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as fh:
        fh.write(b"abc")
        tmp = Path(fh.name)
    chk("sha256(abc)", sha256_of(tmp),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    tmp.unlink()

    # 标签归一
    chk("label 1", _norm_label("1"), "1")
    chk("label toxic", _norm_label("TOXIC"), "1")
    chk("label 0", _norm_label("0"), "0")
    chk("label 空", _norm_label(""), "")

    rows = [
        {"split": "train", "toxicity_label": "1", "quality_pass": "pending", "attack_type": "Homo"},
        {"split": "dev", "toxicity_label": "1", "quality_pass": "pass", "attack_type": "Homo"},
        {"split": "test", "toxicity_label": "1", "quality_pass": "fail", "attack_type": "PY_Full"},
        {"split": "weird", "toxicity_label": "", "quality_pass": "", "attack_type": ""},
    ]
    cols = ["split", "toxicity_label", "quality_pass", "attack_type"]
    s = summarize(rows, cols)
    chk("rows", s["rows"], 4)
    chk("split train", s["split"]["train"], 1)
    chk("split other", s["split"]["other"], 1)
    chk("qc pending", s["qc"]["pending"], 2)      # 空值按 pending 计
    chk("label missing", s["label"]["missing"], 1)
    chk("attack Homo", s["attack"]["Homo"], 2)

    t = crosstab(rows, cols, "attack_type", "quality_pass")
    chk("交叉表 Homo/pending", t[("Homo", "pending")], 1)
    chk("交叉表 Homo/pass", t[("Homo", "pass")], 1)
    chk("交叉表 PY_Full/fail", t[("PY_Full", "fail")], 1)

    if fails:
        print(f"[FAIL] 自测 {len(fails)}/{n[0]} 项失败：")
        for f in fails:
            print("   ", f)
        return 1
    print(f"[OK] 自测 {n[0]} 项全过")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="交付清单生成器（B 要求的字段）")
    ap.add_argument("--seeds", default=None, help="表2 种子集 CSV")
    ap.add_argument("--variants", default=None, help="表2 变体集 CSV")
    ap.add_argument("--rules-dir", default=None, help="映射表目录（可选）")
    ap.add_argument("--version", default="v0.1-pending-qc", help="交付版本号")
    ap.add_argument("--review-round", type=int, default=0, help="质检轮次（0 = 未质检）")
    ap.add_argument("--kappa", type=float, default=None, help="双人一致性 κ")
    ap.add_argument("--disagreement-rate", type=float, default=None, help="分歧率 0-1")
    ap.add_argument("--min-per-class", type=int, default=50,
                    help="攻击类最少条数，低于此值统计只能标探索性")
    ap.add_argument("--outdir", default=str(T / "out"), help="输出目录")
    ap.add_argument("--self-test", action="store_true", help="跑内置自测")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    return build(args)


if __name__ == "__main__":
    sys.exit(main())
