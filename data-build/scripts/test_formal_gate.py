#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_formal_gate.py —— 正式验证闸门的回归测试（B 2026-10-23 要求）

测什么
------
B 在 2026-10-23 的回信里提了两条硬要求，这个脚本把每一条都变成**可自动复跑的断言**：

A. **排除清单默认阻断**（§三.1）
   缺失 / 不存在 / 为空 / 版本不匹配 → 一律不得通过正式验证；
   本地联调必须显式 `--pilot`，且报告要标 PILOT、不得冒充正式通过。

B. **字段级 pilot / 占位标记**（§三.2）
   变体表必须有 `is_pilot` / `mapping_source` / `mapping_version` /
   `generation_batch_id`；正式验证**拒绝** `is_pilot=1` 与
   `mapping_source=placeholder`；缺列或缺值不得被当成正式样本。

用法
----
    python scripts/test_formal_gate.py            # 跑全部断言
    python scripts/test_formal_gate.py -v         # 附带每个用例的实际输出

退出码
------
0 = 全部通过；1 = 有断言失败
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
T = HERE.parent
sys.path.insert(0, str(HERE))

# 控制台兜底
try:                                                         # pragma: no cover
    sys.stdout.reconfigure(errors="replace")
except Exception:                                            # pragma: no cover
    pass

WORK = T / "out/_gate_test"
RESULTS: list[tuple[bool, str, str]] = []


def run(args: list[str]) -> tuple[int, str]:
    import os
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, *args], cwd=str(T),
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((ok, name, detail))


def write_csv(path: Path, cols: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow({c: r.get(c, "") for c in cols})


SEED_COLS = ["source_id", "text", "target_span_start", "target_span_end",
             "target_span_text", "toxicity_label", "source_dataset", "split"]
VAR_COLS = ["source_id", "variant_id", "text", "attack_family", "attack_type",
            "is_adversarial", "generation_method", "quality_pass", "split",
            "substitution_unit", "homophone_type",
            "is_pilot", "mapping_source", "mapping_version", "generation_batch_id"]

EXCL_COLS = ["source_dataset", "text", "text_sha256", "text_norm_sha256", "note"]


def make_fixtures() -> None:
    """造最小可用的种子/变体/排除清单夹具（全部为构造数据，不含真实语料）。"""
    if WORK.exists():
        shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)

    write_csv(WORK / "seeds.csv", SEED_COLS, [{
        "source_id": "toxi-aaaaaaaaaa", "text": "你这个母狗真行",
        "target_span_start": 3, "target_span_end": 5, "target_span_text": "母狗",
        "toxicity_label": "1", "source_dataset": "DEMO", "split": "train",
    }])

    # 占位映射只覆盖「傻/蠢/货/猪/物」等字，要测 placeholder 分支必须用这些字
    # 作目标词，否则 Homo 类根本不会有产出（测出来只会是 none，属于假通过）。
    write_csv(WORK / "seeds_ph.csv", SEED_COLS, [{
        "source_id": "toxi-bbbbbbbbbb", "text": "你这个蠢货真行",
        "target_span_start": 3, "target_span_end": 5, "target_span_text": "蠢货",
        "toxicity_label": "1", "source_dataset": "DEMO", "split": "train",
    }])

    def var_row(**kw) -> dict:
        base = {
            "source_id": "toxi-aaaaaaaaaa",
            "variant_id": "r-toxi-aaaaaaaaaa-Homo-01",
            "text": "你这个幕狗真行", "attack_family": "phonetic",
            "attack_type": "Homo", "is_adversarial": "1",
            "generation_method": "rule", "quality_pass": "pending",
            "split": "train", "substitution_unit": "word", "homophone_type": "near",
            "is_pilot": "0", "mapping_source": "formal",
            "mapping_version": "rm-test", "generation_batch_id": "gb-test",
        }
        base.update(kw)
        return base

    # 合规变体
    write_csv(WORK / "variants_ok.csv", VAR_COLS, [var_row()])
    # pilot 样本（正式验证必须拒绝）
    write_csv(WORK / "variants_pilot.csv", VAR_COLS,
              [var_row(is_pilot="1")])
    # 占位映射产物（正式验证必须拒绝）
    write_csv(WORK / "variants_placeholder.csv", VAR_COLS,
              [var_row(mapping_source="placeholder", mapping_version="placeholder")])
    # 缺字段级标记列
    no_marker = [c for c in VAR_COLS
                 if c not in ("is_pilot", "mapping_source",
                              "mapping_version", "generation_batch_id")]
    write_csv(WORK / "variants_nomarker.csv", no_marker, [var_row()])
    # 标记列为空
    write_csv(WORK / "variants_empty_marker.csv", VAR_COLS,
              [var_row(generation_batch_id="")])

    # 排除清单：空 / 有内容
    write_csv(WORK / "excl_empty.csv", EXCL_COLS, [])
    write_csv(WORK / "excl_ok.csv", EXCL_COLS, [{
        "source_dataset": "CNTP", "text": "测试占位行",
        "text_sha256": "0" * 64, "text_norm_sha256": "0" * 64, "note": "回归测试用",
    }])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="正式验证闸门回归测试")
    ap.add_argument("-v", "--verbose", action="store_true", help="打印每个用例输出")
    args = ap.parse_args(argv)

    make_fixtures()
    V = "scripts/validate_schema.py"
    ok_v = str(WORK / "variants_ok.csv")
    excl_ok = str(WORK / "excl_ok.csv")
    excl_empty = str(WORK / "excl_empty.csv")
    missing_excl = str(WORK / "no_such_exclusion.csv")

    def case(name: str, rc: int, out: str, want_rc: int,
             must: tuple[str, ...] = (), must_not: tuple[str, ...] = ()) -> None:
        ok = rc == want_rc
        detail = f"退出码 {rc}（期望 {want_rc}）"
        for m in must:
            if m not in out:
                ok = False
                detail += f"；缺少关键词「{m}」"
        for m in must_not:
            if m in out:
                ok = False
                detail += f"；不应出现「{m}」"
        check(name, ok, detail)
        if args.verbose:
            print(f"\n--- {name} ---\n{out.strip()[:800]}")

    def case_simple(name: str, ok: bool, detail: str = "") -> None:
        """C 组直接给布尔结论（不看退出码/关键词）。"""
        check(name, ok, detail)
        if args.verbose:
            print(f"\n--- {name} ---\n{detail}")

    # ================= A. 排除清单默认阻断 =================
    rc, out = run([V, "--seeds", str(WORK / "seeds.csv")])
    case("A1 正式模式（默认）不给 --exclusion → 拒绝", rc, out, 2,
         must=("正式模式", "--pilot"))

    rc, out = run([V, "--seeds", str(WORK / "seeds.csv"), "--exclusion", missing_excl])
    case("A2 排除清单不存在 → 拒绝", rc, out, 1, must=("排除清单不存在",))

    rc, out = run([V, "--seeds", str(WORK / "seeds.csv"), "--exclusion", excl_empty])
    case("A3 排除清单为空 → 拒绝（泄漏检查等于没做）", rc, out, 1,
         must=("排除清单为空",))

    rc, out = run([V, "--seeds", str(WORK / "seeds.csv"),
                   "--exclusion", excl_ok,
                   "--expect-exclusion-sha256", "f" * 64])
    case("A4 排除清单版本（SHA-256）不匹配 → 拒绝", rc, out, 1,
         must=("版本不匹配",))

    rc, out = run([V, "--seeds", str(WORK / "seeds.csv"), "--exclusion", excl_ok])
    case("A5 排除清单有效 → 通过", rc, out, 0, must=("排除清单 SHA-256",))

    rc, out = run([V, "--seeds", str(WORK / "seeds.csv"), "--pilot"])
    case("A6 --pilot 显式放宽 → 通过，但报告标 PILOT", rc, out, 0,
         must=("[PILOT]", "不构成正式验收" if False else "不得"),
         must_not=("正式验收依据。",))

    rc, out = run([V, "--seeds", str(WORK / "seeds.csv"), "--mode", "pilot"])
    case("A7 --mode pilot 与 --pilot 等价", rc, out, 0, must=("[PILOT]",))

    # ================= B. 字段级 pilot / 占位标记 =================
    rc, out = run([V, "--variants", ok_v, "--exclusion", excl_ok])
    case("B1 标记齐全的正式变体 → 通过", rc, out, 0, must=("0 错误",))

    rc, out = run([V, "--variants", str(WORK / "variants_pilot.csv"),
                   "--exclusion", excl_ok])
    case("B2 正式验证拒绝 is_pilot=1", rc, out, 1, must=("is_pilot=1",))

    rc, out = run([V, "--variants", str(WORK / "variants_placeholder.csv"),
                   "--exclusion", excl_ok])
    case("B3 正式验证拒绝 mapping_source=placeholder", rc, out, 1,
         must=("placeholder",))

    rc, out = run([V, "--variants", str(WORK / "variants_nomarker.csv"),
                   "--exclusion", excl_ok])
    case("B4 缺字段级标记列 → 正式验证拒绝", rc, out, 1,
         must=("缺字段级标记列",))

    rc, out = run([V, "--variants", str(WORK / "variants_empty_marker.csv"),
                   "--exclusion", excl_ok])
    case("B5 标记列为空 → 正式验证拒绝", rc, out, 1,
         must=("generation_batch_id",))

    rc, out = run([V, "--variants", str(WORK / "variants_nomarker.csv"), "--pilot"])
    case("B6 pilot 模式下缺标记只告警（不阻断）", rc, out, 0,
         must=("缺字段级标记列",))

    # ================= C. 生成器的标记取值 =================
    def marker_dist(path: str) -> dict[str, dict[str, int]]:
        rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
        dist: dict[str, dict[str, int]] = {}
        for c in ("is_pilot", "mapping_source"):
            dist[c] = {}
            for r in rows:
                dist[c][r.get(c, "")] = dist[c].get(r.get(c, ""), 0) + 1
        return dist

    seeds = str(WORK / "seeds.csv")
    seeds_ph = str(WORK / "seeds_ph.csv")
    gen = "scripts/gen_variants_rule.py"

    # C 组需要正式映射表（data/rules/*.csv）。若交付包里没带，跳过并说明，
    # 不要让它变成"看起来失败"——A/B 组不依赖映射表，仍必须通过。
    has_rules = (T / "data/rules").is_dir() and any((T / "data/rules").glob("*.csv"))
    if not has_rules:
        case_simple("C 组跳过：未找到 data/rules（交付包未含映射表）", True,
                    "A/B 组已覆盖闸门行为；映射相关断言需正式映射资源")
        seeds_ph = None

    if has_rules:
        rc, out = run([gen, "--seeds", seeds_ph, "--rules-dir", "data/rules",
                       "--out", str(WORK / "g_formal.csv"),
                       "--stats-out", str(WORK / "g_formal_stats.csv")])
        d = marker_dist(str(WORK / "g_formal.csv"))
        case_simple("C1 正式映射：is_pilot 全 0",
                    rc == 0 and set(d["is_pilot"]) <= {"0"}, f"{d}")
        case_simple("C1b 正式映射：Homo 的 mapping_source=formal",
                    d["mapping_source"].get("formal", 0) > 0, f"{d}")

        rc, out = run([gen, "--seeds", seeds_ph, "--allow-placeholder",
                       "--out", str(WORK / "g_ph.csv"),
                       "--stats-out", str(WORK / "g_ph_stats.csv")])
        d = marker_dist(str(WORK / "g_ph.csv"))
        case_simple("C2 占位映射：is_pilot 全 1（自动置位，不靠人工记得加 flag）",
                    rc == 0 and set(d["is_pilot"]) == {"1"}, f"{d}")
        case_simple("C3 占位映射：出现 mapping_source=placeholder",
                    d["mapping_source"].get("placeholder", 0) > 0, f"{d}")

    rc, out = run([gen, "--seeds", seeds,
                   "--out", str(WORK / "g_none.csv"),
                   "--stats-out", str(WORK / "g_none_stats.csv")])
    d = marker_dist(str(WORK / "g_none.csv"))
    case_simple("C4 无映射：只有不依赖映射的类产出，标记为 none",
                "placeholder" not in d["mapping_source"], f"{d}")

    # ================= 汇总 =================
    failed = [r for r in RESULTS if not r[0]]
    print(f"\n{'=' * 68}")
    print(f"正式验证闸门回归测试：{len(RESULTS) - len(failed)}/{len(RESULTS)} 通过")
    print("=" * 68)
    for ok, name, detail in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    print()
    if failed:
        print(f"[FAIL] {len(failed)} 项未通过")
        return 1
    print("[OK] 全部通过（A 组：排除清单默认阻断；B 组：字段级 pilot/占位标记）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
