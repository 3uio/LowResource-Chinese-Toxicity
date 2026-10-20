#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_swap.py —— `Swap`（换序）变体生成器【默认不启用，等 Q3 评审】

为什么单独一个脚本
------------------
A 的 README 指出：CNTP 的 `perturbed_data/` 8 类扰动"即 V2 所述 attack_type 的确切清单"，
其中有一类 **`Swap_transform`（换序）**，而我的 8 小类里**没有这一类**
（我多出来的是 `Sym_Separator` / `Sym_Leet`）。

是否补这一类属**评审事项（Q3）**，我不单方塞进主生成器，原因有二：
  1. 主生成器一改，8 类产出结构就变了，评审还没定就先改了结构，等于替团队拍板；
  2. `Swap` 归哪个 `attack_family` 也需要定 —— V2 只有 语音 / 字形 / 符号 三族，
     换序严格说哪一族都不算"扰动音形义"，硬塞会给统计口径埋雷。

所以：**本脚本默认什么都不做**，必须显式 `--enable` 才产出。Q3 通过后，
把 `--enable` 接进主流程（或直接批量跑本脚本）即可，不需要重写规则。

做什么
------
对已标好 `target_span` 的种子，在**目标词内部**做换序（不改动 span 以外的字符）：

| 模式 | 例 | 说明 |
|---|---|---|
| `adjacent` | `母狗 → 狗母` | 首对相邻字互换（最像真实用户手滑/规避写法） |
| `reverse` | `王八蛋 → 蛋八王` | 整段逆序 |
| `both` | 两种都出 | 默认 |

不做的情况（会打印原因，不静默丢）：
  * span 长度 < 2（无可换）
  * 换完与原词相同（如 `妈妈`、`哈哈`）
  * 目标词含非中文字符（emoji/数字/字母）——那属 `Emoji_Sub` / `Sym_Leet` 的范畴

用法
----
    # 现在是"关"的状态：只看会生成什么，不写文件
    python scripts/gen_swap.py --seeds out/seed_candidates.csv

    # Q3 通过后才这么跑（必须显式 --enable）
    python scripts/gen_swap.py --seeds out/seed_candidates.csv --enable \\
        --out out/variants_swap.csv --swap-family graphic

    python scripts/gen_swap.py --self-test

退出码
------
0 = 正常（含"未启用、只预演"）；1 = 有种子生成失败；2 = 用法错误
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

# Windows 控制台默认 GBK，遇到 emoji / 生僻符号会 UnicodeEncodeError。
# 统一降级为"打不出的字符显示成 ?"，避免因为**数据里**的字符把脚本崩掉。
try:                                                     # pragma: no cover
    sys.stdout.reconfigure(errors="replace")
except Exception:                                        # pragma: no cover
    pass

T = Path(__file__).resolve().parent.parent

ATTACK_TYPE = "Swap"
# V2 只有 phonetic / graphic / symbolic 三族；换序归哪一族**待 Q3 定**。
# 默认 graphic（字符排列属字形层面），但一定要在交付文档里注明它是待定项。
DEFAULT_FAMILY = "graphic"
VALID_FAMILIES = {"phonetic", "graphic", "symbolic"}

CJK_ONLY = re.compile(r"^[\u4e00-\u9fff]+$")

OUT_COLS = [
    "source_id", "variant_id", "text", "attack_family", "attack_type",
    "is_adversarial", "generation_method", "quality_pass", "split",
    "substitution_unit", "homophone_type",
    "swap_mode", "swapped_from", "swapped_to",
]


def swap_span(span: str, mode: str) -> tuple[str, str] | None:
    """在 span 内部做换序，返回 (新模式名, 新 span)；做不了返回 None。"""
    n = len(span)
    if n < 2:
        return None
    if mode == "adjacent":
        out = span[1] + span[0] + span[2:] if n > 2 else span[1] + span[0]
        if out == span:
            return None
        return ("adjacent", out)
    if mode == "reverse":
        out = span[::-1]
        if out == span:
            return None
        return ("reverse", out)
    raise ValueError(f"未知 swap 模式：{mode}")


def plan_text(text: str, start: int, end: int, mode: str) -> dict | None:
    """把 span 内部换序结果拼回整句。返回 None 表示这条生成不了。"""
    if not (0 <= start < end <= len(text)):
        return {"error": f"span [{start},{end}) 越界（text 长度 {len(text)}）"}
    span = text[start:end]
    if not CJK_ONLY.match(span):
        return {"error": f"目标词含非中文（`{span}`），属 Emoji_Sub/Sym_Leet 范畴，Swap 不处理"}
    got = swap_span(span, mode)
    if got is None:
        return {"error": f"换序后与原词相同或长度不足（`{span}`）"}
    mode_name, new_span = got
    new_text = text[:start] + new_span + text[end:]
    if new_text == text:
        return {"error": "换序后整句未变"}
    return {"mode": mode_name, "new_span": new_span, "new_text": new_text, "old_span": span}


def generate(rows: list[dict], modes: list[str], family: str,
             require_label: bool) -> tuple[list[dict], list[str]]:
    """对每个种子生成 Swap 变体。返回 (变体列表, 失败原因列表)。"""
    variants: list[dict] = []
    fails: list[str] = []
    for r in rows:
        sid = (r.get("source_id") or "").strip()
        text = r.get("text") or ""
        if not sid or not text:
            fails.append("有一行缺 source_id 或 text")
            continue
        if require_label:
            lab = (r.get("toxicity_label") or "").strip()
            if lab not in {"1", "toxic"}:
                fails.append(f"{sid}: 不是毒性正例（label=`{lab}`），Swap 只对有目标词的毒性种子生成")
                continue
        try:
            s = int((r.get("target_span_start") or "").strip())
            e = int((r.get("target_span_end") or "").strip())
        except ValueError:
            fails.append(f"{sid}: 缺 target_span_start/end（Swap 必须有目标词）")
            continue

        seq = 0
        for mode in modes:
            p = plan_text(text, s, e, mode)
            if p is None:
                continue
            if "error" in p:
                fails.append(f"{sid}: {p['error']}")
                continue
            seq += 1
            variants.append({
                "source_id": sid,
                "variant_id": f"r-{sid}-{ATTACK_TYPE}-{seq:02d}",
                "text": p["new_text"],
                "attack_family": family,
                "attack_type": ATTACK_TYPE,
                "is_adversarial": "1",
                "generation_method": "rule",
                "quality_pass": "pending",          # 未经人工质检，绝不自封 pass
                "split": (r.get("split") or "").strip(),   # 变体 split 必须与种子一致
                "substitution_unit": "word",
                "homophone_type": "",
                "swap_mode": p["mode"],
                "swapped_from": p["old_span"],
                "swapped_to": p["new_span"],
            })
    return variants, fails


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=OUT_COLS, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow({c: r.get(c, "") for c in OUT_COLS})


def run(args: argparse.Namespace) -> int:
    seeds_path = Path(args.seeds)
    if not seeds_path.exists():
        print(f"[FAIL] 种子文件不存在：{seeds_path}")
        return 2
    if args.swap_family not in VALID_FAMILIES:
        print(f"[FAIL] --swap-family 只能是 {sorted(VALID_FAMILIES)}")
        return 2

    rows = read_csv(seeds_path)
    modes = ["adjacent", "reverse"] if args.modes == "both" else [args.modes]
    variants, fails = generate(rows, modes, args.swap_family,
                               require_label=not args.allow_non_toxic)

    if not args.enable:
        print("[0] 状态：**未启用**（`Swap` 属 Q3 待评审项，默认不产出）")
        print("[1] 本次只做预演，不写任何文件。预演结果：")
        print(f"      种子 {len(rows)} 条 → 可生成 Swap 变体 {len(variants)} 条"
              f"（模式 {'/'.join(modes)}，族 `{args.swap_family}` 有待 Q3 确认）")
        by_mode: dict[str, int] = {}
        for v in variants:
            by_mode[v["swap_mode"]] = by_mode.get(v["swap_mode"], 0) + 1
        for k, n in sorted(by_mode.items()):
            print(f"      {k:10s} {n} 条")
        for f in fails[:8]:
            print(f"      [skip] {f}")
        if len(fails) > 8:
            print(f"      … 另 {len(fails) - 8} 条跳过")
        print("[2] Q3 定案后加 --enable 即写文件；规则本身不需要改。")
        return 0

    if not args.out:
        print("[FAIL] --enable 时必须给 --out")
        return 2
    out = Path(args.out)
    write_csv(out, variants)
    print(f"[1] 写出 {out}（{len(variants)} 条，全部 quality_pass=pending）")
    print(f"[2] attack_family=`{args.swap_family}` - 待 Q3 确认；"
          f"若评审改成新增第四族，只需改 --swap-family 与 schema 里的枚举")
    if fails:
        print(f"[3] [WARN] {len(fails)} 条种子跳过：")
        for f in fails[:10]:
            print("      -", f)
        return 1
    return 0


# ---------------------------------------------------------------- 自测

def self_test() -> int:
    fails, n = [], [0]

    def chk(name, got, want):
        n[0] += 1
        if got != want:
            fails.append(f"{name}: 得到 {got!r}，期望 {want!r}")

    # 相邻换序
    chk("adjacent 母狗", swap_span("母狗", "adjacent"), ("adjacent", "狗母"))
    chk("adjacent 王八蛋", swap_span("王八蛋", "adjacent"), ("adjacent", "八王蛋"))
    # 整段逆序
    chk("reverse 王八蛋", swap_span("王八蛋", "reverse"), ("reverse", "蛋八王"))
    chk("reverse 母狗", swap_span("母狗", "reverse"), ("reverse", "狗母"))
    # 换不出差别 → None
    chk("adjacent 妈妈", swap_span("妈妈", "adjacent"), None)
    chk("reverse 妈妈", swap_span("妈妈", "reverse"), None)
    chk("长度 1", swap_span("狗", "adjacent"), None)

    # 拼回整句：span 以外的字符必须不动
    p = plan_text("你这个母狗真行", 3, 5, "adjacent")
    chk("拼接后整句", p["new_text"], "你这个狗母真行")
    chk("拼接后旧词", p["old_span"], "母狗")
    # 越界
    chk("越界报错", "error" in plan_text("母狗", 0, 9, "adjacent"), True)
    # 非中文目标词 → 报错不生成
    chk("非中文跳过", "error" in plan_text("sb玩意", 0, 2, "reverse"), True)

    # 批量：只有正例 + 有 span 的才生成
    rows = [
        {"source_id": "toxi-aaaaaaaaaa", "text": "你这个母狗真行",
         "target_span_start": "3", "target_span_end": "5", "toxicity_label": "1",
         "split": "train"},
        {"source_id": "toxi-bbbbbbbbbb", "text": "路人甲路过",
         "target_span_start": "0", "target_span_end": "3", "toxicity_label": "0",
         "split": "train"},
        {"source_id": "toxi-cccccccccc", "text": "没有标注span",
         "target_span_start": "", "target_span_end": "", "toxicity_label": "1",
         "split": "dev"},
    ]
    vs, fs = generate(rows, ["adjacent", "reverse"], "graphic", require_label=True)
    chk("批量条数", len(vs), 2)                     # 第 1 行两种模式各 1 条
    chk("批量 source_id", sorted({v["source_id"] for v in vs}), ["toxi-aaaaaaaaaa"])
    chk("批量 split 继承", {v["split"] for v in vs}, {"train"})
    chk("批量 quality_pass", {v["quality_pass"] for v in vs}, {"pending"})
    chk("批量 variant_id 唯一", len({v["variant_id"] for v in vs}), 2)
    chk("失败被记录", len(fs), 2)                   # 非正例 + 缺 span

    if fails:
        print(f"[FAIL] 自测 {len(fails)}/{n[0]} 项失败：")
        for f in fails:
            print("   ", f)
        return 1
    print(f"[OK] 自测 {n[0]} 项全过")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Swap（换序）变体生成器 —— 默认不启用，等 Q3")
    ap.add_argument("--seeds", default=str(T / "out/seed_candidates.csv"),
                    help="含 target_span 的种子表")
    ap.add_argument("--out", default=None, help="输出变体 CSV（--enable 时必填）")
    ap.add_argument("--modes", default="both", choices=["adjacent", "reverse", "both"],
                    help="换序模式")
    ap.add_argument("--swap-family", default=DEFAULT_FAMILY,
                    help="attack_family 取值（待 Q3，默认 graphic）")
    ap.add_argument("--allow-non-toxic", action="store_true",
                    help="允许对非毒性种子生成（默认不允许）")
    ap.add_argument("--enable", action="store_true",
                    help="真正产出文件。不加此开关只预演，不写任何东西")
    ap.add_argument("--self-test", action="store_true", help="跑内置自测")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
