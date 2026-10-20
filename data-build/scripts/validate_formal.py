#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_formal.py —— 「正式校验」包装器（把 B 的第 2 条做成默认行为）

为什么需要它
------------
B 要求："表5 排除清单缺失时**阻断**正式验证"。

`validate_schema.py` 里已经有这个硬阻断，但它是**开关**（`--require-exclusion`）：
不带开关时只 warn。这样就有个漏洞 —— 谁忘了带开关，泄漏检查等于没做，
而输出看起来还是"校验通过"。

这个包装器把**"正式校验"这件事本身**定义成固定动作：
  * 必须给排除清单，且清单**必须有数据行**（空文件等于没查）；
  * 强制带上 `--require-exclusion`；
  * 输出写进 `out/正式校验日志.md`，可原样贴给 B/D；
  * 退出码与校验结果一致（0 通过 / 1 有问题 / 2 用法错误）。

用法
----
    python scripts/validate_formal.py \\
        --seeds out/seed_candidates.csv \\
        --variants out/variants_v3.csv

    # 排除清单默认取 data/templates/table5_exclusion_list.csv，也可显式指定
    python scripts/validate_formal.py --seeds X.csv --exclusion path/to/list.csv

退出码
------
0 = 通过；1 = 校验有问题 / 排除清单为空；2 = 用法错误
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

T = Path(__file__).resolve().parent.parent

# 控制台兜底
try:                                                         # pragma: no cover
    sys.stdout.reconfigure(errors="replace")
except Exception:                                            # pragma: no cover
    pass

DEFAULT_EXCLUSION = T / "data/templates/table5_exclusion_list.csv"


def count_rows(path: Path) -> int:
    """数排除清单里的**数据行**（不算表头和空行）。

    编码坏了（例如 Excel 默认存成 GBK）要给出人能看懂的提示，
    而不是甩一个 UnicodeDecodeError traceback —— 否则用的人会以为是脚本坏了。
    """
    if not path.exists():
        return -1
    try:
        with open(path, encoding="utf-8-sig", newline="") as fh:
            return sum(1 for r in csv.DictReader(fh)
                       if any((v or "").strip() for v in r.values()))
    except UnicodeDecodeError:
        print(f"[FAIL] 排除清单不是 UTF-8 编码：{path}")
        print("        请在 Excel 里「另存为 → CSV UTF-8」，或用 Python 以 utf-8 重写。")
        print("        （GBK 存出来的表，脚本读不了，也不该猜着救）")
        raise SystemExit(2)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="正式校验（强制要求表5 排除清单）")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--variants", default=None)
    ap.add_argument("--exclusion", default=str(DEFAULT_EXCLUSION))
    ap.add_argument("--log", default=str(T / "out/正式校验日志.md"))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    if not args.seeds and not args.variants:
        print("[FAIL] 至少要给 --seeds 或 --variants 之一")
        return 2
    for p in (args.seeds, args.variants):
        if p and not Path(p).exists():
            print(f"[FAIL] 文件不存在：{p}")
            return 2

    excl = Path(args.exclusion)
    n_ex = count_rows(excl)
    if n_ex < 0:
        print(f"[FAIL] 排除清单不存在：{excl}")
        print("        正式校验必须比对表5（CNTP）排除清单，否则泄漏检查等于没做。")
        print("        拉取到表5 数据后回填 data/templates/table5_exclusion_list.csv 再跑。")
        return 1
    if n_ex == 0:
        print(f"[FAIL] 排除清单为空：{excl}")
        print("        空清单 = 没有排除任何行 = 泄漏检查没生效。")
        print("        至少填表5 的归一化哈希列（不必填原文）。")
        return 1

    cmd = [sys.executable, str(T / "scripts/validate_schema.py")]
    if args.seeds:
        cmd += ["--seeds", str(args.seeds)]
    if args.variants:
        cmd += ["--variants", str(args.variants)]
    cmd += ["--exclusion", str(excl), "--require-exclusion"]

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run(cmd, cwd=str(T), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env)
    out = (r.stdout or "") + (r.stderr or "")

    shown = " ".join(str(c).replace(str(T) + os.sep, "") for c in cmd[1:])
    log = Path(args.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf-8") as fh:
        fh.write("# 正式校验日志\n\n")
        fh.write(f"> 时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        fh.write(f"> 排除清单：`{excl.name}`（{n_ex} 行）\n")
        fh.write("> 本日志由 `scripts/validate_formal.py` 生成；"
                 "**强制带 `--require-exclusion`**，空清单会被直接拒绝。\n\n")
        fh.write("## 命令\n\n```bash\npython scripts/validate_schema.py "
                 + shown + "\n```\n\n")
        fh.write(f"## 退出码：{r.returncode}\n\n")
        fh.write("## 输出\n\n```text\n")
        fh.write(out.strip() + "\n")
        fh.write("```\n")

    print(out.rstrip())
    print(f"\n[log] {log}")
    print(f"[exit] {r.returncode}")
    return r.returncode


def self_test() -> int:
    fails, n = [], [0]

    def chk(name, got, want):
        n[0] += 1
        if got != want:
            fails.append(f"{name}: 得到 {got!r}，期望 {want!r}")

    # 不存在 → -1
    chk("不存在", count_rows(Path("__no_such_file__.csv")), -1)

    # 注意：这里不用 tempfile.TemporaryDirectory() —— 在受限环境里删除临时目录会 PermissionError。
    # 改用仓库内 out/ 下的临时目录（out/ 已 gitignore），best-effort 清理。
    d = T / "out/_selftest_excl"
    d.mkdir(parents=True, exist_ok=True)
    try:
        p = d / "empty.csv"
        p.write_text("seed_normalized_hash,text\n", encoding="utf-8")
        chk("只有表头", count_rows(p), 0)
        p.write_text("seed_normalized_hash,text\n,\n", encoding="utf-8")
        chk("表头+空行", count_rows(p), 0)
        p.write_text("seed_normalized_hash,text\nabc123,\n", encoding="utf-8")
        chk("一行有值", count_rows(p), 1)
        p.write_text("seed_normalized_hash,text\nabc123,\ndef456,\n", encoding="utf-8")
        chk("两行", count_rows(p), 2)
    finally:
        for f in d.glob("*"):
            try:
                f.unlink()
            except OSError:
                pass
        try:
            d.rmdir()
        except OSError:
            pass

    if fails:
        print(f"[FAIL] 自测 {len(fails)}/{n[0]} 项失败：")
        for f in fails:
            print("   ", f)
        return 1
    print(f"[OK] 自测 {n[0]} 项全过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
