#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_norm_hashes.py —— 导出归一化哈希清单（供 D 做跨表/跨期泄漏复核）

为什么需要
----------
D 在评估阶段要复核"测试集有没有漏进训练语料"（V2 的隔离要求）。
如果他手上只有表1 的原文，就得**自己再实现一遍五步归一化**才能比哈希 ——
而两边实现一旦有差异，查重结论就是错的（我们已经踩过一次：4 步实现漏了繁转简，
2,041 条里 16 条对不上，全部是繁体字）。

所以由 C 侧一次性导出**哈希清单**（只有哈希 + 溯源列，**不含原文**），D 直接拿去比对即可：

    # D 侧用法（示例）
    #   1) 把他要查的文本按同一口径算哈希（用 validate_schema.norm/sha）
    #   2) 与表1_归一化哈希.txt 做集合交集，交集非空即泄漏

用法
----
    # 表1（JSONL，每行一个 JSON 对象，取 text 字段）
    python scripts/export_norm_hashes.py \\
        --input "../data/raw/seeds/A交付_20261010/表1_无标注语料.jsonl" \\
        --text-field text --out out/表1_归一化哈希.txt

    # CSV 形式（如种子表/变体表）
    python scripts/export_norm_hashes.py --input out/seed_candidates.csv \\
        --text-field text --out out/种子_归一化哈希.txt

    python scripts/export_norm_hashes.py --self-test

输出格式（TSV，无表头行以 `#` 开头的说明）
------------------------------------------
    # 归一化哈希清单｜来源文件｜条数｜口径：NFKC → 去零宽BOM → 去空白标点 → 小写 → 繁转简(zhconv) → SHA-256
    <sha256>\t<source_dataset>\t<orig_file>\t<orig_line>
    ...

退出码：0 正常；2 用法错误
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

T = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 归一化只允许有一份实现
import validate_schema as V                                   # noqa: E402

# 控制台兜底：Windows GBK 控制台遇到 emoji / 生僻符号会崩
try:                                                         # pragma: no cover
    sys.stdout.reconfigure(errors="replace")
except Exception:                                            # pragma: no cover
    pass

NORM_NOTE = V.NORM_STEPS


def iter_records(path: Path, text_field: str):
    """按扩展名读 jsonl / csv，逐条产出 (text, dataset, orig_file, orig_line)。"""
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                yield (o.get(text_field) or "",
                       o.get("source") or o.get("source_dataset") or "",
                       o.get("orig_file") or "",
                       o.get("origin_id") or i)
    elif path.suffix.lower() == ".csv":
        with open(path, encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                yield (r.get(text_field) or "",
                       r.get("source_dataset") or "",
                       r.get("orig_file") or "",
                       r.get("orig_line") or "")
    else:
        raise SystemExit(f"[FAIL] 不支持的扩展名：{path.suffix}（只支持 .jsonl/.ndjson/.csv）")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="导出归一化哈希清单（给 D 做泄漏复核；不含原文）")
    ap.add_argument("--input", required=False, help="jsonl 或 csv")
    ap.add_argument("--text-field", default="text", help="取哪个字段作为文本")
    ap.add_argument("--out", default=str(T / "out/表1_归一化哈希.txt"))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    src = Path(args.input) if args.input else None
    if src is None or not src.exists():
        print(f"[FAIL] 输入不存在：{args.input}")
        return 2

    seen: dict[str, tuple[str, str, str]] = {}
    total = 0
    empty = 0
    for text, ds, of, ol in iter_records(src, args.text_field):
        total += 1
        if not (text or "").strip():
            empty += 1
            continue
        h = V.sha(V.norm(text))
        if h not in seen:                       # 保留首次出现的溯源信息
            seen[h] = (ds, of, ol)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# 归一化哈希清单｜来源文件｜条数｜口径：" + NORM_NOTE + "\n")
        fh.write(f"# 源文件：{src.name}｜总行数 {total}｜唯一哈希 {len(seen)}｜空文本 {empty}\n")
        fh.write("# 列：sha256\t source_dataset\t orig_file\t orig_line\n")
        for h, (ds, of, ol) in seen.items():
            fh.write(f"{h}\t{ds}\t{of}\t{ol}\n")

    print(f"[1] 源文件 {src.name}：{total} 行，唯一哈希 {len(seen)}，"
          f"空文本 {empty}（重复行 {total - empty - len(seen)}）")
    print(f"[2] 写出 {out}（{out.stat().st_size / 1024:.1f} KB，仅哈希 + 溯源，不含原文）")
    print(f"[3] 口径：{NORM_NOTE}")
    return 0


def self_test() -> int:
    fails, n = [], [0]

    def chk(name, got, want):
        n[0] += 1
        if got != want:
            fails.append(f"{name}: 得到 {got!r}，期望 {want!r}")

    # 归一化必须能抹平全半角 / 零宽 / 空白 / 大小写 / 繁体
    chk("NFKC+小写", V.norm("ＡＢＣ"), V.norm("abc"))
    chk("去空白标点", V.norm("河 南，人！"), V.norm("河南人"))
    chk("去零宽", V.norm("零\u200b宽\ufeff字"), V.norm("零宽字"))
    chk("繁转简", V.norm("繁體測試"), V.norm("繁体测试"))
    chk("拼音插空格", V.norm("sha si"), V.norm("shasi"))
    # 不同内容不能撞
    chk("不同内容不撞", V.norm("河南人") != V.norm("河北人"), True)
    # 哈希稳定：同一条内容的不同表面形式必须得到同一个哈希（这才是查重的基础）
    chk("哈希稳定（空白差异）", V.sha(V.norm("母狗")), V.sha(V.norm("母 狗")))
    chk("哈希稳定（全半角差异）", V.sha(V.norm("母狗")), V.sha(V.norm("母狗")))

    if fails:
        print(f"[FAIL] 自测 {len(fails)}/{n[0]} 项失败：")
        for f in fails:
            print("   ", f)
        return 1
    print(f"[OK] 自测 {n[0]} 项全过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
