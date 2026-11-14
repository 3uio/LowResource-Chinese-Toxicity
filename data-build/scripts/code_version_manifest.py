#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code_version_manifest.py —— 生成「代码与规范版本清单」（B 2026-10-23 §四.3 要求）

为什么需要
----------
B 明确要求交付时说明"schema 与生成/验证代码版本"。
口头说"用的最新版"没有意义：改一行校验逻辑，泄漏检查结论就可能变。
所以每次交付都把涉及的脚本与规范文件的**内容哈希**列出来，
并给一个**整体代码版本号**（所有文件哈希再哈希一次），
对方拿到数据时可以核对自己手上的代码是不是同一版。

用法
----
    python scripts/code_version_manifest.py                # 默认扫 scripts/ 与 docs 规范
    python scripts/code_version_manifest.py --outdir out --include docs

退出码：0 正常
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
T = HERE.parent

# 控制台兜底
try:                                                         # pragma: no cover
    sys.stdout.reconfigure(errors="replace")
except Exception:                                            # pragma: no cover
    pass

# 交付相关的关键文件（按用途分组，便于对方快速定位）
ROLES = {
    "validate_schema.py": "schema 校验 + 表5 泄漏查重（正式闸门）",
    "validate_formal.py": "正式校验包装器（强制排除清单 + 写日志）",
    "test_formal_gate.py": "正式闸门回归测试（18 条断言）",
    "gen_variants_rule.py": "规则型变体生成器（含字段级标记）",
    "gen_swap.py": "Swap 换序生成器（默认不启用）",
    "homophone_type.py": "同音程度判定 perfect/near/far",
    "select_seeds.py": "种子挑选器（旧版逻辑，已被 seed_annotation 取代为主）",
    "seed_annotation.py": "种子分层 / 标注表 / span 校验 / 双人 κ",
    "make_manifest.py": "交付清单生成器（manifest.csv + 交付清单.md）",
    "export_norm_hashes.py": "归一化哈希清单导出（给 D 做泄漏复核）",
    "verify_seed_leakage.py": "交付前泄漏查重（种子 ∩ 表1）",
    "verify_changes.py": "整改验证日志（产出 out/验证日志.md）",
    "code_version_manifest.py": "本脚本：代码与规范版本清单",
    "gen_word_candidates.py": "整词同音候选生成",
    "curate_homophone.py": "谐音词表筛选（配额制）",
    "build_maps_graphsym.py": "建形近/leet/emoji 映射表",
    "import_split_candidates.py": "把 A 的部件候选表导入为 split_char.csv（带过滤报告）",
    "pilot_log.py": "手工试跑记录 / 统计",
    "requirements.txt": "依赖清单",
}
DOCS = {
    "数据schema与接口规范.md": "schema 与接口约定（交付主规范）",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(scripts_dir: Path, docs_dirs: list[Path]) -> list[dict]:
    items: list[dict] = []
    for p in sorted(scripts_dir.glob("*.py")) + sorted(scripts_dir.glob("*.txt")):
        items.append({
            "file": p.name, "group": "代码",
            "role": ROLES.get(p.name, ""),
            "sha256": sha256_file(p), "bytes": p.stat().st_size,
            "lines": sum(1 for _ in open(p, encoding="utf-8", errors="replace")),
        })
    for d in docs_dirs:
        for p in sorted(d.glob("*.md")):
            if p.name in DOCS:
                items.append({
                    "file": p.name, "group": "规范",
                    "role": DOCS[p.name],
                    "sha256": sha256_file(p), "bytes": p.stat().st_size,
                    "lines": sum(1 for _ in open(p, encoding="utf-8", errors="replace")),
                })
    return items


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="生成代码与规范版本清单")
    ap.add_argument("--scripts-dir", default=str(HERE))
    ap.add_argument("--docs-dir", action="append", default=None,
                    help="规范文档目录，可重复；默认取 data-build/docs 与 TrackC/docs")
    ap.add_argument("--outdir", default=str(T / "out"))
    args = ap.parse_args(argv)

    docs_dirs = [Path(d) for d in (args.docs_dir or [
        str(T.parent / "打包到小组仓库/data-build/docs"),   # 交付主规范在这
        str(T / "docs"),                                    # Track C 内部文档
    ])]
    docs_dirs = [d for d in docs_dirs if d.is_dir()]

    items = collect(Path(args.scripts_dir), docs_dirs)
    if not items:
        print("[FAIL] 没找到任何文件，检查 --scripts-dir / --docs-dir")
        return 2

    # 整体代码版本号 = 所有文件哈希再哈希（与顺序无关的稳定写法：按文件名排序）
    h = hashlib.sha256()
    for it in sorted(items, key=lambda x: (x["group"], x["file"])):
        h.update(f"{it['group']}/{it['file']}:{it['sha256']}\n".encode("utf-8"))
    code_version = "cv-" + h.hexdigest()[:12]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    md = outdir / "代码版本清单.md"
    csvp = outdir / "code_manifest.csv"

    with open(csvp, "w", encoding="utf-8-sig", newline="") as fh:
        import csv as _csv
        wr = _csv.writer(fh)
        wr.writerow(["file", "group", "role", "sha256", "bytes", "lines",
                     "code_version"])
        for it in items:
            wr.writerow([it["file"], it["group"], it["role"], it["sha256"],
                         it["bytes"], it["lines"], code_version])

    with open(md, "w", encoding="utf-8") as fh:
        fh.write("# 代码与规范版本清单\n\n")
        fh.write(f"> **整体代码版本号：`{code_version}`**\n")
        fh.write("> 由 `scripts/code_version_manifest.py` 生成；"
                 "该版本号 = 下列全部文件哈希再哈希，任一文件改动即变。\n")
        fh.write("> 用途：交付时随数据一起给出，对方可核对自己手上的代码是否为同一版。\n\n")
        fh.write(f"共 {len(items)} 个文件（"
                 f"代码 {sum(1 for i in items if i['group'] == '代码')} + "
                 f"规范 {sum(1 for i in items if i['group'] == '规范')}）。\n\n")
        for group in ("代码", "规范"):
            rows = [i for i in items if i["group"] == group]
            if not rows:
                continue
            fh.write(f"## {group}\n\n| 文件 | 用途 | 行数 | 字节 | SHA-256（前 16 位） |\n")
            fh.write("|---|---|---|---|---|\n")
            for it in rows:
                fh.write(f"| `{it['file']}` | {it['role']} | {it['lines']} | "
                         f"{it['bytes']:,} | `{it['sha256'][:16]}` |\n")
            fh.write("\n")

    print(f"[1] 整体代码版本号：{code_version}")
    print(f"[2] 人读清单：{md}")
    print(f"[3] 机读清单：{csvp}")
    print(f"[4] 文件 {len(items)} 个："
          f"代码 {sum(1 for i in items if i['group'] == '代码')} / "
          f"规范 {sum(1 for i in items if i['group'] == '规范')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
