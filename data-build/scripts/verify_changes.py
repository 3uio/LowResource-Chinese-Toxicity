"""改完代码后的验证日志（B 要求的"验证日志"）。

对每一项改动，跑一次实证并记录结果。
输出可直接贴给 B/D 作为验证记录。
"""
import csv
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Windows 控制台默认 GBK，遇到 emoji / 生僻符号会 UnicodeEncodeError。
# 统一降级为"打不出的字符显示成 ?"，避免因为**数据里**的字符把脚本崩掉。
try:                                                     # pragma: no cover
    sys.stdout.reconfigure(errors="replace")
except Exception:                                        # pragma: no cover
    pass

# 本脚本在 scripts/ 下；命令行里的相对路径是相对**仓库根**（gen_variants_rule.py 等
# 都按仓库根解析），所以这里显式取上一级。
T = Path(__file__).resolve().parent.parent
LOG = []


def run(cmd, note=""):
    """跑子进程并抓输出。

    必须显式把子进程的 stdout 编码定成 UTF-8：Windows 上子进程写管道时
    默认用本地代码页（GBK），父进程按 UTF-8 解码 → 中文全变成 `?`，
    于是所有"输出里是否包含某句话"的判据都会**假失败**。
    """
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, *cmd], cwd=str(T),
                       capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env)
    return r.returncode, r.stdout + r.stderr


def sec(title):
    LOG.append("")
    LOG.append("=" * 72)
    LOG.append(title)
    LOG.append("=" * 72)


def item(name, ok, detail=""):
    LOG.append(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if isinstance(detail, (list, tuple)):
        detail = "\n".join(str(x) for x in detail)
    if detail:
        for ln in str(detail).strip().splitlines():
            LOG.append(f"         {ln}")


LOG.append("# C 侧代码整改验证日志")
LOG.append(f"> 生成时间：{datetime.now():%Y-%m-%d %H:%M}")
LOG.append("> 对应 B 的四条意见与 D 的字段要求（见 docs/11 §三）")

# ---------------------------------------------------------------- 1
sec("1. homophone_type 判定模块自检（D 要求'具体判定规则和样例'）")
rc, out = run(["scripts/homophone_type.py"])
item("14 条样例全部符合预期", rc == 0, out.strip().splitlines()[-1] if out else "")

# ---------------------------------------------------------------- 2
sec("2. 校验器：--require-exclusion 阻断（B 第 2 条）")
rc, out = run(["scripts/validate_schema.py",
               "--seeds", "examples/pilot_targets.csv",
               "--require-exclusion"])
item("缺 --exclusion 时退出码=2（阻断）", rc == 2,
     [l for l in out.splitlines() if "FAIL" in l][:1])

rc, out = run(["scripts/validate_schema.py",
               "--seeds", "examples/pilot_targets.csv",
               "--exclusion", "data/templates/table5_exclusion_list.csv",
               "--require-exclusion"])
item("排除清单为空时退出码=1（阻断）", rc == 1,
     [l for l in out.splitlines() if "ERROR" in l][:1])

rc, out = run(["scripts/validate_schema.py",
               "--seeds", "examples/pilot_targets.csv",
               "--exclusion", "data/templates/table5_exclusion_list.csv"])
item("未开 --require-exclusion 时仅 warn（本地联调）", rc == 0,
     [l for l in out.splitlines() if "结果" in l][:1])

# ---------------------------------------------------------------- 3
sec("3. 生成器：内置占位映射隔离（B 第 4 条）")

# 判据说明（2026-10-11 修正）：
#   原先这里的判据写成"默认产出 0 条"，**这个是错的**。实测：
#   `PY_Full`/`PY_Abbr` 走 pypinyin、`Sym_Separator` 不需要词表，
#   这三类在没有映射表时**照常产出**；被跳过的是**依赖映射表**的
#   Homo / Graph_Similar / Graph_Split / Sym_Leet / Emoji_Sub。
#   所以真正要验的不变量是：**默认不加载占位表 → 依赖映射的类必须为 0，
#   且必须打印「占位表默认不启用」**；加了 --allow-placeholder 才允许它们有产出。

MAPPING_TYPES = {"Homo", "Graph_Similar", "Graph_Split", "Sym_Leet", "Emoji_Sub"}


def count_types(path):
    p = T / path
    if not p.exists():
        return {}, 0
    rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
    by = {}
    for r in rows:
        by[r.get("attack_type", "")] = by.get(r.get("attack_type", ""), 0) + 1
    return by, len(rows)


rc, out = run(["scripts/gen_variants_rule.py",
               "--seeds", "examples/demo_seeds.csv",
               "--out", "out/_vl_a.csv", "--stats-out", "out/_vl_as.csv"])
by_off, n_off = count_types("out/_vl_a.csv")
n_map_off = sum(by_off.get(t, 0) for t in MAPPING_TYPES)
item("默认不加载占位表：依赖映射的类产出为 0", n_map_off == 0,
     f"总产出 {n_off} 条；依赖映射类 {n_map_off} 条；"
     f"各类：{ {k: v for k, v in sorted(by_off.items())} }")
item("默认运行会明确报告「占位表默认不启用」", "占位表默认不启用" in out,
     [l for l in out.splitlines() if "warn" in l.lower()][:1])

rc, out = run(["scripts/gen_variants_rule.py",
               "--seeds", "examples/demo_seeds.csv",
               "--out", "out/_vl_b.csv", "--stats-out", "out/_vl_bs.csv",
               "--allow-placeholder"])
by_on, n_on = count_types("out/_vl_b.csv")
n_map_on = sum(by_on.get(t, 0) for t in MAPPING_TYPES)
item("显式 --allow-placeholder 才启用（有 WARN）", "不得用于正式交付" in out,
     [l for l in out.splitlines() if "WARN" in l][:1])
item("启用占位表后，依赖映射的类确实有产出（反证上面那条不是假通过）",
     n_map_on > 0, f"依赖映射类 {n_map_on} 条（关闭时 {n_map_off} 条）")

# ---------------------------------------------------------------- 4
sec("4. 新字段：semantic_score_01 / semantic_score_restored / homophone_type")
p = T / "out/_vl_b.csv"
if p.exists():
    rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
    cols = list(rows[0].keys()) if rows else []
    item("semantic_score_normalized 已改名 semantic_score_01",
         "semantic_score_01" in cols and "semantic_score_normalized" not in cols,
         f"表头末 4 列：{cols[-4:]}")
    item("semantic_score_restored 已加入", "semantic_score_restored" in cols)
    item("homophone_type 已加入", "homophone_type" in cols)
    item("substitution_unit 已加入", "substitution_unit" in cols)

# ---------------------------------------------------------------- 5
sec("5. 真实数据端到端（200 条种子）")
rc, out = run(["scripts/gen_variants_rule.py",
               "--seeds", "out/seed_candidates.csv",
               "--out", "out/_vl_real.csv", "--stats-out", "out/_vl_reals.csv",
               "--rules-dir", "data/rules"])
item("生成成功", rc == 0,
     [l for l in out.splitlines() if "写出变体" in l][:1])

p = T / "out/_vl_real.csv"
if p.exists():
    rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
    homo = [r for r in rows if r["attack_type"] == "Homo"]
    from collections import Counter
    dist = Counter(r["homophone_type"] for r in homo)
    item("Homo 有 homophone_type", len(homo) > 0 and all(r["homophone_type"] for r in homo),
         f"Homo {len(homo)} 条，分布 {dict(dist)}")
    other = [r for r in rows if r["attack_type"] != "Homo"]
    item("非 Homo 的 homophone_type 为空", sum(1 for r in other if r["homophone_type"]) == 0)

rc, out = run(["scripts/validate_schema.py",
               "--seeds", "out/seed_candidates.csv",
               "--variants", "out/_vl_real.csv",
               "--exclusion", "data/templates/table5_exclusion_list.csv"])
item("schema 校验通过", rc == 0,
     [l for l in out.splitlines() if "结果" in l][:1])

# ---------------------------------------------------------------- 清理
for f in ("out/_vl_a.csv", "out/_vl_as.csv", "out/_vl_b.csv", "out/_vl_bs.csv",
          "out/_vl_real.csv", "out/_vl_reals.csv"):
    (T / f).unlink(missing_ok=True)

text = "\n".join(LOG)
(T / "out" / "验证日志.md").write_text(text, encoding="utf-8")
print(text)
