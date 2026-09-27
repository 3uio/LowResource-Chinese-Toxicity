# -*- coding: utf-8 -*-
"""批量跑多个随机种子

用法（在项目根目录）：
    .venv\\Scripts\\python.exe 框架开发\\run_batch.py --config 框架开发\\configs\\adapter.yaml --seeds 1 2 3

做两件事：
    1. 对每个 seed 起一个**独立子进程**调用 train.py（互不污染）
    2. 把所有 seed 的指标汇成一张表 + 均值，写到
       runs\\<模式>\\<实验名>\\batch_summary.csv

⚠️ 为什么用子进程而不是在同一个进程里循环：
   同进程里反复 build 模型会留下状态（随机数生成器、adapter 注册表、内存碎片），
   子进程能保证每一个种子的运行都与「单独跑一次」完全一致。
"""
import argparse
import csv
import json
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from src.config import RunConfig, abspath   # noqa: E402

PY = sys.executable
TRAIN = os.path.join(HERE, "train.py")


def main():
    ap = argparse.ArgumentParser(description="批量跑多个随机种子")
    ap.add_argument("--config", required=True, help="yaml 配置文件路径")
    ap.add_argument("--seeds", type=int, nargs="+", required=True, help="要跑的种子，如 --seeds 1 2 3")
    ap.add_argument("--exp-name", default=None, help="覆盖配置里的实验名（所有种子共用）")
    ap.add_argument("--keep-going", action="store_true",
                    help="某个种子失败时继续跑后面的（默认失败即中止）")
    args = ap.parse_args()

    base = RunConfig.load(abspath(args.config))
    if args.exp_name:
        base.exp_name = args.exp_name

    print("=" * 72)
    print("批量运行：模式 %s ｜ 实验 %s ｜ 种子 %s" % (base.mode, base.exp_name, args.seeds))
    print("=" * 72)

    rows = []
    for seed in args.seeds:
        print("")
        print("--- seed %d ---" % seed)
        cmd = [PY, TRAIN, "--config", args.config, "--seed", str(seed)]
        if args.exp_name:
            cmd += ["--exp-name", args.exp_name]
        p = subprocess.run(cmd, cwd=PROJ, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if p.stdout:
            print(p.stdout, end="")
        if p.returncode != 0:
            if p.stderr:
                print("--- stderr ---")
                print(p.stderr[-2000:])
            print("!!! seed %d 失败（rc=%s）" % (seed, p.returncode))
            if not args.keep_going:
                return 1
            continue

        mpath = os.path.join(PROJ, base.runs_dir, base.mode, base.exp_name,
                             "seed%d" % seed, "metrics.json")
        with open(mpath, encoding="utf-8") as f:
            m = json.load(f)
        row = {
            "seed": seed,
            "train_seconds": m.get("train_seconds"),
            "train_loss": m.get("train_loss"),
        }
        for k, v in (m.get("metrics") or {}).items():
            row["dev_" + k] = v
        for k, v in (m.get("metrics_test") or {}).items():
            row["test_" + k] = v
        rows.append(row)

    if not rows:
        print("没有任何成功的运行")
        return 1

    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)

    out_dir = os.path.join(PROJ, base.runs_dir, base.mode, base.exp_name)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "batch_summary.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
        mean = {"seed": "MEAN"}
        for k in keys:
            if k == "seed":
                continue
            vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
            if vals:
                mean[k] = round(statistics.fmean(vals), 6)
        w.writerow(mean)

    print("")
    print("=" * 72)
    print("汇总（%d 个种子）已写入：%s" % (len(rows), csv_path))
    print("")
    print("  %-22s %s" % ("列", "均值"))
    for k in keys:
        if k == "seed":
            continue
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        if vals:
            print("  %-22s %.4f" % (k, statistics.fmean(vals)))
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
