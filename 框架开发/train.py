# -*- coding: utf-8 -*-
"""A-5 训练框架 · 入口

用法：
    .venv\\Scripts\\python.exe 框架开发\\train.py --config 框架开发\\configs\\adapter.yaml

三种模式用同一份代码，靠配置切换：
    mode: full_ft   → 全参数微调
    mode: adapter   → 冻底座，只训一个任务适配器
    mode: madx      → 冻底座 + 叠语言适配器，只训任务适配器

结果自动落到 runs/<模式>/<实验名>/seed<种子>/。
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from src.config import RunConfig, abspath   # noqa: E402
from src.runner import run                  # noqa: E402


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="可配置训练框架（全参数微调 / Adapter / MAD-X）")
    ap.add_argument("--config", required=True, help="yaml 配置文件路径")
    ap.add_argument("--exp-name", default=None, help="覆盖配置里的实验名")
    ap.add_argument("--seed", type=int, default=None, help="覆盖配置里的随机种子")
    ap.add_argument("--checkpoint", help="加载已训练完整模型目录")
    ap.add_argument("--eval-only", action="store_true", help="只评估")
    args = ap.parse_args()

    cfg = RunConfig.load(abspath(args.config), validate=False)
    if args.exp_name:
        cfg.exp_name = args.exp_name
    if args.seed is not None:
        cfg.seed = args.seed
    if args.checkpoint:
        cfg.model_checkpoint = args.checkpoint
    if args.eval_only:
        cfg.eval_only = True
    cfg.validate()

    summary = run(cfg)

    print("")
    print("=" * 72)
    print("完成：mode=%s  exp=%s  seed=%d%s"
          % (summary["mode"], summary["exp_name"], summary["seed"],
             "（eval_only，未训练）" if summary.get("eval_only") else ""))
    print("可训练参数占比：%.4f%%" % summary["params"]["trainable_pct"])
    if summary["metrics"]:
        print("主评估集指标：", {k: round(v, 4) for k, v in summary["metrics"].items()})
    if summary.get("metrics_test"):
        print("test 段指标  ：", {k: round(v, 4) for k, v in summary["metrics_test"].items()})
    print("结果目录：%s" % cfg.run_dir())
    print("=" * 72)


if __name__ == "__main__":
    main()
