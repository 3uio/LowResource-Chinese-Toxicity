# -*- coding: utf-8 -*-
"""训练 / 评估 / 落盘 —— 所有模式共用同一条训练路径

结果目录结构（B 建议的格式）：
    runs/<模式>/<实验名>/seed<种子>/
        config.json        本次实验的完整配置（原样存档）
        metrics.json       指标（Macro-F1 / Toxic Recall / FPR / ...）
        predictions.csv    逐条预测结果（便于事后看错例）
        train.log          运行日志
"""
import csv
import json
import os
import random
import sys
import time

import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(eval_pred):
    """指标口径按 V2 评估体系：Macro-F1 / Toxic Recall / FPR

    约定 label=1 表示「有毒」，因此：
        Toxic Recall = 有毒样本被找出来的比例（越高越好，漏检少）
        FPR          = 无毒样本被误判成有毒的比例（越低越好，误伤少）
    """
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(labels, preds)),
        "toxic_recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
    }


def evaluate_split(trainer, dataset, split_name, run_dir, log,
                   prefix, pred_name):
    """在某个数据集上评估，并写出逐条预测

    dev 与 test 各调用一次，互不干扰：
        - dev  → prefix="eval"、predictions.csv        （主评估集）
        - test → prefix="test"、predictions_test.csv   （外部基准，可选）

    返回该数据集上的指标 dict。
    """
    out = trainer.evaluate(dataset, metric_key_prefix=prefix)
    metrics = {k[len(prefix) + 1:]: v for k, v in out.items()
               if k.startswith(prefix + "_") and isinstance(v, (int, float))}
    # 去掉计时类字段，只留真正的指标
    # （model_preparation_time 只在没训练过的 model 上出现，train 之后不会）
    for drop in ("runtime", "samples_per_second", "steps_per_second",
                 "jit_compile_sec", "model_preparation_time"):
        metrics.pop(drop, None)

    log("      [%s] %d 条" % (split_name, len(dataset)))
    for k in ("loss", "macro_f1", "accuracy", "toxic_recall", "fpr"):
        if k in metrics:
            log("        %-14s %.4f" % (k, metrics[k]))

    # 逐条预测
    pred_out = trainer.predict(dataset, metric_key_prefix=prefix)
    raw = pred_out.predictions
    # ⚠️ adapters 系的模型有时会把多个输出打包返回，predictions 不一定是纯 logits 数组，
    #    直接 np.argmax 会报 "inhomogeneous shape"。这里取第一个元素再转 ndarray。
    if isinstance(raw, (tuple, list)):
        raw = raw[0]
    preds = np.argmax(np.asarray(raw), axis=-1)
    rows = list(dataset.rows)
    # ⚠️ 两个坑：① Trainer 有时把 label_ids 返回成 None；② 返回的可能是标量而不是数组。
    #    统一拍平成一维 list，避免后面报 "'numpy.int64' object is not iterable"。
    labels = pred_out.label_ids
    if labels is None:
        labels = [int(r["label"]) for r in rows]
    else:
        labels = [int(x) for x in np.asarray(labels).reshape(-1)]
    preds = [int(x) for x in np.asarray(preds).reshape(-1)]
    n = min(len(rows), len(labels), len(preds))
    with open(os.path.join(run_dir, pred_name), "w",
              encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "text", "label", "pred", "correct"])
        for i in range(n):
            r, y, yhat = rows[i], labels[i], preds[i]
            w.writerow([i, r.get("text", ""), y, yhat, int(y == yhat)])
    log("        逐条预测已写入 %s（%d 条）" % (pred_name, n))
    return metrics


def run(cfg, logger=print):
    import torch
    from transformers import AutoTokenizer, Trainer, TrainingArguments

    from .builder import TLRSwitchCallback, active_summary, build_model, param_summary
    from .data import load_datasets

    t_start = time.time()
    run_dir = cfg.run_dir()
    os.makedirs(run_dir, exist_ok=True)

    log_lines = []

    def log(*a):
        msg = " ".join(str(x) for x in a)
        log_lines.append(msg)
        logger(msg)

    log("=" * 72)
    log("实验名   :", cfg.exp_name)
    log("模式     :", cfg.mode)
    log("种子     :", cfg.seed)
    log("结果目录 :", run_dir)
    log("=" * 72)

    set_seed(cfg.seed)

    # ---- 1) 分词器与数据 ----
    log("")
    log("[1/5] 加载分词器与数据")
    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone_path)
    ds = load_datasets(cfg, tokenizer)
    for split, d in ds.items():
        log("      %-5s : %s" % (split, "%d 条" % len(d) if d is not None else "（未提供）"))

    # ---- 2) 组装模型 ----
    log("")
    log("[2/5] 组装模型")
    res = build_model(cfg)
    model = res.model
    log("      " + res.note)
    ps = param_summary(model)
    asum = active_summary(model)
    log("      总参数      : {:,}".format(ps["total"]))
    log("      可训练参数  : {:,}（{:.4f}%）".format(ps["trainable"], ps["trainable_pct"]))
    log("      激活 adapter:", asum["active_adapters"])
    log("      激活 head   :", asum["active_head"])

    # ---- 3) 训练 ----
    log("")
    if cfg.eval_only:
        log("[3/5] 跳过训练（eval_only=true：只推理，不更新任何参数）")
    else:
        log("[3/5] 训练")
    args = TrainingArguments(
        output_dir=os.path.join(run_dir, "_trainer"),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        use_cpu=cfg.cpu,
        # ⚠️ 必须为 False：默认 True 会按模型签名过滤列，而 AutoAdapterModel 的
        #    动态签名不含 labels，标签会被静默删掉 → 报 did not return a loss
        remove_unused_columns=False,
        # ⚠️ 同一个根因的另一面：AutoAdapterModel 的动态签名让 Trainer 推断不出标签列名，
        #    不显式指定它就认为「没有标签」→ 评估时不算 eval_loss、也拿不到 label_ids
        #    （表现为 metrics 为空、predictions.csv 只有 1 行）。必须显式告诉它。
        label_names=["labels"],
        seed=cfg.seed,
        disable_tqdm=True,
    )

    eval_ds = ds["dev"] if ds["dev"] is not None else ds["test"]

    # ---- TLR 交替（只在 madx + tlr=true 时出现）----
    callbacks = []
    tlr_cb = None
    if res.tlr_plan is not None:
        tlr_cb = TLRSwitchCallback(model, res.tlr_plan, log=log)
        callbacks.append(tlr_cb)
        log("      ⚠️ TLR 已启用：训练过程中每一步都会切换垫在下面的语言适配器")

    trainer = Trainer(
        model=model,
        args=args,
        # eval_only 时不传训练集（Trainer 允许 train_dataset=None，只要有 eval_dataset）
        train_dataset=None if cfg.eval_only else ds["train"],
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics if eval_ds is not None else None,
        callbacks=callbacks,
    )

    train_seconds = 0.0
    train_loss = None
    if cfg.eval_only:
        log("      按配置跳过训练")
    else:
        t0 = time.time()
        train_result = trainer.train()
        train_seconds = time.time() - t0
        train_loss = float(train_result.training_loss)
        log("      训练完成：%.1fs，train_loss=%.4f" % (train_seconds, train_loss))

    tlr_info = None
    if tlr_cb is not None:
        for line in tlr_cb.report():
            log("      " + line)
        tlr_info = {
            "enabled": True,
            "cycle": res.tlr_plan.la_names,
            "steps": len(tlr_cb.observed),
            "counts": tlr_cb.counts,
            "sequence_head": [n for _, n in tlr_cb.observed[:12]],
        }

    # ⚠️ 关键：训练结束后、评估/推理之前，必须切回目标语言 LA。
    #    TLR 训练时最后一步用的是哪层 LA 取决于步数奇偶，不能拿它去评估。
    if res.eval_adapters is not None:
        model.set_active_adapters(res.eval_adapters)
        log("      推理前已切回目标语言 LA：active_adapters = %s" % model.active_adapters)

    # ---- 4) 评估（dev 与 test 各评一遍，结果分开存）----
    log("")
    log("[4/5] 评估")
    metrics = {}
    metrics_test = {}
    if eval_ds is None:
        log("      （没有提供 dev / test，跳过评估）")
    else:
        main_name = "dev" if ds["dev"] is not None else "test"
        metrics = evaluate_split(trainer, eval_ds, main_name, run_dir, log,
                                 prefix="eval", pred_name="predictions.csv")
        # test 段：仅当提供了 test、且它不是主评估集时再多跑一遍
        if ds["test"] is not None and main_name != "test":
            metrics_test = evaluate_split(trainer, ds["test"], "test", run_dir, log,
                                          prefix="test", pred_name="predictions_test.csv")

    # ---- 5) 落盘 ----
    log("")
    log("[5/5] 落盘")
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, ensure_ascii=False, indent=2)

    summary = {
        "exp_name": cfg.exp_name,
        "mode": cfg.mode,
        "seed": cfg.seed,
        "eval_only": cfg.eval_only,
        "tlr": tlr_info,
        "params": ps,
        "active": asum,
        "train_seconds": round(train_seconds, 2),
        "train_loss": train_loss,          # eval_only=true 时为 null
        "metrics": metrics,                # 主评估集（有 dev 用 dev，否则用 test）
        "metrics_test": metrics_test,      # test 段；未提供 test_file 时为 {}
        "total_seconds": round(time.time() - t_start, 2),
    }
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log("      config.json            ✓")
    log("      metrics.json           ✓")
    if eval_ds is not None:
        log("      predictions.csv        ✓")
    if metrics_test:
        log("      predictions_test.csv   ✓")
    log("")

    with open(os.path.join(run_dir, "train.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")
    log("      train.log        ✓")

    return summary
