# -*- coding: utf-8 -*-
"""A-5 第 2 步 · 跑通三种训练模式的最小版

目的：只验证「全参数微调 / Adapter / MAD-X」三条路在代码层面都能跑通
     （能加载 → 能建模型 → 能训练 → 能前向），并对比可训练参数量。
     ⚠️ 这里用的 16 条样本是**手写示意数据**，不是实验数据，
        本脚本不产出任何实验结论。

三种模式的组装方式：
    full_ft : 底座 + 分类头，全部参数可训练
    adapter : 冻结底座 → 插入 pfeiffer adapter → 只训这一个 adapter
    madx    : 冻结底座 → 叠加 [中文语言适配器 LA, 任务适配器 TA] → 只训 TA

用法（在项目根目录）：
    .venv\\Scripts\\python.exe 框架开发\\03_三种模式跑通.py
"""
import os
import sys
import time
import traceback

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJ, ".model_cache", "xlm-roberta-base")
LA_DIR = os.path.join(PROJ, ".model_cache", "xlm-roberta-base-zh-wiki_pfeiffer")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_03_结果.txt")

lines = []


def log(*a):
    lines.append(" ".join(str(x) for x in a))
    print(lines[-1], flush=True)


# ⚠️ 手写示意数据（仅用于跑通代码路径，不是实验数据）
SAMPLES = [
    ("这个产品真是垃圾", 1),
    ("抓虫大战，人人有责", 1),
    ("你好，今天天气不错", 0),
    ("这个服务态度很好", 0),
    ("你tmd是傻逼吧", 1),
    ("一个黑乐色都没有", 1),
    ("感谢分享，学到了", 0),
    ("就凭你的北京户口我就感觉你比我高级五个阶级", 1),
    ("这部电影我很喜欢", 0),
    ("抛开剂量谈毒性都是垃圾", 1),
    ("明天记得带伞", 0),
    ("楼主说得有道理", 0),
    ("滚出去别在这丢人", 1),
    ("请问这个怎么用", 0),
    ("脑子有病才信这个", 1),
    ("支持一下", 0),
]


def build_mode(mode):
    """按模式组装模型，返回 (model, 说明)"""
    from adapters import AutoAdapterModel, AdapterConfig
    from adapters.composition import Stack

    model = AutoAdapterModel.from_pretrained(MODEL_DIR)
    model.add_classification_head("toxic", num_labels=2)

    if mode == "full_ft":
        model.active_head = "toxic"
        note = "全部参数可训练（含底座 + 分类头）"
        setup = None

    elif mode == "adapter":
        cfg = AdapterConfig.load("pfeiffer", reduction_factor=16)
        model.add_adapter("task", config=cfg)
        model.train_adapter("task")          # 冻结其他一切，只训 task
        model.set_active_adapters("task")
        model.active_head = "toxic"
        note = "冻结底座，只训练 1 个任务 adapter（reduction_factor=16）"
        setup = "task"

    elif mode == "madx":
        cfg = AdapterConfig.load("pfeiffer", reduction_factor=16)
        model.load_adapter(LA_DIR, load_as="zh", set_active=False)   # 中文语言适配器
        model.add_adapter("task", config=cfg)                        # 任务适配器
        model.train_adapter("task")                                  # 只训任务 adapter
        model.set_active_adapters(Stack("zh", "task"))               # 语言层 + 任务层
        model.active_head = "toxic"
        note = "冻结底座 + 冻结语言 adapter（zh），只训练任务 adapter，两层 Stack 叠加"
        setup = "Stack(zh, task)"

    else:
        raise ValueError(mode)

    return model, note, setup


def run_mode(mode, tokenizer, dataset_cls):
    import torch
    from transformers import Trainer, TrainingArguments

    log("")
    log("=" * 72)
    log("模式：%s" % mode)
    log("=" * 72)

    t0 = time.time()
    model, note, setup = build_mode(mode)
    log("组装说明   :", note)
    log("激活的 adapter :", model.active_adapters)
    log("激活的 head    :", model.active_head)
    log("组装耗时   : %.1fs" % (time.time() - t0))

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log("")
    log("总参数量     : %s（%.2f 亿）" % ("{:,}".format(total), total / 1e8))
    log("可训练参数量 : %s（%.4f 亿）" % ("{:,}".format(trainable), trainable / 1e8))
    log("可训练占比   : %.4f%%" % (100.0 * trainable / total))
    log("")

    from torch.utils.data import Dataset

    class Tiny(Dataset):
        def __init__(self):
            enc = tokenizer([s for s, _ in SAMPLES], padding=True,
                            truncation=True, max_length=32, return_tensors="pt")
            self.enc = enc
            self.y = torch.tensor([y for _, y in SAMPLES])

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            item = {k: v[i] for k, v in self.enc.items()}
            item["labels"] = self.y[i]
            return item

    args = TrainingArguments(
        output_dir=os.path.join(PROJ, "框架开发", "_tmp_train", mode),
        num_train_epochs=2,
        per_device_train_batch_size=4,
        learning_rate=1e-4,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        use_cpu=True,
        # ⚠️ 必须设为 False：Trainer 默认 remove_unused_columns=True 会按模型 forward
        #    签名过滤列，而 AutoAdapterModel 的动态签名里没有 labels，
        #    结果标签被删掉 -> 报 "did not return a loss ... only logits"
        remove_unused_columns=False,
        seed=42,
        disable_tqdm=True,
    )

    trainer = Trainer(model=model, args=args, train_dataset=Tiny())

    t0 = time.time()
    result = trainer.train()
    dt = time.time() - t0
    log("训练完成   : %.1fs  （loss=%.4f）" % (dt, result.training_loss))
    log("每秒步数   : %.2f 步/秒" % (4 * 2 / max(dt, 1e-6)))

    model.eval()
    with torch.no_grad():
        enc = tokenizer([s for s, _ in SAMPLES], padding=True,
                        truncation=True, max_length=32, return_tensors="pt")
        out = model(**enc)
    log("训练后前向 : logits %s" % (tuple(out.logits.shape),))

    return {
        "mode": mode,
        "total": total,
        "trainable": trainable,
        "pct": 100.0 * trainable / total,
        "seconds": dt,
        "loss": result.training_loss,
    }


def main():
    log("=" * 72)
    log("A-5 第 2 步 · 三种训练模式跑通验证")
    log("=" * 72)
    log("主干模型   :", MODEL_DIR)
    log("语言适配器 :", LA_DIR)
    log("样本数     : %d 条（⚠️ 手写示意数据，非实验数据）" % len(SAMPLES))
    log("训练轮数   : 2 epoch，batch 4，CPU")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    results = []
    for mode in ["full_ft", "adapter", "madx"]:
        try:
            r = run_mode(mode, tokenizer, None)
            r["ok"] = True
            results.append(r)
        except Exception as e:
            log("")
            log("!!! %s 失败：%r" % (mode, e))
            log(traceback.format_exc())
            results.append({"mode": mode, "ok": False, "err": repr(e)})

    log("")
    log("=" * 72)
    log("汇总")
    log("=" * 72)
    log("%-10s %-6s %14s %16s %10s %10s" %
        ("模式", "状态", "总参数", "可训练参数", "占比", "耗时(s)"))
    for r in results:
        if r.get("ok"):
            log("%-10s %-6s %14s %16s %9.4f%% %10.1f" %
                (r["mode"], "✅", "{:,}".format(r["total"]),
                 "{:,}".format(r["trainable"]), r["pct"], r["seconds"]))
        else:
            log("%-10s %-6s   %s" % (r["mode"], "❌", r.get("err")))

    ok = sum(1 for r in results if r.get("ok"))
    log("")
    log(">>> %d/3 种模式跑通" % ok)


try:
    main()
except Exception as e:
    log("顶层失败：%r" % e)
    log(traceback.format_exc())

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
