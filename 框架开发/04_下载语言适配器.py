# -*- coding: utf-8 -*-
"""A-5 第 4 步 · 下载两个语言适配器（中/英），并验证它们能同时挂载与切换

为什么需要两个：
    TLR（Target Language-Ready adapters）的做法是「训练任务适配器时，
    让源语言 LA 和目标语言 LA 轮流上场」。
    本项目：源语言 = 英语（任务数据是英文 Jigsaw），目标语言 = 中文。
    所以两个都要有。

本脚本做三件事：
    1. 下载/补齐 中文 LA（zh）与 英语 LA（en）
    2. 验证两个 LA 能同时 load 到一个模型上（load_as 区分）
    3. 验证 Stack(en, task) 与 Stack(zh, task) 能来回切换，且输出确实不同
       （这是 TLR 能实现的前提）

用法（在项目根目录）：
    .venv\\Scripts\\python.exe 框架开发\\04_下载语言适配器.py
"""
import os
import time
import traceback

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(PROJ, ".model_cache")
MODEL_DIR = os.path.join(CACHE, "xlm-roberta-base")

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_04_下载结果.txt")
lines = []


def log(*a):
    lines.append(" ".join(str(x) for x in a))
    print(lines[-1], flush=True)


# 别名 -> (AdapterHub 仓库名, 本地目录名)
# 本地目录名与仓库名末段保持一致，便于直接对照
TARGETS = {
    "zh": ("AdapterHub/xlm-roberta-base-zh-wiki_pfeiffer",
           "xlm-roberta-base-zh-wiki_pfeiffer"),
    "en": ("AdapterHub/xlm-roberta-base-en-wiki_pfeiffer",
           "xlm-roberta-base-en-wiki_pfeiffer"),
}

KEY_FILES = ["adapter_config.json", "pytorch_adapter.bin"]

SAMPLES = [
    "这个产品真是垃圾",
    "抓虫大战，人人有责",
    "你好，今天天气不错",
]


def download_one(repo_id, local_dir):
    """下载到 local_dir，并对 0 字节的关键文件做自愈"""
    from huggingface_hub import hf_hub_download, snapshot_download

    if os.path.exists(os.path.join(local_dir, "adapter_config.json")):
        log("      本地已存在，跳过下载")
    else:
        t0 = time.time()
        snapshot_download(repo_id, local_dir=local_dir)
        log("      下载完成（%.1fs）" % (time.time() - t0))

    # ⚠️ 自愈：旧缓存里的坏符号链接（0 字节）会被当成「文件已存在」直接复制过来
    for fn in KEY_FILES:
        p = os.path.join(local_dir, fn)
        size = os.path.getsize(p) if os.path.exists(p) else -1
        if size <= 0:
            log("      ⚠️ %s 是 %d 字节 -> 强制重下" % (fn, size))
            hf_hub_download(repo_id, fn, local_dir=local_dir, force_download=True)
            log("         -> %d 字节 ✓" % os.path.getsize(p))


def main():
    log("=" * 70)
    log("A-5 第 4 步 · 下载语言适配器（zh / en）")
    log("=" * 70)
    log("HF_ENDPOINT :", os.environ.get("HF_ENDPOINT"))
    log("落地目录    :", CACHE)
    log("")

    dirs = {}
    for alias, (repo_id, dir_name) in TARGETS.items():
        local_dir = os.path.join(CACHE, dir_name)
        dirs[alias] = local_dir
        log("[下载] %s  %s" % (alias, repo_id))
        try:
            download_one(repo_id, local_dir)
            files = sorted(os.listdir(local_dir))
            total = sum(os.path.getsize(os.path.join(local_dir, f))
                        for f in files if os.path.isfile(os.path.join(local_dir, f)))
            log("      OK，%d 个文件 / %.2f MB" % (len(files), total / 1e6))
            log("      内容:", files)
        except Exception as e:
            log("      ✗ 失败：%r" % e)
            dirs.pop(alias, None)
        log("")

    if len(dirs) < 2:
        log("!!! 两个语言适配器没有齐全，后面的挂载验证跳过")
        return

    # ---------- 验证：两个 LA 同时挂载 + 来回切换 ----------
    log("=" * 70)
    log("验证：两个语言适配器能否同时挂载并切换")
    log("=" * 70)

    from adapters import AutoAdapterModel, AdapterConfig
    from adapters.composition import Stack
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoAdapterModel.from_pretrained(MODEL_DIR)
    log("主干加载 OK，参数量 %.2f 亿" % (sum(p.numel() for p in model.parameters()) / 1e8))

    for alias, local_dir in dirs.items():
        name = model.load_adapter(local_dir, load_as=alias, set_active=False)
        log("加载 %s -> 名称 %r" % (alias, name))

    model.add_classification_head("toxic", num_labels=2)
    model.add_adapter("task", config=AdapterConfig.load("pfeiffer", reduction_factor=16))
    model.train_adapter("task")
    model.active_head = "toxic"
    log("任务适配器 task 已建，只训练它")

    enc = tok(SAMPLES, padding=True, truncation=True, max_length=64, return_tensors="pt")

    outs = {}
    for alias in ("en", "zh"):
        model.set_active_adapters(Stack(alias, "task"))
        with_dropout_off = model.eval()
        import torch
        with torch.no_grad():
            out = with_dropout_off(**enc)
        outs[alias] = out.logits.detach().numpy()
        log("Stack(%-2s, task) 前向 OK  logits %s  active=%s"
            % (alias, tuple(out.logits.shape), model.active_adapters))

    same = (outs["en"] == outs["zh"]).all()
    log("")
    log("两种叠加的 logits 是否完全相同 :", bool(same))
    log("en 首个样本 logits :", outs["en"][0].round(4).tolist())
    log("zh 首个样本 logits :", outs["zh"][0].round(4).tolist())
    if same:
        log("⚠️ 完全相同 —— 说明切换没生效，TLR 无法实现")
    else:
        log("✅ 不同 —— 切换确实改变了前向路径，TLR 的前提满足")

    # 同时确认：语言适配器挂着但没有被训练
    # ⚠️ 必须用带点的精确模式（.adapters.en.），不能写 "en" in 参数名 ——
    #    "encoder" / "attention" 里都含 "en"，会误判。
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    n_all = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lang_train = [n for n in trainable if ".adapters.en." in n or ".adapters.zh." in n]
    task_train = [n for n in trainable if ".adapters.task." in n]
    log("")
    log("总参数      %d" % n_all)
    log("可训练参数  %d（%.4f%%）" % (n_train, 100.0 * n_train / n_all))
    log("可训练参数里属于语言层(en/zh)的条数 :", len(lang_train), "（应为 0 = 语言层已冻结）")
    log("可训练参数里属于任务层(task)的条数  :", len(task_train))
    log("可训练参数示例（前 3 个）:")
    for n in trainable[:3]:
        log("    " + n)


try:
    main()
except Exception as e:
    log("")
    log("!!! 失败：%r" % e)
    log(traceback.format_exc())

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
