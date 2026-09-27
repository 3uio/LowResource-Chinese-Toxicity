# -*- coding: utf-8 -*-
"""A-5 第 1 步 · 下载主干模型 + 中文语言 adapter，并跑通推理

⚠️ 为什么要用 local_dir 下载，而不是直接 from_pretrained("名称")：
   本项目环境实测 —— huggingface_hub 默认把 snapshots/ 下的文件做成「符号链接」，
   而 Windows 沙箱下符号链接创建会失败，留下 0 字节文件，导致读取时报
   JSONDecodeError。改用 local_dir 参数后，文件是真实复制的，问题消失。

涉及的两个资产：
   主干模型        : xlm-roberta-base（约 1.1 GB，只下 safetensors，跳过 tf/flax/pytorch_model.bin）
   中文语言 adapter: AdapterHub/xlm-roberta-base-zh-wiki_pfeiffer（很小，约 1 MB）

用法（在项目根目录）：
    .venv\\Scripts\\python.exe 框架开发\\02_下载与推理验证.py
"""
import os
import sys
import time
import traceback

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(PROJ, ".model_cache")
MODEL = "xlm-roberta-base"
LA_ZH = "AdapterHub/xlm-roberta-base-zh-wiki_pfeiffer"
MODEL_DIR = os.path.join(CACHE, "xlm-roberta-base")
LA_DIR = os.path.join(CACHE, "xlm-roberta-base-zh-wiki_pfeiffer")

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# ⚠️ 必须禁用 Xet 存储后端：实测 hf-mirror 会把大文件重定向到
#    cas-bridge.xethub.hf.co，该域名在本机被代理拦（Read timed out）。
#    置 1 后回退到传统 CDN 下载路径。
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_02_验证结果.txt")
lines = []


def log(*a):
    lines.append(" ".join(str(x) for x in a))
    print(lines[-1], flush=True)


SAMPLES = [
    "这个产品真是垃圾",
    "抓虫大战，人人有责",
    "你好，今天天气不错",
    "就凭你的北京户口我就感觉你比我高级五个阶级",
]


def main():
    log("=" * 70)
    log("A-5 第 1 步 · 下载与推理验证")
    log("=" * 70)
    log("HF_ENDPOINT      :", os.environ.get("HF_ENDPOINT"))
    log("主干模型         :", MODEL)
    log("中文语言 adapter :", LA_ZH)
    log("落地目录         :", CACHE)
    log("")

    from huggingface_hub import snapshot_download

    log("[1/6] 下载主干模型 xlm-roberta-base ...")
    log("      （只取 *.json / *.model / *.safetensors / *.txt，跳过 tf/flax/旧 bin 格式）")
    t0 = time.time()
    snapshot_download(
        MODEL,
        local_dir=MODEL_DIR,
        allow_patterns=["*.json", "*.model", "*.safetensors", "*.txt"],
        max_workers=4,
    )
    log("      完成（%.1fs）" % (time.time() - t0))

    log("")
    log("[2/6] 下载中文语言 adapter ...")
    t0 = time.time()
    snapshot_download(LA_ZH, local_dir=LA_DIR)
    log("      完成（%.1fs）" % (time.time() - t0))

    # ⚠️ 自愈：旧缓存里有坏掉的符号链接（0 字节，见 .workbuddy/memory 记录），
    #    snapshot_download 可能直接从缓存把这个空文件复制过来，
    #    导致后面读 json 时报 JSONDecodeError。这里对 0 字节的关键文件强制重下。
    from huggingface_hub import hf_hub_download

    def ensure(repo_id, local_dir, files):
        for fn in files:
            p = os.path.join(local_dir, fn)
            size = os.path.getsize(p) if os.path.exists(p) else -1
            if size <= 0:
                log("      ⚠️ %s 是 %d 字节 -> 强制重下" % (fn, size))
                hf_hub_download(repo_id, fn, local_dir=local_dir, force_download=True)
                log("         -> %d 字节 ✓" % os.path.getsize(p))

    log("      检查关键文件完整性 ...")
    ensure(MODEL, MODEL_DIR, ["config.json", "tokenizer_config.json",
                              "tokenizer.json", "sentencepiece.bpe.model"])
    ensure(LA_ZH, LA_DIR, ["adapter_config.json", "pytorch_adapter.bin"])
    log("      主干目录  :", sorted(os.listdir(MODEL_DIR)))
    log("      adapter 目录:", sorted(os.listdir(LA_DIR)))

    log("")
    log("[3/6] 从本地目录加载主干 ...")
    t0 = time.time()
    from adapters import AutoAdapterModel
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoAdapterModel.from_pretrained(MODEL_DIR)
    n_param = sum(p.numel() for p in model.parameters())
    log("      完成（%.1fs），参数量 %.2f 亿" % (time.time() - t0, n_param / 1e8))

    log("")
    log("[4/6] 加载中文语言 adapter ...")
    t0 = time.time()
    la_name = model.load_adapter(LA_DIR)
    model.set_active_adapters(la_name)
    log("      完成（%.1fs），名称 = %s" % (time.time() - t0, la_name))
    log("      active_adapters =", model.active_adapters)
    log("      active_head     =", model.active_head)

    log("")
    log("[5/6] 前向推理 ...")
    enc = tok(SAMPLES, padding=True, truncation=True, max_length=64, return_tensors="pt")
    # ⚠️ 取句子表示要用 base_model，不要直接 model(**enc)
    #    因为 AutoAdapterModel 会按 config.architectures 自动挂上 MaskedLM 头，
    #    直接 forward 得到的是 MaskedLMOutput（只有 logits，没有 last_hidden_state）
    base_out = model.base_model(**enc)
    log("      input_ids        :", tuple(enc["input_ids"].shape))
    log("      last_hidden_state:", tuple(base_out.last_hidden_state.shape))
    log("      pooler_output    :", tuple(base_out.pooler_output.shape))

    log("")
    log("[6/6] 加分类头（B 的实验形式）+ 再过一遍 ...")
    model.add_classification_head("toxic", num_labels=2)
    model.set_active_adapters(la_name)
    # ⚠️ 没有 set_active_head() 方法，直接给属性赋值
    model.active_head = "toxic"
    out2 = model(**enc)
    log("      logits 形状:", tuple(out2.logits.shape))
    log("      logits 数值:", out2.logits.detach().numpy().round(4).tolist())

    log("")
    log("=" * 70)
    log(">>> 全部通过：模型下载 OK / 中文 LA 加载 OK / 前向 OK / 分类头 OK")
    log("=" * 70)


try:
    main()
except Exception as e:
    log("")
    log("!!! 失败：%r" % e)
    log(traceback.format_exc())

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
