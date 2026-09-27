# -*- coding: utf-8 -*-
"""A-5 第 1 步 · 环境与库可用性验证（只做 import 检查，不下载模型）

用法（在项目根目录）：
    .venv\Scripts\python.exe 框架开发\01_环境验证.py
"""
import os
import sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_01_检查结果.txt")
lines = []


def log(*a):
    lines.append(" ".join(str(x) for x in a))


log("=" * 68)
log("A-5 第 1 步 · 环境与库验证")
log("=" * 68)
log("Python     :", sys.version.replace("\n", " "))
log("可执行文件 :", sys.executable)
log("虚拟环境   :", sys.prefix)
log("")

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
log("HF_ENDPOINT =", os.environ.get("HF_ENDPOINT"), "（国内镜像，huggingface.co 直连不通）")
log("")

log("--- 1) 基础库 import ---")
for name in ["torch", "transformers", "adapters", "huggingface_hub",
             "tokenizers", "numpy", "safetensors"]:
    try:
        m = __import__(name)
        log("[OK]   %-18s %s" % (name, getattr(m, "__version__", "?")))
    except Exception as e:
        log("[FAIL] %-18s %r" % (name, e))
log("")

log("--- 2) adapter-transformers 关键 API ---")
try:
    from adapters import AutoAdapterModel
    log("[OK]   adapters.AutoAdapterModel")
except Exception as e:
    log("[FAIL] AutoAdapterModel: %r" % e)

try:
    from adapters.composition import Stack
    log("[OK]   adapters.composition.Stack  （叠加语言/任务 adapter 用）")
except Exception as e:
    log("[FAIL] Stack: %r" % e)

for nm in ["PfeifferConfig", "AdapterConfig", "HoulsbyConfig", "ParallelConfig"]:
    try:
        getattr(__import__("adapters", fromlist=[nm]), nm)
        log("[OK]   adapters.%s" % nm)
    except Exception as e:
        log("[--]   adapters.%s 不可用（%s）" % (nm, type(e).__name__))
log("")

log("--- 3) torch 后端 ---")
try:
    import torch
    log("torch.cuda.is_available() =", torch.cuda.is_available())
    log("说明：本环境装的是 CPU 版 torch，False 属正常（仅用于验证代码）")
    log("torch.get_num_threads()   =", torch.get_num_threads())
except Exception as e:
    log("torch 检查失败: %r" % e)

text = "\n".join(lines)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(text + "\n")
print(text)
