# -*- coding: utf-8 -*-
"""配置定义与加载

设计原则：一份 yaml 配置决定整个实验怎么跑，改配置不改代码。
"""
import os
from dataclasses import dataclass, asdict, fields

import yaml

# 项目根目录（本文件位于 框架开发/src/，往上三级）
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def abspath(p):
    """把配置里的相对路径按「项目根目录」解析；空值返回空串"""
    if not p:
        return ""
    if os.path.isabs(p):
        return p
    return os.path.join(PROJ_ROOT, p)


@dataclass
class RunConfig:
    """一次实验的完整配置"""

    # ---------- 实验标识 ----------
    exp_name: str = "demo"
    # full_ft | adapter | madx    ← 改这一行就切换训练方式
    mode: str = "adapter"

    # ---------- 模型 ----------
    backbone: str = ".model_cache/xlm-roberta-base"
    # madx 模式：目标语言适配器（LA）—— 「要迁移到的那门语言」，本项目＝中文
    language_adapter: str = ".model_cache/xlm-roberta-base-zh-wiki_pfeiffer"
    # madx + tlr 模式：源语言适配器（LA）—— 「任务数据所用的语言」，本项目＝英语
    #   （任务数据是英文 Jigsaw，所以训练时任务适配器是先在英语底座上调出来的）
    source_language_adapter: str = ".model_cache/xlm-roberta-base-en-wiki_pfeiffer"
    # TLR（Target Language-Ready adapters，目标语言就绪适配器）
    #   问题：原版 MAD-X 训练任务适配器时只让它见源语言 LA，推理时才换成目标语言 LA
    #         → 两者从没配合过，效果掉，且只在推理时暴露
    #   解法：训练时就让两者轮流上场 —— 第 n 个训练步使用 la_names[n % len] 层 LA
    #   本项目只有中文一个目标语言（K=1）→ 槽位 = [源语言 LA, 目标语言 LA]，即英语↔中文交替
    tlr: bool = False
    task_adapter_name: str = "task"
    # adapter 的压缩比例，越大 adapter 越小
    reduction_factor: int = 16
    num_labels: int = 2
    head_name: str = "toxic"

    # ---------- 数据（jsonl，每行 {"text": "...", "label": 0/1}）----------
    train_file: str = ""
    dev_file: str = ""
    test_file: str = ""
    max_length: int = 64

    # ---------- 训练 ----------
    epochs: int = 2
    batch_size: int = 4
    learning_rate: float = 1e-4
    seed: int = 42
    cpu: bool = True
    # 只推理、不训练 —— 用于 zero-shot / translate-classify 这类
    # 「拿一个现成的模型直接在数据上跑前向」的实验。开启后 train_file 可留空。
    eval_only: bool = False

    # ---------- 输出 ----------
    runs_dir: str = "runs"

    # ------------------------------------------------------------------
    @staticmethod
    def load(path):
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        known = {f.name for f in fields(RunConfig)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError("配置文件里有未知字段：%s\n可用字段：%s"
                             % (sorted(unknown), sorted(known)))
        cfg = RunConfig(**raw)
        cfg.validate()
        return cfg

    def validate(self):
        if self.mode not in ("full_ft", "adapter", "madx"):
            raise ValueError("mode 只能是 full_ft / adapter / madx，当前为 %r" % self.mode)
        if not self.train_file and not self.eval_only:
            raise ValueError("必须指定 train_file（若只想推理不训练，请设 eval_only: true）")
        if self.eval_only and not (self.dev_file or self.test_file):
            raise ValueError("eval_only 模式必须至少提供 dev_file 或 test_file，否则无数据可评")
        for name in ("train_file", "dev_file", "test_file"):
            p = abspath(getattr(self, name))
            if p and not os.path.exists(p):
                raise FileNotFoundError("找不到数据文件 %s = %s" % (name, p))
        if self.mode == "madx" and not os.path.exists(abspath(self.language_adapter)):
            raise FileNotFoundError("madx 模式需要语言适配器，找不到：%s"
                                    % abspath(self.language_adapter))
        if self.tlr:
            if self.mode != "madx":
                raise ValueError("tlr 只在 madx 模式下有意义（其他模式没有语言层可交替），"
                                 "当前 mode=%r" % self.mode)
            if not self.source_language_adapter:
                raise ValueError("tlr=true 必须提供 source_language_adapter（源语言 LA）")
            if not os.path.exists(abspath(self.source_language_adapter)):
                raise FileNotFoundError("tlr=true 找不到源语言 LA：%s"
                                        % abspath(self.source_language_adapter))
        return self

    def to_dict(self):
        return asdict(self)

    # ---------- 路径便捷属性 ----------
    @property
    def backbone_path(self):
        return abspath(self.backbone)

    @property
    def language_adapter_path(self):
        return abspath(self.language_adapter)

    @property
    def source_language_adapter_path(self):
        return abspath(self.source_language_adapter)

    def run_dir(self):
        """结果落盘目录：runs/<模式>/<实验名>/seed<种子>/"""
        return os.path.join(abspath(self.runs_dir),
                            self.mode, self.exp_name, "seed%d" % self.seed)
