# -*- coding: utf-8 -*-
"""数据读取与封装

约定数据格式：jsonl，每行一个对象
    {"text": "这条评论的内容", "label": 1}     # label: 0=无毒 / 1=有毒
额外字段（如 source_id / attack_type）会被保留在 meta 里，方便后续按类别拆开分析。
"""
import json

import torch
from torch.utils.data import Dataset


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "text" not in obj or "label" not in obj:
                raise ValueError("%s 第 %d 行缺少 text 或 label 字段" % (path, i + 1))
            rows.append(obj)
    return rows


class ToxicDataset(Dataset):
    """把 jsonl 的行转成 Trainer 能吃的样本"""

    def __init__(self, rows, tokenizer, max_length=64, head_name="toxic"):
        self.rows = rows
        self.head_name = head_name
        texts = [r["text"] for r in rows]
        self.enc = tokenizer(texts, padding=True, truncation=True,
                             max_length=max_length, return_tensors="pt")
        self.labels = torch.tensor([int(r["label"]) for r in rows], dtype=torch.long)
        # 保留原始 meta，便于落盘时写进 predictions.csv
        self.meta = [{k: v for k, v in r.items() if k not in ("text",)} for r in rows]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


def load_datasets(cfg, tokenizer):
    """按配置读取 train / dev / test（没有的返回 None）"""
    out = {}
    for split in ("train", "dev", "test"):
        p = getattr(cfg, "%s_file" % split)
        if not p:
            out[split] = None
            continue
        from .config import abspath
        rows = read_jsonl(abspath(p))
        out[split] = ToxicDataset(rows, tokenizer,
                                  max_length=cfg.max_length,
                                  head_name=cfg.head_name)
    return out
