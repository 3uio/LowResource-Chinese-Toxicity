# -*- coding: utf-8 -*-
"""A-5 训练框架（Track A）

模块划分：
    config.py   配置定义与加载（一份 yaml 决定整个实验）
    data.py     数据读取（jsonl → Trainer 可用的 Dataset）
    builder.py  模型组装（三种训练模式的差异都在这里）
    runner.py   训练 / 评估 / 落盘（三种模式共用同一条路径）
"""
