# LowResource-Chinese-Toxicity

低标注中文毒性检测与扰动鲁棒性研究。基于已有 XLM-R-base 编码器，比较继续预训练、全参数微调、Adapter 与 MAD-X/TLR 等路线，不是从零训练生成式大模型。

## 目录与入口

| 目录 | 用途 | 阅读入口 |
|---|---|---|
| `框架开发/` | Track A 提供的分类框架、DAPT 脚本与回归测试，供 B 运行实验 | [框架说明](框架开发/README.md) |
| `data-build/` | Track C 的数据构建工具链与接口规范 | [数据接口](data-build/docs/数据schema与接口规范.md) |

## 当前状态

- 2026-10-15 完成单卡 RTX 4090、BF16、seed42 的全量单轮 DAPT，内部 MLM 验证 loss 从 3.327670 降至 3.039086。
- 这只证明 MLM 训练与产物验收通过，**不证明毒性分类 F1 或鲁棒性提升**。原底座 FT 与 DAPT+FT 的同条件对照仍由 B 执行。
- 四种分类分支已做离线小模型 CPU 回归；不代表正式数据上的 GPU 分类实验全部完成。
- 真实语料、模型权重、云端上传包、下载结果、内部项目记忆不公开入库。clone 后需按团队交接获取对应资源，不能直接复现真实数据实验。

## 使用

从仓库根目录执行命令，先按[环境搭建说明](框架开发/环境搭建说明.md)选择分类或 DAPT 依赖，不要把本地 CPU torch 装到 GPU 实例上。

```bash
# 分类示例使用仓库内的示意数据；需另行准备底座模型
python 框架开发/train.py --config 框架开发/configs/adapter.yaml

# 不下载模型、不用真实语料的离线小模型回归测试
python -m unittest discover -s 框架开发/tests -v
```

DAPT 命令见 [DAPT 运行说明](框架开发/DAPT运行说明.md)，实测配置、结果与限制见 [seed42 实验记录](框架开发/实验记录/DAPT_seed42_20261015.md)。

## B 的 DAPT 接入

在原分类配置的副本中，将 `backbone` 改为私下交付的 DAPT `model/` 路径，`model_checkpoint` 留空，再运行有监督分类训练。`model_checkpoint` 只用于重载本框架已经训练好的完整分类器。保持原模型对照的数据划分、分类超参、seed 和评估口径一致。

数据使用与再分发须遵守上游数据集、模型及适配器各自的许可；本仓库不代替第三方授权。
