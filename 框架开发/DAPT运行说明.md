# Track A：DAPT 运行说明

更新：2026-10-16。已经完成一次 BF16 全量单轮训练，**不要因文档更新而重跑已验收实验**。历史指标与边界见[实测记录](实验记录/DAPT_seed42_20261015.md)。

## 1. 目的与资源

DAPT 用中文无标注文本，对已有 XLM-R-base 做 MLM 掩码语言模型继续预训练。它不是毒性分类训练，也不直接得到分类 F1；B 还需做原底座 FT 与 DAPT+FT 的同条件对照。

从仓库根目录运行。仅 clone 不包含下面的真实语料与底座，须通过团队私下交接准备：

```text
框架开发/
.model_cache/xlm-roberta-base/     完整原始底座、配置与分词器
DAPT语料/表1_无标注语料.jsonl
DAPT语料/表1_隔离审计.json
DAPT语料/表1_冻结哈希.json
```

当前冻结语料 27,851 条，COLD 25,562、ToxiCN 2,289。脚本校验语料与冻结集合 SHA256、审计状态及归一化哈希交集；缺失或不匹配就停止。`zhconv` 缺失会报错，不允许退化为不转繁简。

本轮隔离范围是已有的官方 dev/test、全 CNTP 家族和表2预留池。不能据此声称尚未交付的 C/D 最终表、未来新增基准已经全部审计。保持已冻结文件不变，有新增基准时先报告交集，再由团队决定处理方式。

## 2. 环境

实测：Ubuntu 22.04、Python 3.12.3、RTX 4090 24 GB、torch 2.8.0+cu128、transformers 4.57.6、BF16。使用平台预置 CUDA torch，不要覆盖成本地 CPU 版本。

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA'); print('BF16:', torch.cuda.is_bf16_supported())"
python -m pip install -r 框架开发/requirements-dapt.txt
python -m pip check
```

`requirements-dapt.txt` 仅包含 DAPT 所需的固定直接依赖。B 的 Adapter/分类实验使用 `requirements.txt`，本次 DAPT 的云端通过不等于全部分类依赖与实验已验证。

Linux 服务器上，模型已完整落地后可设离线环境：

```bash
export USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
```

离线小模型回归测试不下载权重、不用真实语料，但依赖完整分类环境：

```bash
python -m unittest discover -s 框架开发/tests -v
```

## 3. 新实验的执行顺序

以下命令的输出目录必须为空或不存在。复跑用新名字；不要同时启动两次。均从原始底座开始，不把 pilot 权重继续接到全量实验。

### 3.1 BF16 烟雾测试

```bash
python 框架开发/dapt.py --device cuda --precision bf16 --fraction 0.1 --validation-fraction 0.05 --epochs 1 --batch-size 4 --gradient-accumulation 8 --max-length 128 --learning-rate 5e-5 --mlm-probability 0.15 --seed 42 --max-samples 64 --max-steps 10 --output runs/dapt/bf16_smoke_seed42
```

`max_steps` 覆盖 epoch 上限。当前最多 64 条的测试实际为 60 训练、4 验证，10 步会循环约 5 轮，仅验证流程，不作为科研结果。

首跑 FP16 两步曾出现 loss 有限、`global_step=2`，但 Adam 实际 step 为 0、权重完全不变，原因是 AMP 跳过了更新。改为 BF16 后，烟雾测试 202 个共同参数张量全部变化，Adam step 为 10。

新版脚本增加 `optimizer_step_values` 检查：Adam 无状态或全部 step 为 0 时拒绝保存最终成功产物。这是最低更新检查，不替代参数变化、非有限值与下游效果验收；续训时该字段包含历史累计步数，不能独立证明本次恢复后执行了多少更新。

### 3.2 10% pilot

```bash
python 框架开发/dapt.py --device cuda --precision bf16 --fraction 0.1 --validation-fraction 0.05 --epochs 1 --batch-size 4 --gradient-accumulation 8 --max-length 128 --learning-rate 5e-5 --mlm-probability 0.15 --seed 42 --output runs/dapt/bf16_pilot_seed42
```

seed42 为 2,644 训练、140 验证，共 83 步。pilot 用于检查耗时、显存和 MLM 训练是否工作；是否有分类增益仍看 B 的对照。

### 3.3 全量单轮

团队确认范围和预算后使用：

```bash
python 框架开发/dapt.py --device cuda --precision bf16 --fraction 1.0 --validation-fraction 0.05 --epochs 1 --batch-size 4 --gradient-accumulation 8 --max-length 128 --learning-rate 5e-5 --mlm-probability 0.15 --seed 42 --output runs/dapt/bf16_full_seed42
```

全量是全语料参与划分，其中 26,459 条训练、1,392 条内部 MLM 验证，不是 27,851 条都训练。归一化同文分组后划分，训练动态掩码、验证固定掩码。学习率预热比例 0.06、weight decay 0.01、有效 batch32。

本轮单卡实测总流程约 19 分 28 秒、训练约 16 分 14 秒；不承诺其他机器相同时长。显存峰值约 6.34 GiB 是 PyTorch 已分配峰值，不是整卡总占用。收费按平台实际开机时长与价格核算，包含准备、传输和空闲时间。

## 4. 结果、断点与关机

```text
runs/dapt/<本次实验名>/
  config.json          参数与语料/冻结集合指纹
  selection.json       训练、内部验证行 ID
  checkpoints/         Trainer 断点与优化器等状态
  model/               最终 MLM 模型与分词器，交 B
  metrics.json         loss、步数、时长、显存、设备与版本
  train_log.json       Trainer 指标历史
```

每 epoch 保存，只留最后一个 Trainer 断点；中途无断点时不能从最终模型目录恢复优化器。输出与断点可能占数 GB，应留足磁盘。

恢复沿用原参数和 `--output`，增加 `--resume-from-checkpoint <该输出目录内的checkpoints/checkpoint-N>`。脚本核对配置、划分和指纹；`model/` 不是 Trainer 断点。动态训练掩码生成器状态未单独保存，不保证恢复与连续训练逐位一致。

长任务用后台会话或 `nohup` 并独立保留日志、退出码。自动关机只是结束实例运行，不代表训练成功；必须同时检查退出码、指标和产物。关机前确认结果落在平台保留的数据盘，之后及时下载或备份，**不要释放/删除唯一副本实例**。平台的关机、存储与释放规则以当时控制台为准。

## 5. B 的接入与验收边界

私下交付的正式模型目录为 `dapt_result/bf16_full_20261015T145750Z_4002/model`。在 B 原分类配置副本中修改：

```yaml
backbone: dapt_result/bf16_full_20261015T145750Z_4002/model
model_checkpoint: ""
```

相对路径以仓库根目录为基准；若模型放在别处就填实际绝对路径。不用重复跑 DAPT。正式分类超参、数据划分和 seed 与原底座基线保持一致。

`model_checkpoint` 用于本框架已训练的分类器，不能用来加载无分类头的 DAPT 产物。仓库示意数据、小模型测试、MLM loss 降低都不能作为正式分类效果。真实语料、权重、逐条预测和云端压缩包不上传 GitHub；本仓库仅公开代码、说明和聚合实验记录。
