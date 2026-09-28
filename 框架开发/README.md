# A-5 训练框架（Track A 交付物）

**一句话**：改一行配置，就能在「全参数微调 / Adapter / MAD-X / MAD-X+TLR」几种训练方式之间切换，
结果自动落到统一目录。**用的人不需要改代码。**

> 对应 V2 §2 技术路线图里 XLM-R-base 下方的三条并行分支。
> 作者：Track A ｜ 主要使用者：Track B

---

## 一、怎么用

**第 0 步 · 先把环境装好** → 见 [`环境搭建说明.md`](环境搭建说明.md)（含依赖清单说明与 7 个已知的坑）。

```powershell
cd "E:\信息安全yhy\LLM去毒化"

# ① 选一个配置文件（决定用哪种训练方式）
#    框架开发\configs\full_ft.yaml    全参数微调
#    框架开发\configs\adapter.yaml    Adapter
#    框架开发\configs\madx.yaml       MAD-X（不交替，全程目标语言 LA）
#    框架开发\configs\madx_tlr.yaml   MAD-X + TLR（训练时英/中语言适配器交替）
#    框架开发\configs\eval_only.yaml  只推理不训练（zero-shot 等）

# ② 跑
.venv\Scripts\python.exe 框架开发\train.py --config 框架开发\configs\adapter.yaml

# ③ 去 runs\adapter\adapter\seed42\ 看结果
```

**跑多个随机种子**（推荐，省去手写循环）：

```powershell
# 依次跑 seed 1/2/3，最后自动汇总成 batch_summary.csv（含均值行）
.venv\Scripts\python.exe 框架开发\run_batch.py --config 框架开发\configs\adapter.yaml --seeds 1 2 3 --exp-name adapter_3seed
```

`train.py` 的可选参数：
- `--exp-name 我的实验名` 覆盖配置里的实验名
- `--seed 1` 覆盖随机种子（⚠️ 一次只接一个值）

`run_batch.py` 的可选参数：
- `--seeds 1 2 3` 要跑的种子列表（必填）
- `--exp-name` 覆盖实验名（所有种子共用）
- `--keep-going` 某个种子失败时继续跑后面的（默认失败即停）

---

## 二、目录结构与每个文件的作用

```
框架开发\
│
├── README.md                 ← 本文件（使用说明）
│
├── train.py                  ★ 入口脚本。读配置 → 跑一遍 → 打印结果路径
├── run_batch.py              ★ 批量跑多个随机种子，并汇总成 batch_summary.csv
│
├── src\                      ★ 框架核心（4 个模块，职责单一）
│   ├── __init__.py               模块说明（空壳，只为让 Python 认出这是个包）
│   ├── config.py                 配置的定义、加载、校验
│   │                             （定义 RunConfig 类；所有相对路径按项目根目录解析）
│   ├── data.py                   数据读取
│   │                             （读 jsonl → 转成 Trainer 能吃的 Dataset）
│   ├── builder.py                ★ 模型组装 —— 各模式的差异全在这里
│   │                             （full_ft / adapter / madx 各自的冻结与叠加方式
│   │                               + TLR 交替计划与切换回调）
│   └── runner.py                 训练 + 评估 + 落盘
│                                 （各模式共用同一条训练路径；dev / test 分两段评估）
│
├── configs\                   ★ 配置文件（一份配置 = 一次实验）
│   ├── full_ft.yaml              模式 ①：全参数微调
│   ├── adapter.yaml              模式 ②：Adapter
│   ├── madx.yaml                 模式 ③：MAD-X（tlr: false）
│   ├── madx_tlr.yaml             模式 ③ + TLR（tlr: true，语言适配器交替）
│   └── eval_only.yaml            只推理不训练（eval_only: true）
│
├── data\
│   ├── demo_train.jsonl          ⚠️ 示意数据（16 条），等 Track C 的表2 到位后替换
│   ├── demo_dev.jsonl            ⚠️ 示意数据（8 条）
│   └── demo_test.jsonl           ⚠️ 示意数据（8 条），演示 test 段（将来换成表5 外部基准）
│
├── 环境搭建说明.md            ★ 从零装环境的分步指引（含 7 个已知的坑）
├── requirements.txt           ★ 依赖清单（版本已固定；torch 需按机器单独装）
│
├── 01_环境验证.py             ┐
├── 02_下载与推理验证.py        │ 上面几个是「第 1、2、4 步」的验证 / 下载脚本，
├── 03_三种模式跑通.py         │ 用于排查环境与验证能力，不是框架本体
└── 04_下载语言适配器.py       ┘ （04 下载中/英两个语言适配器，并验证可同时挂载切换）
```

> ℹ️ 上面 4 个验证 / 下载脚本运行后会在同目录生成 `_01_检查结果.txt` ~ `_04_下载结果.txt`。
> 这些是**脚本输出，不入仓库**（已在 `.gitignore` 里按 `框架开发/_*.txt` 排除），需要时重跑脚本即可。

**框架本体只有 7 个文件**：`train.py` + `run_batch.py` + `src/` 下的 4 个模块（+ `configs/` 里的配置）。其余都是验证脚本和记录。

---

## 三、各模式的区别（只差在"冻结什么"）

| 模式 | 底座 | 语言适配器 | 任务适配器 | 可训练参数占比 |
|---|---|---|---|---|
| `full_ft` | 训练 | — | — | **100%** |
| `adapter` | 冻结 | — | 训练 | **约 0.83%** |
| `madx`（tlr: false） | 冻结 | 冻结（现成的，1 个） | 训练 | **约 0.81%** |
| `madx_tlr`（tlr: true） | 冻结 | 冻结（现成的，2 个，交替使用） | 训练 | **约 0.79%** |

> 占比略降是因为挂了两个语言适配器，分母（总参数）变大了；**可训练参数个数完全一样**
> （都是 2,328,788），说明语言适配器全程冻结、只训任务适配器。

实现全在 `src/builder.py` 的 `build_model()` 里，几套组装配方：

```python
# full_ft：什么都不冻
model.active_head = "toxic"

# adapter：冻结底座 → 插任务适配器 → 只训它
model.add_adapter("task", config=AdapterConfig.load("pfeiffer", reduction_factor=16))
model.train_adapter("task")
model.set_active_adapters("task")

# madx：再叠一层语言适配器（现成的，自动冻结）
model.load_adapter("<目标语言 LA 路径>", load_as="lang_target", set_active=False)
model.add_adapter("task", ...)
model.train_adapter("task")
model.set_active_adapters(Stack("lang_target", "task"))    # 语言层 + 任务层

# madx + tlr：再挂一个源语言 LA，训练时两者交替
model.load_adapter("<源语言 LA 路径>", load_as="lang_source", set_active=False)
# → 每个训练步切换：Stack(lang_source, task) / Stack(lang_target, task)
```

### 关于 TLR（`tlr: true`）

**它解决的是 MAD-X 的一个已知毛病：training-inference mismatch。**

| | 原版 MAD-X（`tlr: false`… ） | TLR（`tlr: true`） |
|---|---|---|
| 训练时任务适配器见到的语言层 | 只见到**一种** | **源语言 ↔ 目标语言轮流** |
| 推理时用的语言层 | 目标语言 | 目标语言 |

原版的毛病：如果训练时垫的是源语言 LA、推理时突然换成目标语言 LA，**两者从没配合过**，
效果会掉，而且**只在推理时暴露**（训练 loss 一切正常）。TLR 让它们在训练阶段就磨合过。

实现规则（论文原文 `step % (K+1)`）：**第 n 个训练步只激活第 `n % 层数` 层语言适配器**。
本项目只有中文一个目标语言（K=1）→ 槽位 = `[lang_source(英), lang_target(中)]`，即逐步交替。

**两个实现上的关键点**（都在 `runner.py` / `builder.py` 里）：
1. 交替只改「垫在下面的是哪层 LA」，**语言适配器全程冻结**，训练的还是任务适配器
2. **训练结束后、评估之前必须切回目标语言 LA** —— 因为训练最后一步用哪层 LA 取决于步数奇偶，
   不能拿它去评估（`runner.py` 里由 `res.eval_adapters` 保证）

**怎么确认交替真的生效**：训练日志里会打印统计，长这样：

```
TLR 交替统计：总步数 8
  各语言层使用次数：lang_source=4，lang_target=4
  前 8 步序列：lang_source → lang_target → lang_source → lang_target → ...
  相邻两步相同的次数：0（交替正常）
推理前已切回目标语言 LA：active_adapters = Stack[lang_target, task]
```

`metrics.json` 里也会存一份（`tlr` 字段），便于事后核对这次实验到底交没交替。

### 关于 eval_only（只推理不训练）

配置里加一行 `eval_only: true`，就跳过训练、直接进评估。用途：

- **zero-shot 测速**（随机分类头 + 只读测试集，纯看吞吐）
- 加载别人训好的 checkpoint，直接在中文数据上评
- 任何"只想拿模型跑一遍前向"的场景

开启后：
1. 不调用训练，**不更新任何参数**
2. `train_file` **可以不填**（但必须至少有 `dev_file` 或 `test_file`）
3. `metrics.json` 里 `train_seconds = 0`、`train_loss = null`、`eval_only = true`

`configs/eval_only.yaml` 是一份可直接用的样例。

---

## 四、数据格式

**jsonl**，每行一个对象，至少含 `text` 和 `label` 两个字段：

```json
{"text": "这个产品真是垃圾", "label": 1}
{"text": "你好，今天天气不错", "label": 0}
```

- `label`：**1 = 有毒，0 = 无毒**（这个约定会影响 Toxic Recall / FPR 的算法）
- 额外字段（如 `source_id`、`attack_type`）会被保留，便于后续按类别拆开分析

**三段数据的用途**（对应配置里的 `train_file` / `dev_file` / `test_file`）：

| 字段 | 用途 | 三方都给了会发生什么 |
|---|---|---|
| `train_file` | 训练用 | 先训练 |
| `dev_file` | 主评估集（调参、看进展） | 训完在 dev 上评一遍 → `predictions.csv` |
| `test_file` | **最终评测**（COLD / ToxiCN 官方 test、表5 外部基准） | 再在 test 上评一遍 → `predictions_test.csv` |

- `dev_file` 和 `test_file` **可以只给一个**（那就只评一次，结果仍叫 `predictions.csv`）
- **要不要保留 test 段由你决定** —— 留空就只评 dev
- ⚠️ **test 集绝不能混进 train**，否则分数虚高（这是数据泄漏）

---

## 五、结果落在哪、长什么样

```
runs\<模式>\<实验名>\seed<种子>\
├── config.json            本次实验的完整配置（原样存档，便于复现）
├── metrics.json           参数统计 + 指标 + 耗时
├── predictions.csv        逐条预测（主评估集：有 dev 用 dev，否则用 test）
├── predictions_test.csv   逐条预测（test 段；只有提供 test_file 时才有）
└── train.log              运行日志

runs\<模式>\<实验名>\batch_summary.csv    ← 用 run_batch.py 跑多 seed 时才有
                                             每个种子一行 + 一行 MEAN 均值
```

**指标口径**（按 V2 评估体系）：

| 指标 | 含义 |
|---|---|
| `macro_f1` | 两类各算 F1 再平均，比准确率更能反映不均衡数据上的真实水平 |
| `accuracy` | 整体准确率 |
| `toxic_recall` | **有毒样本被找出来的比例**（漏检少 = 高） |
| `fpr` | **无毒样本被误判成有毒的比例**（误伤少 = 低） |

**`metrics.json` 里的字段**：

| 字段 | 内容 |
|---|---|
| `metrics` | 主评估集（dev）的指标 |
| `metrics_test` | **test 段的指标**（没给 `test_file` 时为 `{}`） |
| `params` | 总参数 / 可训练参数 / 占比 |
| `active` | 落盘时激活的 adapter 与 head |
| `tlr` | 是否启用交替、每层 LA 用了几次（未启用为 `null`） |
| `eval_only` | 是否只推理没训练 |
| `train_seconds` / `train_loss` | 训练耗时与最终 loss（`eval_only` 时为 `0` / `null`） |

---

## 六、已知的坑（改代码前先看这条）

1. ⚠️ **`TrainingArguments` 要设两个参数，它们是同一个根因的两面**
   `AutoAdapterModel` 的 forward 签名是**动态生成的**，`Trainer` 靠签名推断不出任何东西：
   - `remove_unused_columns=False` —— 否则它会把 `labels` 当"无用列"删掉 → 报 `did not return a loss`
   - `label_names=["labels"]` —— 否则它认为"没有标签" → 评估时不算 `eval_loss`、拿不到 `label_ids`
     （表现为 `metrics.json` 指标全空、`predictions.csv` 只写 1 行）
   两个都已在 `runner.py` 里设好，改训练参数时别漏。

1.5 ⚠️ **`trainer.predict()` 的返回形状不固定**
   `predictions` 可能是 tuple、`label_ids` 可能是 `None` 或标量。`runner.py` 里统一用
   `np.asarray(x).reshape(-1)` 拍平并改用显式索引循环，避免 `inhomogeneous shape` 与
   `'numpy.int64' object is not iterable` 两类报错。

2. ⚠️ **需要额外装 `accelerate`**（`Trainer` 强制要求，`transformers` 不会自动带）

3. ⚠️ **配置构造用 `AdapterConfig.load("pfeiffer", reduction_factor=16)`**
   `AdapterConfig.from_dict({...})` 会报错。

4. ⚠️ **切换 Head 用属性赋值**：`model.active_head = "toxic"`
   没有 `set_active_head()` 这个方法。

5. ⚠️ **`model(**enc)` 拿不到 hidden states**
   因为自动挂了 MaskedLM 头。要句子表示请用 `model.base_model(**enc)`。

6. ⚠️ **判断"某层是否被冻结"不能用 `"en" in 参数名`**
   `encoder` / `attention` 里都含 "en"，会误判成"语言层在训练"。
   要用带点的精确模式：`.adapters.lang_source.` / `.adapters.task.`。
   实测：可训练参数里属于语言层的 **0 条**（已冻结）、属于任务层的 **48 条**。

7. ℹ️ **stderr 里可能出现 `There are adapters available but none are activated for the forward pass.`**
   这是 `adapters` 库在 `add_adapter` / `train_adapter` 期间的内部提示，**无害**
   —— 训练与评估时的 `active_adapters` 是正确的（日志里会打印出来核对）。

---

## 七、完成度与已知边界

**已完成**

| 步 | 内容 | 状态 |
|---|---|---|
| 1 | 环境 + 库验证 + 语言适配器 | ✅ |
| 2 | 三种训练模式跑通 | ✅ |
| 3 | 配置驱动的框架 | ✅ |
| 4 | TLR 交替机制 | ✅ |
| — | dev + test 两段评测 | ✅ |
| — | `eval_only` 只推理模式 | ✅ |
| — | 多 seed 批量脚本 `run_batch.py` | ✅ |
| 5 | 接口说明文档 + 交付 Track B | ⬜ |

**还没做 / 边界**

- **数据**：`data/` 下三份都是示意数据，等 Track C 的表2 到位后替换
- **环境**：当前是 CPU 版 torch（本地验证用）；上云服务器后要按那边的 CUDA 重装并锁定版本
  → 步骤见 [`环境搭建说明.md`](环境搭建说明.md) §3 与 §10
- **任务形式**：框架只做「文本分类」这一种形式（毒性检测）—— 二分类，`num_labels` 可配
- **检索增强（Retrieval-Augmented）**：检索组件**不在框架内**。框架负责"拿到文本 → 跑前向 → 落盘"，
  检索与拼接是使用者自己的环节
- **多语言 LA 与 backbone 绑定**：现有语言适配器都是 xlm-roberta-base 的，所以 `madx` 模式只能用
  xlm-roberta-base 作 backbone（mBERT 走 `full_ft`）
