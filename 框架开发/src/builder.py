# -*- coding: utf-8 -*-
"""按配置组装模型 —— 三种训练模式的差异全集中在这个文件里

三种模式的区别只在「冻结什么、训练什么、怎么叠」：

    full_ft : 底座 + 分类头，全部参数都训练（最贵、最容易过拟合）
    adapter : 冻结底座 → 插入一个任务 adapter → 只训练它（约占 0.8% 参数）
    madx    : 冻结底座 → 叠 [语言适配器, 任务适配器] → 只训练任务适配器
              （语言适配器是现成下载的，DAPT 阶段之外还必须它才能做 TLR）

TLR（Target Language-Ready adapters）也在本文件：
    madx 模式下若 cfg.tlr=True，训练时会按 step 轮流切换垫在下面的语言适配器
    （源语言 LA ↔ 目标语言 LA 交替），避免「训练只见过源语言 LA、推理才换成
    目标语言 LA」造成的 training-inference mismatch。

⚠️ 四个已知的 API 坑（实测踩过，改代码时别踩回去）：
    1. `AdapterConfig.load("pfeiffer", reduction_factor=16)` 是正确构造方式；
       `AdapterConfig.from_dict({...})` 会报 no attribute '__dataclass_fields__'
    2. 这些方法（add_adapter / train_adapter / load_adapter）是动态注入到**实例**上的，
       在类上取签名会 AttributeError
    3. `train_adapter(name)` 会把「除该 adapter 外」的所有参数冻结，
       所以必须先 add_classification_head 再 train_adapter 也没用——
       分类头不会被 train_adapter 管理，它始终保持 requires_grad=True
    4. 判断「某层是否被冻结」时不能用 `"en" in 参数名` ——
       `encoder` / `attention` 里都含 "en"，会误判。
       必须用带点的精确模式：`.adapters.lang_source.` / `.adapters.task.`
"""
from dataclasses import dataclass

from adapters import AutoAdapterModel, AdapterConfig
from adapters.composition import Stack
from transformers import TrainerCallback

# 语言适配器在模型内部的别名（两层 Stack 用这两块牌子）
LA_TARGET = "lang_target"   # 目标语言层：要迁移到的那门语言（本项目＝中文）
LA_SOURCE = "lang_source"   # 源语言层：任务数据所用语言（本项目＝英语）


class TLRPlan:
    """TLR 交替计划

    对应论文的 `step % (K+1)`：第 n 个训练步只激活 la_names[n % len] 那一层语言适配器。
        K = 目标语言数量。本项目只有中文一个目标语言 → K=1
        → 槽位 = [源语言 LA, 目标语言 LA]（0 号是源语言，与论文一致）
        → 即「英语 ↔ 中文」逐步交替
    """

    def __init__(self, la_names, task_name):
        self.la_names = list(la_names)
        self.task_name = task_name

    def stack_for(self, step):
        """返回 (该步要激活的 Stack, 该层 LA 的名字)"""
        name = self.la_names[step % len(self.la_names)]
        return Stack(name, self.task_name), name

    def describe(self):
        return " → ".join(self.la_names) + "（每步轮换，一圈 %d 步）" % len(self.la_names)


class TLRSwitchCallback(TrainerCallback):
    """每个训练步开始时切换垫在下面的语言适配器

    注意：
      - 只切换「激活哪层 LA」，语言适配器本身全程冻结，训练的还是任务适配器
      - 步数用自己维护的计数器，不用 state.global_step —— 后者在
        gradient_accumulation_steps>1 时的语义与「每个训练步」不一致
    """

    def __init__(self, model, plan, log=print):
        self.model = model
        self.plan = plan
        self.log = log
        self.counts = {}
        self.observed = []      # [(global_step, LA 名字), ...]

    def on_step_begin(self, args, state, control, **kwargs):
        step = len(self.observed)
        stack, name = self.plan.stack_for(step)
        self.model.set_active_adapters(stack)
        self.observed.append((state.global_step, name))
        self.counts[name] = self.counts.get(name, 0) + 1
        return control

    def report(self):
        seq = [n for _, n in self.observed]
        head = seq[:12]
        out = ["TLR 交替统计：总步数 %d" % len(seq)]
        out.append("  各语言层使用次数：" + "，".join(
            "%s=%d" % (k, v) for k, v in sorted(self.counts.items())))
        out.append("  前 %d 步序列：%s" % (len(head), " → ".join(head)))
        # 校验是否真的在交替：相邻两步不应相同
        bad = sum(1 for i in range(1, len(seq)) if seq[i] == seq[i - 1])
        out.append("  相邻两步相同的次数：%d（%s）"
                   % (bad, "交替正常" if bad == 0 else "⚠️ 交替异常"))
        return out


@dataclass
class BuildResult:
    """组装结果"""
    model: object
    note: str
    # 训练结束后、评估之前应重新激活的 adapter（推理阶段一律用目标语言 LA）
    eval_adapters: object = None
    # TLR 交替计划；None 表示不做交替（原版：全程固定同一层 LA）
    tlr_plan: object = None


def build_model(cfg):
    """按 cfg.mode 组装模型，返回 BuildResult"""
    model = AutoAdapterModel.from_pretrained(cfg.backbone_path)
    model.add_classification_head(cfg.head_name, num_labels=cfg.num_labels)

    if cfg.mode == "full_ft":
        # ---- 模式 ①：全参数微调 ----
        # 什么都不用冻，默认所有参数 requires_grad=True
        model.active_head = cfg.head_name
        return BuildResult(model, "全参数微调：底座 + 分类头全部可训练")

    if cfg.mode == "adapter":
        # ---- 模式 ②：Adapter ----
        task_cfg = AdapterConfig.load("pfeiffer", reduction_factor=cfg.reduction_factor)
        model.add_adapter(cfg.task_adapter_name, config=task_cfg)
        model.train_adapter(cfg.task_adapter_name)      # 冻结其他一切
        model.set_active_adapters(cfg.task_adapter_name)
        model.active_head = cfg.head_name
        return BuildResult(model, ("Adapter：冻结底座，只训练 1 个任务 adapter"
                                   "（reduction_factor=%d）" % cfg.reduction_factor))

    if cfg.mode == "madx":
        # ---- 模式 ③：MAD-X（语言层 + 任务层）----
        # 语言适配器用现成的（AdapterHub 下载），加载后自动冻结
        model.load_adapter(cfg.language_adapter_path, load_as=LA_TARGET, set_active=False)
        names = [LA_TARGET]

        if cfg.tlr:
            # 源语言 LA 也要挂上，才能交替
            model.load_adapter(cfg.source_language_adapter_path,
                               load_as=LA_SOURCE, set_active=False)
            # 槽位顺序：源语言在前（对应论文 step % (K+1) 的 0 号槽）
            names = [LA_SOURCE, LA_TARGET]

        task_cfg = AdapterConfig.load("pfeiffer", reduction_factor=cfg.reduction_factor)
        model.add_adapter(cfg.task_adapter_name, config=task_cfg)
        model.train_adapter(cfg.task_adapter_name)       # 只训任务层
        # 两层叠起来：前向时先过语言适配器，再过任务适配器
        eval_stack = Stack(LA_TARGET, cfg.task_adapter_name)
        model.set_active_adapters(eval_stack)
        model.active_head = cfg.head_name

        note = ("MAD-X：冻结底座 + 冻结语言适配器(%s)，只训练任务适配器，两层 Stack 叠加"
                % LA_TARGET)
        plan = None
        if cfg.tlr:
            plan = TLRPlan(names, cfg.task_adapter_name)
            note += "；TLR 已启用 → 训练时 %s" % plan.describe()
        return BuildResult(model, note, eval_adapters=eval_stack, tlr_plan=plan)

    raise ValueError("未知模式：%r" % cfg.mode)



def param_summary(model):
    """统计参数量，用于记录与核对"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "trainable_pct": round(100.0 * trainable / total, 4) if total else 0.0,
    }


def active_summary(model):
    """记录当前激活的 adapter / head，便于落盘核对"""
    try:
        adapters = str(model.active_adapters)
    except Exception:
        adapters = "?"
    return {"active_adapters": adapters, "active_head": str(model.active_head)}
