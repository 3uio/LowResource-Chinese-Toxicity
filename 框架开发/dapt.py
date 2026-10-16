"""DAPT: masked-language-model continued pretraining, not toxicity classification."""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import random
import time

import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForMaskedLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments, default_data_collator, set_seed)

from src.config import abspath
from src.isolation import normalized_hash, read_verified_corpus


class EncodedRows(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class MLMCollator:
    """Dynamic training masks; precomputed validation masks stay fixed."""
    def __init__(self, tokenizer, probability, seed):
        self.tokenizer = tokenizer
        self.mlm = DataCollatorForLanguageModeling(tokenizer, mlm_probability=probability, seed=seed)

    def __call__(self, examples):
        if "labels" in examples[0]:
            return default_data_collator(examples)
        batch = self.mlm(examples)
        # Short comments may receive zero random masks, yielding an undefined loss.
        for i in range(len(examples)):
            if (batch["labels"][i] != -100).any():
                continue
            candidates = [j for j, token in enumerate(batch["input_ids"][i].tolist())
                          if token not in self.tokenizer.all_special_ids
                          and batch["attention_mask"][i, j].item()]
            if not candidates:
                raise ValueError("A corpus row has no maskable tokens")
            j = candidates[0]
            batch["labels"][i, j] = batch["input_ids"][i, j]
            batch["input_ids"][i, j] = self.tokenizer.mask_token_id
        return batch


def select_rows(rows, fraction, validation_fraction, seed, max_samples=0):
    groups = defaultdict(list)
    for row in rows:
        groups[normalized_hash(row["text"])].append(row)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    selected = keys[:max(2, math.ceil(len(keys) * fraction))]
    if max_samples:
        limited, count = [], 0
        for key in selected:
            if count >= max_samples:
                break
            limited.append(key)
            count += len(groups[key])
        selected = limited
    if len(selected) < 2:
        raise ValueError("Need at least two normalized text groups for train/MLM validation")
    n_val = max(1, min(len(selected) - 1, math.ceil(len(selected) * validation_fraction)))
    validation = [row for key in selected[:n_val] for row in groups[key]]
    train = [row for key in selected[n_val:] for row in groups[key]]
    return train, validation


def checked_optimizer_steps(optimizer):
    """Trainer global_step can advance even when AMP skips every Adam update."""
    steps = sorted({int(state["step"]) for state in optimizer.state.values()
                    if "step" in state})
    if not steps or max(steps) == 0:
        raise RuntimeError("No completed optimizer updates; do not use this run's checkpoints. "
                           "Check mixed-precision overflow; retry a fresh run with bf16 if supported.")
    return steps


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", default="DAPT语料/表1_无标注语料.jsonl")
    p.add_argument("--audit", default="DAPT语料/表1_隔离审计.json")
    p.add_argument("--frozen-hashes", default="DAPT语料/表1_冻结哈希.json")
    p.add_argument("--backbone", default=".model_cache/xlm-roberta-base")
    p.add_argument("--output", default="runs/dapt/pilot_seed42")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--precision", choices=["none", "fp16", "bf16"], default="none")
    p.add_argument("--fraction", type=float, default=0.1)
    p.add_argument("--validation-fraction", type=float, default=0.05)
    p.add_argument("--max-samples", type=int, default=0, help="Smoke-only sample cap; groups stay intact")
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--gradient-accumulation", type=int, default=4)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--mlm-probability", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume-from-checkpoint", help="Trainer checkpoint directory, not final model/")
    return p


def run(args):
    started = time.time()
    if not (0 < args.fraction <= 1 and 0 < args.validation_fraction < 1
            and 0 < args.mlm_probability < 1 and args.batch_size > 0
            and args.gradient_accumulation > 0 and args.max_length >= 4
            and args.epochs > 0 and args.learning_rate > 0 and args.max_samples >= 0
            and (args.max_steps == -1 or args.max_steps > 0)):
        raise ValueError("Invalid DAPT hyperparameters")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu only for smoke testing")
    if args.device == "cpu" and args.precision != "none":
        raise ValueError("CPU smoke uses --precision none")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("GPU does not support bf16; select fp16 or none")
    output = Path(abspath(args.output))
    if output.exists() and any(output.iterdir()) and not args.resume_from_checkpoint:
        raise FileExistsError("Output is not empty; choose a new --output")
    set_seed(args.seed)
    rows, audit = read_verified_corpus(abspath(args.corpus), abspath(args.audit), abspath(args.frozen_hashes))
    train, validation = select_rows(rows, args.fraction, args.validation_fraction, args.seed, args.max_samples)
    selection = {"train_ids": [r["id"] for r in train], "validation_ids": [r["id"] for r in validation]}
    config = vars(args).copy()
    config.update(corpus_sha256=audit["corpus_sha256"], frozen_sha256=audit["frozen_sha256"])
    output.mkdir(parents=True, exist_ok=True)
    if args.resume_from_checkpoint:
        old = json.loads((output / "config.json").read_text(encoding="utf-8"))
        for key, value in config.items():
            if key not in {"resume_from_checkpoint", "max_steps", "epochs"} and old.get(key) != value:
                raise ValueError("Resume configuration mismatch: " + key)
        if json.loads((output / "selection.json").read_text(encoding="utf-8")) != selection:
            raise ValueError("Resume dataset selection mismatch")
        checkpoint = Path(abspath(args.resume_from_checkpoint)).resolve()
        if not checkpoint.is_relative_to(output.resolve()):
            raise ValueError("Resume checkpoint must belong to the specified output directory")
        if not (checkpoint / "trainer_state.json").exists():
            raise ValueError("Resume path is not a Trainer checkpoint")
    for name, payload in (("config.json", config), ("selection.json", selection)):
        (output / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(abspath(args.backbone))
    if not tokenizer.mask_token:
        raise ValueError("Backbone tokenizer has no mask token")
    if args.max_length > tokenizer.model_max_length:
        raise ValueError("max_length exceeds tokenizer limit")

    def encode(data):
        enc = tokenizer([r["text"] for r in data], padding="max_length", truncation=True,
                        max_length=args.max_length, return_special_tokens_mask=True)
        special_ids = set(tokenizer.all_special_ids)
        for i, ids in enumerate(enc["input_ids"]):
            if not any(token not in special_ids for token in ids):
                raise ValueError("No maskable token after tokenization: " + data[i]["id"])
        return [{k: enc[k][i] for k in enc} for i in range(len(data))]

    training_items = encode(train)
    evaluation_collator = MLMCollator(tokenizer, args.mlm_probability, args.seed + 1)
    validation_items = []
    for item in encode(validation):
        masked = evaluation_collator([item])
        validation_items.append({k: v[0] for k, v in masked.items()})
    model = AutoModelForMaskedLM.from_pretrained(abspath(args.backbone))
    if args.max_length > model.config.max_position_embeddings - tokenizer.pad_token_id - 1:
        raise ValueError("max_length exceeds model position capacity")
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    training_args = TrainingArguments(
        output_dir=str(output / "checkpoints"), num_train_epochs=args.epochs,
        max_steps=args.max_steps, per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate, warmup_ratio=0.06, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        logging_steps=10, report_to=[], use_cpu=args.device == "cpu",
        fp16=args.precision == "fp16", bf16=args.precision == "bf16",
        seed=args.seed, data_seed=args.seed, remove_unused_columns=False, disable_tqdm=True)
    trainer = Trainer(model=model, args=training_args,
                      train_dataset=EncodedRows(training_items), eval_dataset=EncodedRows(validation_items),
                      data_collator=MLMCollator(tokenizer, args.mlm_probability, args.seed))
    initial = trainer.evaluate()
    if not math.isfinite(initial["eval_loss"]):
        raise RuntimeError("Initial MLM loss is not finite; abort before training")
    train_started = time.time()
    result = trainer.train(resume_from_checkpoint=abspath(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    train_seconds = time.time() - train_started
    optimizer_steps = checked_optimizer_steps(trainer.optimizer)
    final = trainer.evaluate()
    if not all(math.isfinite(v) for v in (final["eval_loss"], result.training_loss)):
        raise RuntimeError("Non-finite training/final loss; do not use this run's Trainer checkpoints")
    final_dir = output / "model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)
    import transformers
    metrics = dict(stage="DAPT_MLM", corpus_rows=len(rows), train_rows=len(train),
                   validation_rows=len(validation), global_step=trainer.state.global_step,
                   optimizer_step_values=optimizer_steps,
                   smoke_only=bool(args.max_samples or args.max_steps > 0),
                   initial_eval_loss=initial["eval_loss"], final_eval_loss=final["eval_loss"],
                   train_loss=result.training_loss, train_seconds=round(train_seconds, 3),
                   total_seconds=round(time.time() - started, 3),
                   peak_cuda_gib=torch.cuda.max_memory_allocated() / 1024**3 if args.device == "cuda" else None,
                   device=torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
                   torch_version=torch.__version__, transformers_version=transformers.__version__,
                   corpus_sha256=audit["corpus_sha256"], final_model=str(final_dir))
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


if __name__ == "__main__":
    run(parser().parse_args())
