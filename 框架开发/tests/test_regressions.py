"""Offline CPU regression tests with tiny XLM-R fixtures (no model downloads)."""
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast, XLMRobertaConfig, XLMRobertaForMaskedLM
from adapters import AdapterConfig, AutoAdapterModel

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from src.builder import LA_SOURCE, LA_TARGET, build_model
from src.config import RunConfig
from src.data import read_jsonl
from src.isolation import normalized_hash, read_verified_corpus, sha256_file
from src.runner import compute_metrics, evaluate_split, run
import dapt


class RegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory(prefix="trackA_regression_")
        cls.root = Path(cls.temp.name)
        cls.base = cls.root / "base"
        cls.base.mkdir()
        vocab = {token: i for i, token in enumerate(
            ["<s>", "<pad>", "</s>", "<unk>", "<mask>", "good", "bad", "text", "sample", "other"])}
        tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        tok.post_processor = processors.TemplateProcessing(
            single="<s> $A </s>", special_tokens=[("<s>", 0), ("</s>", 2)])
        cls.tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<s>",
                                               pad_token="<pad>", eos_token="</s>",
                                               unk_token="<unk>", mask_token="<mask>", model_max_length=64)
        cls.tokenizer.save_pretrained(cls.base)
        config = XLMRobertaConfig(vocab_size=len(vocab), hidden_size=16, intermediate_size=24,
                                 num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=66,
                                 pad_token_id=1, bos_token_id=0, eos_token_id=2)
        XLMRobertaForMaskedLM(config).save_pretrained(cls.base)
        model = AutoAdapterModel.from_pretrained(cls.base)
        cls.source, cls.target = cls.root / "en", cls.root / "zh"
        for name, path in (("en", cls.source), ("zh", cls.target)):
            model.add_adapter(name, config=AdapterConfig.load("pfeiffer", reduction_factor=2))
            model.save_adapter(str(path), name)
        cls.data = cls.root / "data.jsonl"
        cls.rows = [dict(text="good text", label=0, source_id="family0", attack_type="original"),
                    dict(text="bad text", label=1, source_id="family1", attack_type="homo")]
        cls.data.write_text("\n".join(json.dumps(r) for r in cls.rows) + "\n", encoding="utf-8")
        cls.dev = cls.root / "dev.jsonl"
        cls.dev_rows = [dict(text="good sample", label=0, source_id="family2", attack_type="original"),
                        dict(text="bad sample", label=1, source_id="family3", attack_type="homo")]
        cls.dev.write_text("\n".join(json.dumps(r) for r in cls.dev_rows) + "\n", encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def cfg(self, mode="adapter", tlr=False):
        return RunConfig(backbone=str(self.base), train_file=str(self.data), dev_file=str(self.dev),
                         mode=mode, tlr=tlr, language_adapter=str(self.target),
                         source_language_adapter=str(self.source), reduction_factor=2,
                         epochs=1, batch_size=1, max_length=12, cpu=True,
                         exp_name="trained_" + str(tlr), runs_dir=str(self.root / "runs"))

    def test_01_normalization_edge_cases(self):
        self.assertEqual(normalized_hash("繁體測試，ＡＢＣ！"), normalized_hash("繁体测试abc"))
        self.assertEqual(normalized_hash("零\u200b宽\ufeff文字"), normalized_hash("零宽文字"))
        self.assertNotEqual(normalized_hash("保留🙂表情"), normalized_hash("保留表情"))

    def test_02_metrics_single_class_and_tuple(self):
        metrics = compute_metrics(((np.array([[2, 0]]),), np.array([0])))
        self.assertEqual(metrics["macro_f1"], 0.5)

    def test_03_prediction_metadata(self):
        fake = SimpleNamespace(predict=lambda *a, **k: SimpleNamespace(
            predictions=(np.array([[2, 0], [0, 2]]),), label_ids=np.array([0, 1]),
            metrics={"eval_macro_f1": 1.0}))
        class Rows:
            rows = self.rows
            def __len__(self):
                return 2
        evaluate_split(fake, Rows(), "dev",
                       str(self.root), lambda *a: None, "eval", "unused.csv")

    def test_04_madx_source_train_target_eval(self):
        cfg = self.cfg("madx")
        result = build_model(cfg)
        self.assertIn(LA_SOURCE, str(result.model.active_adapters))
        result = build_model(replace(cfg, eval_only=True))
        self.assertIn(LA_TARGET, str(result.model.active_adapters))
        self.assertIsNone(result.tlr_plan)
        with self.assertRaises(FileNotFoundError):
            replace(cfg, source_language_adapter="not_available").validate()

    def test_05_save_reload_all_modes(self):
        for mode, tlr in (("full_ft", False), ("adapter", False), ("madx", False), ("madx", True)):
            with self.subTest(mode=mode, tlr=tlr):
                cfg = self.cfg(mode, tlr)
                result = run(cfg, logger=lambda *a: None)
                checkpoint = Path(result["model_checkpoint"])
                self.assertTrue((checkpoint / "model.safetensors").exists())
                with open(Path(cfg.run_dir()) / "predictions.csv", encoding="utf-8-sig") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual([r["source_id"] for r in rows], ["family2", "family3"])
                self.assertEqual(rows[1]["attack_type"], "homo")
                # Remove external LA references: everything must load from the checkpoint.
                evaluation = replace(cfg, exp_name="eval_" + str(tlr), eval_only=True,
                                     model_checkpoint=str(checkpoint), language_adapter="missing",
                                     source_language_adapter="missing")
                replay = run(evaluation, logger=lambda *a: None)
                for key in ("loss", "macro_f1", "accuracy", "toxic_recall", "fpr"):
                    self.assertEqual(result["metrics"][key], replay["metrics"][key])
                first = AutoAdapterModel.from_pretrained(checkpoint)
                second = build_model(evaluation).model
                for key, tensor in first.state_dict().items():
                    self.assertTrue(torch.equal(tensor, second.state_dict()[key]), key)
                if tlr:
                    self.assertEqual(result["tlr"]["sequence_head"], [LA_SOURCE, LA_TARGET])
                with self.assertRaises(FileExistsError):
                    run(cfg, logger=lambda *a: None)

    def test_06_invalid_labels_and_empty(self):
        path = self.root / "invalid.jsonl"
        for payload in ("", '{"text":"valid", "label":2}', '{"text":"valid", "label":0.5}'):
            path.write_text(payload, encoding="utf-8")
            with self.assertRaises(ValueError):
                read_jsonl(path)

    def corpus_fixture(self):
        corpus = self.root / "corpus.jsonl"
        rows = [dict(id=str(i), text=text, source="synthetic", source_split="train")
                for i, text in enumerate(["good text", "bad text", "other text", "sample text",
                                           "other sample", "good sample", "bad sample", "sample other"])]
        corpus.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        frozen = self.root / "frozen.json"
        frozen.write_text(json.dumps([normalized_hash("excluded test")]), encoding="utf-8")
        audit = self.root / "audit.json"
        audit.write_text(json.dumps(dict(schema_version=1, passed=True, rows=len(rows),
                        policy="official_dev_test_and_full_CNTP_and_table2_reserved",
                        corpus_sha256=sha256_file(corpus), frozen_sha256=sha256_file(frozen))), encoding="utf-8")
        return corpus, audit, frozen

    def test_07_dapt_training_resume_and_downstream_load(self):
        corpus, audit, frozen = self.corpus_fixture()
        args = dapt.parser().parse_args(["--corpus", str(corpus), "--audit", str(audit),
                    "--frozen-hashes", str(frozen), "--backbone", str(self.base),
                    "--output", str(self.root / "dapt"), "--device", "cpu", "--fraction", "1",
                    "--batch-size", "2", "--gradient-accumulation", "1", "--max-length", "12",
                    "--max-steps", "2"])
        result = dapt.run(args)
        self.assertEqual(result["global_step"], 2)
        self.assertEqual(result["optimizer_step_values"], [2])
        self.assertTrue(np.isfinite(result["final_eval_loss"]))
        with self.assertRaises(FileExistsError):
            dapt.run(args)
        args.resume_from_checkpoint = str(self.root / "dapt/checkpoints/checkpoint-2")
        args.max_steps = 3
        resumed = dapt.run(args)
        self.assertEqual(resumed["global_step"], 3)
        self.assertEqual(resumed["optimizer_step_values"], [3])
        cfg = replace(self.cfg(), backbone=result["final_model"])
        self.assertIn(cfg.head_name, build_model(cfg).model.heads)

    def test_08_audit_rejects_modified_corpus(self):
        corpus, audit, frozen = self.corpus_fixture()
        read_verified_corpus(corpus, audit, frozen)
        with corpus.open("a", encoding="utf-8") as f:
            f.write('{}\n')
        with self.assertRaises(ValueError):
            read_verified_corpus(corpus, audit, frozen)

    def test_09_grouped_mlm_split(self):
        rows = [dict(id=str(i), text=text) for i, text in enumerate(["AbC!", "abc", "good", "bad", "other"])]
        train, validation = dapt.select_rows(rows, 1, 0.4, 42)
        self.assertFalse({normalized_hash(r["text"]) for r in train}
                         & {normalized_hash(r["text"]) for r in validation})

    def test_10_prediction_count_mismatch(self):
        fake = SimpleNamespace(predict=lambda *a, **k: SimpleNamespace(
            predictions=np.array([[2, 0]]), label_ids=np.array([0]), metrics={}))
        class Rows:
            rows = self.rows
            def __len__(self):
                return 2
        with self.assertRaises(ValueError):
            evaluate_split(fake, Rows(), "dev", str(self.root), lambda *a: None, "eval", "bad.csv")

    def test_11_supervised_group_split_guard(self):
        from src.data import load_datasets
        cfg = replace(self.cfg(), dev_file=str(self.data))
        with self.assertRaises(ValueError):
            load_datasets(cfg, self.tokenizer)
        path = self.root / "group_leak.jsonl"
        path.write_text(json.dumps(dict(text="other text", label=0, source_id="different-row",
                                       seed_normalized_hash="shared-parent")), encoding="utf-8")
        train = self.root / "group_train.jsonl"
        train.write_text(json.dumps(dict(text="good text", label=0, source_id="different-row-2",
                                        seed_normalized_hash="shared-parent")), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_datasets(replace(self.cfg(), train_file=str(train), dev_file=str(path)), self.tokenizer)


    def test_12_reject_all_skipped_optimizer_updates(self):
        for states in ({}, {0: {}}, {0: {"step": torch.tensor(0.)}}):
            with self.subTest(states=states), self.assertRaises(RuntimeError):
                dapt.checked_optimizer_steps(SimpleNamespace(state=states))
        states = {0: {"step": torch.tensor(10.)}, 1: {"step": 10}}
        self.assertEqual(dapt.checked_optimizer_steps(SimpleNamespace(state=states)), [10])


if __name__ == "__main__":
    unittest.main()
