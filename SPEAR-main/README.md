# SPEAR

This implementation follows the TIFS version of SPEAR. It uses one shared BERT-style encoder for Call Sequence Modeling and Masked Sequence Modeling, supports SPEAR+Aug during downstream training, and includes the paper's leakage-controlled splitting and structural-probe workflows.

## Environment

Use Python 3.10 and install the pinned dependencies:

```bash
pip install -r requirements.txt
```

## Pre-training data

Input may be a JSON, JSONL, XML, text file, or a directory containing those files. JSON reports may use `behavior.processes`, `processes`, or a process object containing `calls`, `apis`, `events`, or `actions`.

```bash
python -m pre_training.preprocess \
  --corpus_path corpus_a corpus_b corpus_c \
  --dataset_path artifacts/pretraining.pt
```

The output contains full traces for MSM and five balanced CSM candidate pools. Resource-linked patterns are formed within each process from file, registry, process/thread, and memory references.

## Tokenizer

Train BPE only from the retained pre-training traces. The vocabulary is capped at 50,000 entries and fixes `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, and `[MASK]` at IDs 0 through 4.

```bash
python -m data_process.vocab_gen \
  --dataset artifacts/pretraining.pt \
  --model-prefix artifacts/spear_bpe \
  --vocab-size 50000
```

## Pre-training

The defaults are sequence length 1,024, 12 layers, hidden size 768, feed-forward size 3,072, 12 attention heads, dropout 0.1, AdamW learning rate 2e-5, weight decay 0.01, warm-up ratio 0.2, MSM weight 0.1, 120,000 updates, and checkpoints every 30,000 updates.

```bash
torchrun --standalone --nproc_per_node=3 -m pre_training.pretrain \
  --dataset_path artifacts/pretraining.pt \
  --tokenizer_model_path artifacts/spear_bpe.model \
  --output_model_path artifacts/spear.bin
```

`--batch_size 64` is per process, giving the paper's effective batch size of 192 with three GPUs. MSM masking is generated dynamically for every sampled trace with the 15% and 80/10/10 policy. MSM and CSM use separate mini-batches and separate encoder passes in every update.

## Leakage-controlled partitions

Each input record must contain `label` and may contain `sha256`, `sample_id`, `source_family`, `first_seen`, `api_names`, `report`, or `source_path` as required by the selected dataset.

```bash
python -m data_process.data_split \
  --input records.jsonl \
  --output splits \
  --dataset mcd \
  --pretraining_hashes pretraining_hashes.txt
```

Catak, Nunes, and MCD use transitive connected components over API-name 3-gram Jaccard similarity at 0.90, followed by class-stratified whole-cluster assignment with seed 42 and target proportions 70/10/20. Avast-CTU uses the fixed temporal windows and reserves Adload and HarHar for testing. Pass the released assignment manifest with `--official_assignments` to reproduce the paper's fixed Avast monthly downsampling exactly; `--validate_paper_counts` checks 23,424/4,291/11,504.

## Fine-tuning

TSV, JSON, and JSONL are supported. Each record needs `label` plus `text`, `trace`, `report`, or `source_path`. Validation selects the checkpoint and testing is performed once after training.

```bash
python -m fine_tuning.run_classifier \
  --train_path splits/train.jsonl \
  --dev_path splits/validation.jsonl \
  --test_path splits/test.jsonl \
  --tokenizer_model_path artifacts/spear_bpe.model \
  --pretrained_model_path artifacts/spear.bin \
  --output_model_path artifacts/classifier.bin
```

The defaults are batch size 128, four epochs, AdamW learning rate 2e-5, weight decay 0.01, and warm-up ratio 0.2. Full traces use segment ID 0, mean pooling over all non-padding final-layer states, and an exactly two-logit head for binary data.

For SPEAR+Aug, use the same tokenizer and pre-trained checkpoint and add:

```bash
--augmentation_factor 4
```

The clean trace and four generated variants form a per-trace candidate pool. One member is sampled whenever the training trace is selected, so epoch size and optimizer-update count do not change. Validation and test traces remain clean.

For ATT&CK technique prediction, store the technique identifiers in `labels` or `ttps` and add `--task multilabel`. Selection and evaluation use macro-AUPRC.

## Structural probes

Generate probes only after the downstream partitions are fixed:

```bash
python -m data_process.probe_generation \
  --split_dir splits \
  --output probes
```

Train each probe with the same fine-tuning entry point and add `--freeze_encoder`. Incompleteness and disorder are balanced binary tasks evaluated by F1. Parameter prediction masks one eligible file path, registry-key name, or memory-operation size and uses `--task parameter` for top-1-accuracy validation and evaluation.
