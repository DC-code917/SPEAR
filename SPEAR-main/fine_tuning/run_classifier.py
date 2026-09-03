import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data_process.data_augment import generate_candidate_pool
from spare.data import load_reports, serialize_report
from spare.modeling import SpearClassifier, SpearConfig, encoder_state_dict
from spare.tokenization import SpearTokenizer, normalize_api_text


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_source_report(value, base_path):
    if isinstance(value, dict) or isinstance(value, list):
        return value
    source = Path(value)
    if not source.is_absolute():
        source = base_path / source
    return next(load_reports(str(source)))


def read_records(path):
    source = Path(path)
    base_path = source.parent
    if source.suffix.lower() == ".tsv":
        with source.open("r", encoding="utf-8", newline="") as stream:
            records = list(csv.DictReader(stream, delimiter="\t"))
    elif source.suffix.lower() == ".jsonl":
        with source.open("r", encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
    else:
        with source.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
        records = value if isinstance(value, list) else [value]
    normalized = []
    for record in records:
        record = dict(record)
        if "source_path" in record and "report" not in record:
            record["report"] = _read_source_report(record["source_path"], base_path)
        if "report" in record:
            record["report"] = _read_source_report(record["report"], base_path)
        normalized.append(record)
    return normalized


def _multilabel_values(record):
    value = record.get("labels", record.get("ttps", record.get("label", [])))
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            value = json.loads(stripped)
        else:
            value = [item for item in stripped.replace(",", " ").split() if item]
    if not isinstance(value, (list, tuple, set)):
        value = [value]
    return [str(item) for item in value]


def build_label_mapping(records, multilabel, positive_label="malware"):
    if multilabel:
        labels = sorted({label for record in records for label in _multilabel_values(record)})
    else:
        labels = sorted({str(record["label"]) for record in records})
        if len(labels) == 2 and positive_label in labels:
            labels = [label for label in labels if label != positive_label] + [positive_label]
    if len(labels) < 2:
        raise ValueError("At least two labels are required.")
    return {label: index for index, label in enumerate(labels)}


def record_label(record, mapping, multilabel):
    if multilabel:
        vector = torch.zeros(len(mapping), dtype=torch.float32)
        for label in _multilabel_values(record):
            if label not in mapping:
                raise ValueError(f"Unknown label {label}.")
            vector[mapping[label]] = 1.0
        return vector
    label = str(record["label"])
    if label not in mapping:
        raise ValueError(f"Unknown label {label}.")
    return torch.tensor(mapping[label], dtype=torch.long)


def clean_text(record):
    if "report" in record:
        return normalize_api_text(serialize_report(record["report"]))
    for key in ("text", "text_a", "trace"):
        if key in record:
            return normalize_api_text(record[key])
    raise ValueError("Each record requires report, source_path, text, text_a, or trace.")


class DownstreamDataset(Dataset):
    def __init__(
        self,
        records,
        tokenizer,
        label_mapping,
        max_length=1024,
        multilabel=False,
        augmentation_factor=0,
        seed=7,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.multilabel = multilabel
        self.seed = seed
        self.epoch = 0
        self.examples = []
        for index, record in enumerate(records):
            label = record_label(record, label_mapping, multilabel)
            if augmentation_factor:
                if "report" not in record:
                    raise ValueError("SPEAR+Aug requires structured raw reports in the training split.")
                reports = generate_candidate_pool(
                    record["report"],
                    factor=augmentation_factor,
                    seed=seed + index,
                )
                texts = [normalize_api_text(serialize_report(report)) for report in reports]
            else:
                texts = [clean_text(record)]
            if not texts[0]:
                raise ValueError(f"Record {index} has no serialized API content.")
            self.examples.append((texts, label))

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        texts, label = self.examples[index]
        rng = random.Random((self.seed + 1) * 1000003 + self.epoch * 9176 + index)
        text = texts[rng.randrange(len(texts))]
        encoded = self.tokenizer.encode_full_trace(text, self.max_length)
        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "segment_ids": torch.tensor(encoded["segment_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.bool),
            "labels": label.clone(),
        }


def linear_schedule(optimizer, total_steps, warmup_ratio):
    warmup_steps = int(total_steps * warmup_ratio)

    def multiplier(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(total_steps - step) / float(max(1, total_steps - warmup_steps)),
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def evaluate(model, loader, device, multilabel):
    from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support

    model.eval()
    all_labels = []
    all_logits = []
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].to(device)
            result = model(
                batch["input_ids"].to(device),
                labels=labels,
                segment_ids=batch["segment_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            total_loss += result["loss"].item()
            batches += 1
            all_labels.append(labels.cpu())
            all_logits.append(result["logits"].cpu())
    labels = torch.cat(all_labels).numpy()
    logits = torch.cat(all_logits).numpy()
    metrics = {"loss": total_loss / max(1, batches)}
    if multilabel:
        probabilities = torch.sigmoid(torch.from_numpy(logits)).numpy()
        metrics["macro_auprc"] = float(average_precision_score(labels, probabilities, average="macro"))
        return metrics
    predictions = logits.argmax(axis=1)
    average = "binary" if logits.shape[1] == 2 else "macro"
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average=average,
        pos_label=1,
        zero_division=0,
    )
    metrics.update(
        accuracy=float(accuracy_score(labels, predictions)),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
    )
    return metrics


def load_encoder(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.encoder.load_state_dict(encoder_state_dict(checkpoint), strict=False)
    if missing or unexpected:
        raise ValueError(f"Encoder checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return checkpoint


def save_checkpoint(path, model, config, label_mapping, multilabel, metrics):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": config.to_dict(),
            "label_mapping": label_mapping,
            "multilabel": multilabel,
            "metrics": metrics,
        },
        output,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--dev_path", required=True)
    parser.add_argument("--test_path")
    parser.add_argument("--tokenizer_model_path", "--spm_model_path", dest="tokenizer_model_path", required=True)
    parser.add_argument("--pretrained_model_path")
    parser.add_argument("--output_model_path", default="models/finetuned_model.bin")
    parser.add_argument("--task", choices=["multiclass", "multilabel", "parameter"], default="multiclass")
    parser.add_argument("--positive_label", default="malware")
    parser.add_argument("--augmentation_factor", type=int, default=0)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--epochs_num", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.augmentation_factor < 0:
        raise ValueError("augmentation_factor must be non-negative.")
    set_seed(args.seed)
    multilabel = args.task == "multilabel"
    train_records = read_records(args.train_path)
    dev_records = read_records(args.dev_path)
    test_records = read_records(args.test_path) if args.test_path else None
    mapping_records = train_records + dev_records + (test_records or [])
    label_mapping = build_label_mapping(mapping_records, multilabel, args.positive_label)
    tokenizer = SpearTokenizer(args.tokenizer_model_path)
    checkpoint = None
    if args.pretrained_model_path:
        checkpoint = torch.load(args.pretrained_model_path, map_location="cpu")
    if checkpoint and "config" in checkpoint:
        config = SpearConfig.from_dict(checkpoint["config"])
        if config.max_seq_length != args.seq_length:
            raise ValueError("seq_length must match the pre-trained checkpoint.")
        if config.pad_token_id != tokenizer.pad_id or config.vocab_size != tokenizer.vocab_size:
            raise ValueError("Tokenizer and pre-trained checkpoint do not match.")
    else:
        config = SpearConfig(
            vocab_size=tokenizer.vocab_size,
            max_seq_length=args.seq_length,
            pad_token_id=tokenizer.pad_id,
        )
    model = SpearClassifier(config, len(label_mapping), multilabel=multilabel)
    if args.pretrained_model_path:
        load_encoder(model, args.pretrained_model_path)
    if args.freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
    train_dataset = DownstreamDataset(
        train_records,
        tokenizer,
        label_mapping,
        max_length=args.seq_length,
        multilabel=multilabel,
        augmentation_factor=args.augmentation_factor,
        seed=args.seed,
    )
    dev_dataset = DownstreamDataset(
        dev_records,
        tokenizer,
        label_mapping,
        max_length=args.seq_length,
        multilabel=multilabel,
        seed=args.seed,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=generator,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    total_steps = math.ceil(len(train_dataset) / args.batch_size) * args.epochs_num
    scheduler = linear_schedule(optimizer, total_steps, args.warmup)
    selection_key = "macro_auprc" if multilabel else ("accuracy" if args.task == "parameter" else "f1")
    best_metric = float("-inf")
    for epoch in range(args.epochs_num):
        train_dataset.set_epoch(epoch)
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            result = model(
                batch["input_ids"].to(device),
                labels=batch["labels"].to(device),
                segment_ids=batch["segment_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            result["loss"].backward()
            optimizer.step()
            scheduler.step()
        metrics = evaluate(model, dev_loader, device, multilabel)
        print(json.dumps({"epoch": epoch + 1, "validation": metrics}, sort_keys=True))
        if metrics[selection_key] > best_metric:
            best_metric = metrics[selection_key]
            save_checkpoint(
                args.output_model_path,
                model,
                config,
                label_mapping,
                multilabel,
                metrics,
            )
    best = torch.load(args.output_model_path, map_location=device)
    model.load_state_dict(best["model"])
    if test_records is not None:
        test_dataset = DownstreamDataset(
            test_records,
            tokenizer,
            label_mapping,
            max_length=args.seq_length,
            multilabel=multilabel,
            seed=args.seed,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        print(json.dumps({"test": evaluate(model, test_loader, device, multilabel)}, sort_keys=True))


if __name__ == "__main__":
    main()
