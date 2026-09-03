import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from spare.data import api_name, load_reports, process_calls, report_processes


SPLITS = ("train", "validation", "test")
SPLIT_RATIOS = (0.7, 0.1, 0.2)
AVAST_WINDOWS = {
    "train": (datetime(2017, 9, 1, tzinfo=timezone.utc), datetime(2019, 4, 1, tzinfo=timezone.utc)),
    "validation": (datetime(2019, 4, 1, tzinfo=timezone.utc), datetime(2019, 6, 1, tzinfo=timezone.utc)),
    "test": (datetime(2019, 6, 1, tzinfo=timezone.utc), datetime(2019, 12, 1, tzinfo=timezone.utc)),
}
AVAST_HELD_OUT_FAMILIES = {"adload", "harhar"}
AVAST_EXPECTED_COUNTS = {"train": 23424, "validation": 4291, "test": 11504}


class _DisjointSet:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item):
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left, right):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def sample_identifier(record, index=None):
    for key in ("sample_id", "sha256", "id", "source_path", "path"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    if index is None:
        raise ValueError("Record has no stable identifier.")
    return str(index)


def _sha256(record):
    value = record.get("sha256", record.get("sha_256", ""))
    return str(value).strip().lower()


def deduplicate_and_remove_pretraining_overlap(records, pretraining_hashes=()):
    overlap = {str(value).strip().lower() for value in pretraining_hashes if str(value).strip()}
    seen = set()
    retained = []
    duplicate_count = 0
    overlap_count = 0
    for record in records:
        digest = _sha256(record)
        if digest:
            if digest in seen:
                duplicate_count += 1
                continue
            seen.add(digest)
            if digest in overlap:
                overlap_count += 1
                continue
        retained.append(record)
    return retained, {"duplicates": duplicate_count, "pretraining_overlap": overlap_count}


def api_name_ngrams(names, size=3):
    normalized = [str(name) for name in names if str(name)]
    if len(normalized) < size:
        return set()
    return {tuple(normalized[index:index + size]) for index in range(len(normalized) - size + 1)}


def jaccard(left, right):
    if not left and not right:
        return 1.0
    union = len(left | right)
    return len(left & right) / union if union else 0.0


def behavior_clusters(signatures, threshold=0.9):
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1].")
    disjoint = _DisjointSet(len(signatures))
    inverted = defaultdict(list)
    empty = []
    for index, signature in enumerate(signatures):
        signature = set(signature)
        if not signature:
            empty.append(index)
            continue
        candidates = set()
        for gram in signature:
            candidates.update(inverted[gram])
        for candidate in candidates:
            other = signatures[candidate]
            larger = max(len(signature), len(other))
            smaller = min(len(signature), len(other))
            if larger and smaller / larger < threshold:
                continue
            if jaccard(signature, other) >= threshold:
                disjoint.union(index, candidate)
        for gram in signature:
            inverted[gram].append(index)
    for index in empty[1:]:
        disjoint.union(empty[0], index)
    groups = defaultdict(list)
    for index in range(len(signatures)):
        groups[disjoint.find(index)].append(index)
    return list(groups.values())


def stratified_group_assignment(records, groups, seed=42, ratios=SPLIT_RATIOS):
    if len(ratios) != 3 or any(ratio < 0 for ratio in ratios) or sum(ratios) <= 0:
        raise ValueError("ratios must contain three non-negative values.")
    ratio_sum = sum(ratios)
    ratios = tuple(ratio / ratio_sum for ratio in ratios)
    labels = [str(record["label"]) for record in records]
    label_totals = Counter(labels)
    total = len(records)
    targets = {
        split: {
            label: label_totals[label] * ratios[split_index]
            for label in label_totals
        }
        for split_index, split in enumerate(SPLITS)
    }
    total_targets = {split: total * ratios[index] for index, split in enumerate(SPLITS)}
    rng = random.Random(seed)
    shuffled = list(groups)
    rng.shuffle(shuffled)
    shuffled.sort(key=lambda group: len(group), reverse=True)
    assignments = {split: [] for split in SPLITS}
    label_counts = {split: Counter() for split in SPLITS}
    total_counts = Counter()
    for group in shuffled:
        group_labels = Counter(labels[index] for index in group)
        best_splits = []
        best_score = None
        for split in SPLITS:
            score = 0.0
            for candidate_split in SPLITS:
                for label in label_totals:
                    value = label_counts[candidate_split][label]
                    if candidate_split == split:
                        value += group_labels[label]
                    target = targets[candidate_split][label]
                    score += ((value - target) ** 2) / max(target, 1.0)
                value = total_counts[candidate_split]
                if candidate_split == split:
                    value += len(group)
                target = total_targets[candidate_split]
                score += ((value - target) ** 2) / max(target, 1.0)
            if best_score is None or score < best_score - 1e-12:
                best_score = score
                best_splits = [split]
            elif abs(score - best_score) <= 1e-12:
                best_splits.append(split)
        selected = rng.choice(best_splits)
        assignments[selected].extend(group)
        label_counts[selected].update(group_labels)
        total_counts[selected] += len(group)
    return assignments


def _timestamp(value):
    text = str(value).strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def avast_temporal_assignment(records):
    assignments = {split: [] for split in SPLITS}
    removed = []
    for index, record in enumerate(records):
        family = str(record.get("source_family", record.get("family", ""))).lower()
        if family in AVAST_HELD_OUT_FAMILIES:
            assignments["test"].append(index)
            continue
        value = record.get("first_seen", record.get("timestamp"))
        if value is None:
            raise ValueError("Avast-CTU records require first_seen timestamps.")
        observed = _timestamp(value)
        assigned = False
        for split, (start, end) in AVAST_WINDOWS.items():
            if start <= observed < end:
                assignments[split].append(index)
                assigned = True
                break
        if not assigned:
            removed.append(index)
    return assignments, removed


def normalize_official_assignments(value):
    if isinstance(value, dict) and "splits" in value:
        value = value["splits"]
    if all(split in value for split in SPLITS):
        return {
            split: {str(identifier) for identifier in value[split]}
            for split in SPLITS
        }
    assignments = {split: set() for split in SPLITS}
    for identifier, split in value.items():
        normalized = "validation" if split in {"dev", "valid", "validation"} else str(split)
        if normalized not in assignments:
            raise ValueError(f"Unknown split {split}.")
        assignments[normalized].add(str(identifier))
    return assignments


def apply_official_assignments(records, official):
    lookup = normalize_official_assignments(official)
    assigned = {split: [] for split in SPLITS}
    seen = set()
    for index, record in enumerate(records):
        identifier = sample_identifier(record, index)
        matches = [split for split in SPLITS if identifier in lookup[split]]
        if len(matches) != 1:
            raise ValueError(f"Official assignment for {identifier} is missing or ambiguous.")
        assigned[matches[0]].append(index)
        seen.add(identifier)
    expected = set().union(*lookup.values())
    if seen != expected:
        raise ValueError("Official assignments contain identifiers absent after Stage 1.")
    return assigned


def _record_api_signature(record, base_path):
    names = record.get("api_names")
    if isinstance(names, str):
        return api_name_ngrams([name for name in names.replace(",", " ").split() if name])
    if isinstance(names, list):
        return api_name_ngrams([str(name) for name in names])
    report = record.get("report")
    if report is None:
        path = record.get("source_path", record.get("path"))
        if path is None:
            raise ValueError("Cluster splitting requires api_names, report, or source_path.")
        source = Path(path)
        if not source.is_absolute():
            source = base_path / source
        report = next(load_reports(str(source)))
    signature = set()
    for process in report_processes(report):
        names = [api_name(call) for call in process_calls(process) if api_name(call)]
        signature.update(api_name_ngrams(names))
    return signature


def read_records(path):
    source = Path(path)
    if source.suffix.lower() == ".tsv":
        with source.open("r", encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream, delimiter="\t"))
    if source.suffix.lower() == ".jsonl":
        with source.open("r", encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    with source.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    return value if isinstance(value, list) else value.get("records", [value])


def read_hashes(path):
    if path is None:
        return set()
    source = Path(path)
    if source.suffix.lower() in {".json", ".jsonl", ".tsv"}:
        records = read_records(path)
        hashes = set()
        for record in records:
            digest = _sha256(record) if isinstance(record, dict) else str(record).strip().lower()
            if digest:
                hashes.add(digest)
        return hashes
    with source.open("r", encoding="utf-8") as stream:
        return {line.strip().lower() for line in stream if line.strip()}


def write_outputs(output_dir, records, assignments, metadata):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"metadata": metadata, "splits": {}}
    for split in SPLITS:
        path = output / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for index in assignments[split]:
                stream.write(json.dumps(records[index], ensure_ascii=False) + "\n")
        manifest["splits"][split] = [sample_identifier(records[index], index) for index in assignments[split]]
    with (output / "split_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", choices=["avast-ctu", "catak", "nunes", "mcd"], required=True)
    parser.add_argument("--pretraining_hashes")
    parser.add_argument("--official_assignments")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--validate_paper_counts", action="store_true")
    args = parser.parse_args()
    records = read_records(args.input)
    records, stage1 = deduplicate_and_remove_pretraining_overlap(
        records,
        read_hashes(args.pretraining_hashes),
    )
    if args.official_assignments:
        with Path(args.official_assignments).open("r", encoding="utf-8") as stream:
            assignments = apply_official_assignments(records, json.load(stream))
        removed_stage2 = 0
    elif args.dataset == "avast-ctu":
        assignments, removed = avast_temporal_assignment(records)
        removed_stage2 = len(removed)
    else:
        base_path = Path(args.input).parent
        signatures = [_record_api_signature(record, base_path) for record in records]
        groups = behavior_clusters(signatures, threshold=args.threshold)
        assignments = stratified_group_assignment(records, groups, seed=args.seed)
        removed_stage2 = 0
    counts = {split: len(assignments[split]) for split in SPLITS}
    if args.validate_paper_counts:
        if args.dataset != "avast-ctu" or counts != AVAST_EXPECTED_COUNTS:
            raise ValueError(f"Split counts do not match the paper: {counts}")
    metadata = {
        "dataset": args.dataset,
        "stage1_removed": stage1,
        "stage2_removed": removed_stage2,
        "counts": counts,
        "seed": args.seed,
        "jaccard_threshold": args.threshold if args.dataset != "avast-ctu" else None,
    }
    write_outputs(args.output, records, assignments, metadata)
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
