import argparse
import copy
import json
import random
import re
from pathlib import Path

from data_process.data_split import SPLITS, read_records, sample_identifier
from spare.data import (
    api_name,
    infer_resource_type,
    load_reports,
    report_processes,
    resource_linked_patterns,
    serialize_calls,
)
from spare.tokenization import normalize_api_text


SIZE_TERMS = ("size", "length", "bytes", "numberofbytes", "regionsize")


def _report(record, base_path):
    value = record.get("report")
    if value is None:
        value = record.get("source_path", record.get("path"))
    if isinstance(value, (dict, list)):
        return value
    if value is None:
        raise ValueError("Probe records require report, source_path, or path.")
    source = Path(value)
    if not source.is_absolute():
        source = base_path / source
    return next(load_reports(str(source)))


def _patterns(report):
    return [
        pattern
        for process in report_processes(report)
        for pattern in resource_linked_patterns(process)
        if len(pattern) >= 2
    ]


def _binary_examples(pattern, rng):
    complete = normalize_api_text(serialize_calls(pattern))
    removed_index = rng.randrange(len(pattern))
    incomplete_calls = pattern[:removed_index] + pattern[removed_index + 1 :]
    incomplete = normalize_api_text(serialize_calls(incomplete_calls))
    order = list(range(len(pattern)))
    rng.shuffle(order)
    if order == list(range(len(pattern))):
        order = order[1:] + order[:1]
    disordered = normalize_api_text(serialize_calls(pattern[index] for index in order))
    incomplete_examples = []
    disorder_examples = []
    if incomplete and incomplete != complete:
        incomplete_examples = [
            {"text": complete, "label": "0"},
            {"text": incomplete, "label": "1"},
        ]
    if disordered and disordered != complete:
        disorder_examples = [
            {"text": complete, "label": "0"},
            {"text": disordered, "label": "1"},
        ]
    return incomplete_examples, disorder_examples


def _argument_locations(call):
    arguments = call.get("arguments", call.get("args", call.get("parameters", call.get("api_args", []))))
    if isinstance(arguments, dict):
        for key in list(arguments):
            yield arguments, key, str(key)
    elif isinstance(arguments, list):
        for argument in arguments:
            if not isinstance(argument, dict):
                continue
            if "name" in argument and "value" in argument:
                yield argument, "value", str(argument["name"])
            elif len(argument) == 1:
                key = next(iter(argument))
                yield argument, key, str(key)


def _parameter_kind(api, field, value):
    text = str(value).strip()
    field_lower = field.lower().replace("_", "")
    resource_type = infer_resource_type(api, field)
    if resource_type == "file" and re.search(r"(^[A-Za-z]:\\|^\\\\|/)", text):
        return "file_path"
    if resource_type == "registry" or text.lower().startswith(("hkey_", "hklm\\", "hkcu\\")):
        return "registry_key"
    if resource_type == "memory" and any(term in field_lower for term in SIZE_TERMS):
        return "memory_size"
    if any(term in api.lower() for term in ("memory", "virtual", "heap")) and any(
        term in field_lower for term in SIZE_TERMS
    ):
        return "memory_size"
    return None


def _parameter_example(pattern, rng):
    copied = copy.deepcopy(pattern)
    eligible = []
    for call in copied:
        name = api_name(call)
        for container, key, field in _argument_locations(call):
            value = container[key]
            if isinstance(value, (dict, list)):
                continue
            kind = _parameter_kind(name, field, value)
            if kind is not None and str(value).strip():
                eligible.append((container, key, kind, str(value)))
    if not eligible:
        return None
    container, key, kind, target = rng.choice(eligible)
    container[key] = "[MASK]"
    text = normalize_api_text(serialize_calls(copied))
    return {"text": text, "label": target, "field_type": kind}


def generate_probes(records, base_path, seed):
    rng = random.Random(seed)
    outputs = {"incompleteness": [], "disorder": [], "parameter": []}
    for record_index, record in enumerate(records):
        identifier = sample_identifier(record, record_index)
        for pattern_index, pattern in enumerate(_patterns(_report(record, base_path))):
            incompleteness, disorder = _binary_examples(pattern, rng)
            parameter = _parameter_example(pattern, rng)
            source = {"source_id": identifier, "pattern_index": pattern_index}
            outputs["incompleteness"].extend({**example, **source} for example in incompleteness)
            outputs["disorder"].extend({**example, **source} for example in disorder)
            if parameter is not None:
                outputs["parameter"].append({**parameter, **source})
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split_dir = Path(args.split_dir)
    output = Path(args.output)
    counts = {}
    for split_index, split in enumerate(SPLITS):
        source = split_dir / f"{split}.jsonl"
        records = read_records(str(source))
        generated = generate_probes(records, source.parent, args.seed + split_index)
        counts[split] = {}
        for probe, examples in generated.items():
            destination = output / probe / f"{split}.jsonl"
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("w", encoding="utf-8") as stream:
                for example in examples:
                    stream.write(json.dumps(example, ensure_ascii=False) + "\n")
            counts[split][probe] = len(examples)
    print(json.dumps(counts, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
