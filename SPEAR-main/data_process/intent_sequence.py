import argparse
import json
from pathlib import Path

from spare.data import (
    load_reports,
    process_calls,
    process_identifier,
    report_processes,
    resource_linked_pattern_indices,
)


def call_matches(call, key, value, matching_category=None, keyset=None):
    for argument in call.get("arguments", []):
        if isinstance(argument, dict) and argument.get("name") == key and argument.get("value") == value:
            return True
    return call.get("return") == value


def extract_report_patterns(report):
    extracted = []
    for process_index, process in enumerate(report_processes(report)):
        calls = process_calls(process)
        patterns = []
        for indices in resource_linked_pattern_indices(process):
            patterns.append(
                {
                    "event_indices": list(indices),
                    "calls": [calls[index] for index in indices],
                }
            )
        extracted.append(
            {
                "process_id": process_identifier(process, str(process_index)),
                "patterns": patterns,
            }
        )
    return extracted


def extract_and_save_patten(input_dirs, output_base_dir, keyset=None):
    output_dir = Path(output_base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    report_index = 0
    for input_path in input_dirs:
        for report in load_reports(str(input_path)):
            patterns = extract_report_patterns(report)
            if not any(process["patterns"] for process in patterns):
                report_index += 1
                continue
            source_name = Path(input_path).stem
            output_path = output_dir / f"{source_name}_{report_index}.json"
            with output_path.open("w", encoding="utf-8") as stream:
                json.dump(patterns, stream, ensure_ascii=False)
            written.append(str(output_path))
            report_index += 1
    return written


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    written = extract_and_save_patten(args.input, args.output)
    print(json.dumps({"written": len(written)}))


if __name__ == "__main__":
    main()
