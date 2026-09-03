import argparse
import pickle
import random
from pathlib import Path

from spare.data import (
    add_cross_report_csm_pairs,
    csm_candidates_for_report,
    load_reports,
    serialize_report,
)
from spare.tokenization import normalize_api_text


def build_pretraining_data(paths, seed=7, max_examples_per_class=0):
    rng = random.Random(seed)
    full_traces = []
    csm_pools = [[] for _ in range(5)]
    global_process_patterns = []
    report_index = 0
    for path in paths:
        for report in load_reports(path):
            if isinstance(report, dict) and "serialized_trace" in report:
                trace = normalize_api_text(report["serialized_trace"])
                if trace:
                    full_traces.append(trace)
                report_index += 1
                continue
            trace = normalize_api_text(serialize_report(report))
            if not trace:
                report_index += 1
                continue
            full_traces.append(trace)
            pools, process_patterns = csm_candidates_for_report(report, rng)
            for label, pool in enumerate(pools):
                csm_pools[label].extend(
                    (normalize_api_text(left), normalize_api_text(right))
                    for left, right in pool
                    if left and right
                )
            global_process_patterns.extend(
                (
                    f"{report_index}:{identifier}",
                    [normalize_api_text(pattern) for pattern in patterns if pattern],
                )
                for identifier, patterns in process_patterns
            )
            report_index += 1
    target_cross_process = max((len(pool) for pool in csm_pools[:4]), default=0)
    add_cross_report_csm_pairs(
        csm_pools[4],
        global_process_patterns,
        target_cross_process,
        rng,
    )
    if max_examples_per_class > 0:
        for pool in csm_pools:
            rng.shuffle(pool)
            del pool[max_examples_per_class:]
    return {
        "format": "spear-pretraining-v2",
        "full_traces": full_traces,
        "csm_pools": csm_pools,
        "seed": seed,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_path", nargs="+", required=True)
    parser.add_argument("--dataset_path", default="dataset.pt")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max_examples_per_class", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    payload = build_pretraining_data(
        args.corpus_path,
        seed=args.seed,
        max_examples_per_class=args.max_examples_per_class,
    )
    if not payload["full_traces"]:
        raise ValueError("No full API traces were extracted.")
    empty_classes = [index + 1 for index, pool in enumerate(payload["csm_pools"]) if not pool]
    if empty_classes:
        raise ValueError(f"No CSM candidates were extracted for classes {empty_classes}.")
    output = Path(args.dataset_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    counts = [len(pool) for pool in payload["csm_pools"]]
    print(f"full_traces={len(payload['full_traces'])} csm_classes={counts}")


if __name__ == "__main__":
    main()
