import json
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


RESOURCE_TYPES = ("file", "registry", "process_thread", "memory")
RETURN_KEYS = ("return", "return_value", "retval", "result")
CALL_KEYS = ("calls", "apis", "events", "actions")
IGNORED_TEXT_KEYS = {"ttps"}


def _first(mapping: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def api_name(call: Dict[str, Any]) -> str:
    value = _first(call, ("api", "api_name", "name", "action"), "")
    return str(value).strip()


def process_identifier(process: Dict[str, Any], fallback: str) -> str:
    value = _first(process, ("process_id", "pid", "id", "process_name", "name"), fallback)
    return str(value)


def process_calls(process: Any) -> List[Dict[str, Any]]:
    if isinstance(process, list):
        return [call for call in process if isinstance(call, dict)]
    if not isinstance(process, dict):
        return []
    calls = _first(process, CALL_KEYS, [])
    if isinstance(calls, dict):
        calls = list(calls.values())
    if not isinstance(calls, list):
        return []
    return [call for call in calls if isinstance(call, dict)]


def report_processes(report: Any) -> List[Dict[str, Any]]:
    if isinstance(report, list):
        if all(isinstance(item, dict) and api_name(item) for item in report):
            return [{"process_id": "0", "calls": report}]
        processes = []
        for item in report:
            if isinstance(item, dict) and process_calls(item):
                processes.append(item)
        return processes
    if not isinstance(report, dict):
        return []
    behavior = report.get("behavior")
    if isinstance(behavior, dict) and isinstance(behavior.get("processes"), list):
        return [item for item in behavior["processes"] if isinstance(item, dict)]
    processes = report.get("processes")
    if isinstance(processes, list):
        return [item for item in processes if isinstance(item, dict)]
    if process_calls(report):
        return [report]
    return []


def _iter_scalars(value: Any) -> Iterator[Any]:
    if isinstance(value, dict):
        for nested in value.values():
            yield from _iter_scalars(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _iter_scalars(nested)
    elif value is not None and not isinstance(value, bool):
        yield value


def argument_items(call: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    arguments = _first(call, ("arguments", "args", "parameters", "api_args"), [])
    if isinstance(arguments, dict):
        for name, value in arguments.items():
            yield str(name), value
        return
    if not isinstance(arguments, list):
        return
    for index, argument in enumerate(arguments):
        if isinstance(argument, dict):
            if "name" in argument:
                yield str(argument["name"]), argument.get("value")
            elif len(argument) == 1:
                name, value = next(iter(argument.items()))
                yield str(name), value
            else:
                for name, value in argument.items():
                    yield str(name), value
        else:
            yield str(index), argument


def infer_resource_type(api: str, field: str) -> Optional[str]:
    api_lower = api.lower()
    field_lower = field.lower()
    if any(term in field_lower for term in ("registry", "regkey", "hkey", "keyhandle", "keyname")):
        return "registry"
    if any(term in field_lower for term in ("memory", "virtual", "address", "baseaddress", "heap")):
        return "memory"
    if any(term in field_lower for term in ("process", "thread", "processid", "threadid", "pid", "tid")):
        return "process_thread"
    if any(term in field_lower for term in ("file", "path", "directory", "filename")):
        return "file"
    if "handle" in field_lower or field_lower.startswith("h"):
        if "reg" in api_lower or "key" in api_lower:
            return "registry"
        if any(term in api_lower for term in ("memory", "virtual", "heap")):
            return "memory"
        if any(term in api_lower for term in ("file", "directory", "find")):
            return "file"
        if "process" in api_lower or "thread" in api_lower:
            return "process_thread"
    if "reg" in api_lower or "key" in api_lower:
        return "registry"
    if any(term in api_lower for term in ("memory", "virtual", "address", "heap")):
        return "memory"
    if any(term in api_lower for term in ("file", "directory", "find")):
        return "file"
    if "process" in api_lower or "thread" in api_lower:
        return "process_thread"
    return None


def _resource_value(value: Any) -> Optional[str]:
    text = str(value).strip()
    if not text or text.lower() in {
        "none",
        "null",
        "false",
        "true",
        "0",
        "0x0",
        "0x00000000",
        "-1",
        "0xffffffff",
        "invalid_handle_value",
    }:
        return None
    return text.lower()


class _DisjointSet:
    def __init__(self) -> None:
        self.parent: Dict[Tuple[str, str], Tuple[str, str]] = {}

    def find(self, item: Tuple[str, str]) -> Tuple[str, str]:
        self.parent.setdefault(item, item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: Tuple[str, str], right: Tuple[str, str]) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def call_resource_references(call: Dict[str, Any]) -> List[Tuple[str, str]]:
    name = api_name(call)
    references: List[Tuple[str, str]] = []
    for field, value in argument_items(call):
        resource_type = infer_resource_type(name, field)
        if resource_type is None and "handle" in field.lower():
            resource_type = "unknown"
        if resource_type is None:
            continue
        for scalar in _iter_scalars(value):
            normalized = _resource_value(scalar)
            if normalized is not None:
                references.append((resource_type, normalized))
    for key in RETURN_KEYS:
        if key not in call:
            continue
        resource_type = infer_resource_type(name, key)
        if resource_type is None:
            resource_type = infer_resource_type(name, "handle")
        if resource_type is None and "handle" in name.lower():
            resource_type = "unknown"
        if resource_type is None:
            continue
        for scalar in _iter_scalars(call[key]):
            normalized = _resource_value(scalar)
            if normalized is not None:
                references.append((resource_type, normalized))
    return list(dict.fromkeys(references))


def resource_linked_pattern_indices(process: Dict[str, Any]) -> List[Tuple[int, ...]]:
    calls = process_calls(process)
    disjoint = _DisjointSet()
    call_references: List[List[Tuple[str, str]]] = []
    for call in calls:
        references = call_resource_references(call)
        call_references.append(references)
        by_type: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for reference in references:
            by_type[reference[0]].append(reference)
        for references_of_type in by_type.values():
            for reference in references_of_type[1:]:
                disjoint.union(references_of_type[0], reference)
    typed_by_value: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    unknown_references = []
    for references in call_references:
        for reference in references:
            if reference[0] == "unknown":
                unknown_references.append(reference)
            else:
                typed_by_value[reference[1]].append(reference)
    for reference in unknown_references:
        typed = list(dict.fromkeys(typed_by_value.get(reference[1], [])))
        if len({item[0] for item in typed}) == 1 and typed:
            disjoint.union(typed[0], reference)
    grouped: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for index, references in enumerate(call_references):
        for root in {disjoint.find(reference) for reference in references}:
            grouped[root].append(index)
    unique = {
        tuple(dict.fromkeys(indices))
        for root, indices in grouped.items()
        if root[0] != "unknown"
        if len(set(indices)) >= 2
    }
    return sorted(unique, key=lambda indices: (indices[0], indices[-1], len(indices), indices))


def resource_linked_patterns(process: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    calls = process_calls(process)
    return [[calls[index] for index in indices] for indices in resource_linked_pattern_indices(process)]


def _flatten_text(value: Any, output: List[str]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).lower() in IGNORED_TEXT_KEYS:
                continue
            output.append(str(key))
            _flatten_text(nested, output)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _flatten_text(nested, output)
    elif value is not None:
        output.append(str(value))


def serialize_call(call: Dict[str, Any]) -> str:
    output: List[str] = []
    _flatten_text(call, output)
    return " ".join(output)


def serialize_calls(calls: Iterable[Dict[str, Any]]) -> str:
    return " ".join(serialize_call(call) for call in calls).strip()


def serialize_report(report: Any) -> str:
    return " ".join(
        serialize_calls(process_calls(process))
        for process in report_processes(report)
    ).strip()


def csm_candidates_for_report(
    report: Any,
    rng: Optional[random.Random] = None,
) -> Tuple[List[List[Tuple[str, str]]], List[Tuple[str, List[str]]]]:
    rng = rng or random.Random()
    pools: List[List[Tuple[str, str]]] = [[] for _ in range(5)]
    process_patterns: List[Tuple[str, List[str]]] = []
    for process_index, process in enumerate(report_processes(report)):
        calls = [call for call in process_calls(process) if api_name(call)]
        serialized_calls = [serialize_call(call) for call in calls]
        for left, right in zip(serialized_calls, serialized_calls[1:]):
            pools[0].append((left, right))
            pools[1].append((right, left))
        patterns = [serialize_calls(pattern) for pattern in resource_linked_patterns(process)]
        patterns = [pattern for pattern in patterns if pattern]
        for left, right in zip(patterns, patterns[1:]):
            pools[2].append((left, right))
            pools[3].append((right, left))
        identifier = process_identifier(process, str(process_index))
        process_patterns.append((identifier, patterns))
    for left_index in range(len(process_patterns)):
        for right_index in range(left_index + 1, len(process_patterns)):
            left_patterns = process_patterns[left_index][1]
            right_patterns = process_patterns[right_index][1]
            if not left_patterns or not right_patterns:
                continue
            count = max(len(left_patterns), len(right_patterns))
            for _ in range(count):
                pools[4].append((rng.choice(left_patterns), rng.choice(right_patterns)))
    return pools, process_patterns


def add_cross_report_csm_pairs(
    pool: List[Tuple[str, str]],
    process_patterns: Sequence[Tuple[str, List[str]]],
    count: int,
    rng: random.Random,
) -> None:
    eligible = [(identifier, patterns) for identifier, patterns in process_patterns if patterns]
    if len(eligible) < 2:
        return
    attempts = 0
    while len(pool) < count and attempts < count * 20:
        left, right = rng.sample(eligible, 2)
        attempts += 1
        if left[0] == right[0]:
            continue
        pool.append((rng.choice(left[1]), rng.choice(right[1])))


def _looks_like_call(value: Any) -> bool:
    return isinstance(value, dict) and bool(api_name(value))


def load_reports(path: str) -> Iterator[Any]:
    source = Path(path)
    if source.is_dir():
        for child in sorted(source.rglob("*.json")):
            yield from load_reports(str(child))
        for child in sorted(source.rglob("*.jsonl")):
            yield from load_reports(str(child))
        for child in sorted(source.rglob("*.xml")):
            yield from load_reports(str(child))
        return
    if source.suffix.lower() == ".jsonl":
        with source.open("r", encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    if source.suffix.lower() == ".json":
        with source.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
        if isinstance(value, list) and value and not all(_looks_like_call(item) for item in value):
            if all(isinstance(item, dict) and report_processes(item) for item in value):
                yield from value
                return
        yield value
        return
    if source.suffix.lower() == ".xml":
        root = ET.parse(source).getroot()
        processes: Dict[str, Dict[str, Any]] = {}
        for action_index, action in enumerate(root.findall(".//action")):
            process_id = str(
                _first(action.attrib, ("process_id", "pid", "process", "processId"), "0")
            )
            process = processes.setdefault(process_id, {"process_id": process_id, "calls": []})
            call: Dict[str, Any] = {
                "api": _first(action.attrib, ("api_name", "api", "name"), f"action_{action_index}"),
                "arguments": [],
            }
            for argument in action.findall(".//apiArg"):
                call["arguments"].append(
                    {
                        "name": _first(argument.attrib, ("name", "key", "type"), "arg"),
                        "value": _first(argument.attrib, ("value", "val"), argument.text or ""),
                    }
                )
            returned = _first(action.attrib, RETURN_KEYS)
            if returned is not None:
                call["return"] = returned
            process["calls"].append(call)
        yield {"processes": list(processes.values())}
        return
    with source.open("r", encoding="utf-8") as stream:
        for line in stream:
            text = line.strip()
            if text:
                yield {"serialized_trace": text}
