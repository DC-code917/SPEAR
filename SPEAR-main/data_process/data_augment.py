import argparse
import copy
import json
import random
import re
import string
from pathlib import Path

from spare.data import RETURN_KEYS, infer_resource_type, process_calls, report_processes


HEX_PATTERN = re.compile(r"^(0[xX])([0-9a-fA-F]+)$")
DECIMAL_PATTERN = re.compile(r"^[0-9]+$")
HANDLE_TERMS = ("handle", "hfile", "hkey", "hprocess", "hthread", "hmodule", "hservice", "hdevice")
ADDRESS_TERMS = ("address", "baseaddress", "base_address", "virtualaddress", "virtual_address")
PID_TERMS = ("pid", "processid", "process_id")
TID_TERMS = ("tid", "threadid", "thread_id")


def _field_kind(api, field, value):
    name = str(field).lower().replace("-", "_")
    text = str(value).strip()
    if HEX_PATTERN.fullmatch(text):
        if any(term in name for term in ADDRESS_TERMS):
            return "address"
        if any(term in name for term in HANDLE_TERMS):
            return "handle"
        inferred = infer_resource_type(api, name)
        if inferred == "memory":
            return "address"
        if field in RETURN_KEYS and inferred in {"file", "registry", "process_thread"}:
            return "handle"
    if DECIMAL_PATTERN.fullmatch(text):
        if any(term == name or name.endswith("_" + term) for term in PID_TERMS):
            return "pid"
        if any(term == name or name.endswith("_" + term) for term in TID_TERMS):
            return "tid"
    return None


def _hex_style(original, value):
    match = HEX_PATTERN.fullmatch(original)
    prefix, digits = match.groups()
    rendered = format(value, f"0{len(digits)}x")
    if any(character.isalpha() and character.isupper() for character in digits):
        rendered = rendered.upper()
    return prefix + rendered


def _new_hex(original, rng, used, address=False):
    match = HEX_PATTERN.fullmatch(original)
    digits = match.group(2)
    bits = max(4, 4 * len(digits))
    original_value = int(digits, 16)
    maximum = (1 << bits) - 1
    if address:
        boundary = 1 << (bits - 1)
        if original_value >= boundary:
            lower, upper = boundary, maximum
        else:
            lower, upper = 1, max(1, boundary - 1)
    else:
        lower, upper = 1, maximum
    available = upper - lower + 1 - int(lower <= original_value <= upper) - sum(
        1 for value in used if lower <= value <= upper and value != original_value
    )
    if available <= 0:
        raise ValueError("Unable to generate a distinct hexadecimal replacement.")
    candidate = rng.randint(lower, upper)
    attempts = 0
    while candidate in used or candidate == original_value:
        candidate = rng.randint(lower, upper)
        attempts += 1
        if attempts > 10000:
            raise ValueError("Unable to generate a distinct hexadecimal replacement.")
    used.add(candidate)
    return _hex_style(original, candidate)


def _new_decimal(original, rng, used):
    width = len(original)
    upper = max(65535, (10 ** width) - 1)
    candidate = rng.randint(100, upper)
    attempts = 0
    while candidate in used or str(candidate) == original:
        candidate = rng.randint(100, upper)
        attempts += 1
        if attempts > 10000:
            raise ValueError("Unable to generate a distinct decimal replacement.")
    used.add(candidate)
    rendered = str(candidate)
    return rendered.zfill(width) if original.startswith("0") else rendered


def _temporary_filename(value, field):
    if not isinstance(value, str) or "\\" not in value:
        return False
    directory, _, filename = value.rpartition("\\")
    if not directory or not filename:
        return False
    stem = filename.rsplit(".", 1)[0]
    field_hint = "temp" in str(field).lower()
    directory_hint = any(part.lower() in {"temp", "tmp"} for part in directory.split("\\"))
    random_hint = (
        len(stem) >= 6
        and bool(re.search(r"[A-Za-z]", stem))
        and bool(re.search(r"[0-9]", stem))
        and all(character.isalnum() or character in "-_" for character in stem)
    )
    return field_hint or (directory_hint and random_hint)


def _new_filename(original, rng, used):
    directory, _, filename = original.rpartition("\\")
    if "." in filename:
        stem, extension = filename.rsplit(".", 1)
        suffix = "." + extension
    else:
        stem, suffix = filename, ""
    alphabet = string.ascii_letters + string.digits
    length = max(1, len(stem))
    candidate = "".join(rng.choice(alphabet) for _ in range(length))
    attempts = 0
    while candidate in used or candidate == stem:
        candidate = "".join(rng.choice(alphabet) for _ in range(length))
        attempts += 1
        if attempts > 10000:
            raise ValueError("Unable to generate a distinct temporary filename.")
    used.add(candidate)
    return directory + "\\" + candidate + suffix


def _locations(report):
    locations = []
    for process in report_processes(report):
        for key in ("process_id", "pid", "thread_id", "tid"):
            if key in process:
                locations.append((process, key, "", key))
        for call in process_calls(process):
            api = str(call.get("api", call.get("api_name", call.get("name", ""))))
            for key in ("process_id", "pid", "thread_id", "tid"):
                if key in call:
                    locations.append((call, key, api, key))
            arguments = call.get("arguments", call.get("args", call.get("parameters", [])))
            if isinstance(arguments, dict):
                for key in list(arguments):
                    locations.append((arguments, key, api, key))
            elif isinstance(arguments, list):
                for argument in arguments:
                    if not isinstance(argument, dict):
                        continue
                    if "name" in argument and "value" in argument:
                        locations.append((argument, "value", api, argument["name"]))
                    elif len(argument) == 1:
                        key = next(iter(argument))
                        locations.append((argument, key, api, key))
            for key in RETURN_KEYS:
                if key in call:
                    locations.append((call, key, api, key))
    return locations


def augment_report(report, rng=None, fields=None):
    rng = rng or random.Random()
    enabled = set(fields or ("handle", "address", "pid", "tid", "temporary_filename"))
    augmented = copy.deepcopy(report)
    mappings = {kind: {} for kind in ("handle", "address", "pid", "tid", "temporary_filename")}
    used = {kind: set() for kind in mappings}
    locations = _locations(augmented)
    for container, key, api, field in locations:
        value = container[key]
        if isinstance(value, (dict, list)):
            continue
        kind = _field_kind(api, field, value)
        if kind is None and _temporary_filename(value, field):
            kind = "temporary_filename"
        if kind not in enabled:
            continue
        original = str(value)
        if original not in mappings[kind]:
            if kind == "handle":
                replacement = _new_hex(original, rng, used[kind])
            elif kind == "address":
                replacement = _new_hex(original, rng, used[kind], address=True)
            elif kind in {"pid", "tid"}:
                replacement = _new_decimal(original, rng, used[kind])
            else:
                replacement = _new_filename(original, rng, used[kind])
            mappings[kind][original] = replacement
    reverse_lookup = {}
    for kind, mapping in mappings.items():
        for original, replacement in mapping.items():
            reverse_lookup.setdefault(original, []).append((kind, replacement))
    for container, key, api, field in locations:
        value = container[key]
        if isinstance(value, (dict, list)):
            continue
        original = str(value)
        candidates = reverse_lookup.get(original, [])
        kind = _field_kind(api, field, value)
        if kind is None and _temporary_filename(value, field):
            kind = "temporary_filename"
        matched = [replacement for candidate_kind, replacement in candidates if candidate_kind == kind]
        if len(matched) == 1:
            container[key] = matched[0]
        elif len(candidates) == 1:
            container[key] = candidates[0][1]
    return augmented


def generate_candidate_pool(report, factor=4, seed=7, fields=None):
    if factor < 0:
        raise ValueError("factor must be non-negative.")
    pool = [copy.deepcopy(report)]
    for index in range(factor):
        pool.append(augment_report(report, random.Random(seed + index), fields=fields))
    return pool


def _legacy_transform(api, field, seed=None):
    report = json.loads(api)
    return json.dumps(augment_report(report, random.Random(seed), fields={field}), ensure_ascii=False)


def random_handle_value(API, keys=None):
    return _legacy_transform(API, "handle")


def random_ptid_value(API):
    report = json.loads(API)
    return json.dumps(
        augment_report(report, random.Random(), fields={"pid", "tid"}),
        ensure_ascii=False,
    )


def random_address_value(API, key="Address"):
    return _legacy_transform(API, "address")


def random_file_path(API):
    return _legacy_transform(API, "temporary_filename")


def enhance_based_data(path, filename=None, new_file_path=None, enhance_factor=4, seed=7):
    if filename is None:
        raise ValueError("filename is required.")
    source = Path(path) / filename
    output_dir = Path(new_file_path or path)
    output_dir.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8") as stream:
        report = json.load(stream)
    pool = generate_candidate_pool(report, factor=enhance_factor, seed=seed)
    written = []
    for index, variant in enumerate(pool[1:]):
        output = output_dir / f"{source.stem}_aug{index}{source.suffix}"
        with output.open("w", encoding="utf-8") as stream:
            json.dump(variant, stream, ensure_ascii=False)
        written.append(str(output))
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--factor", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    source = Path(args.input)
    written = enhance_based_data(
        str(source.parent),
        source.name,
        args.output,
        enhance_factor=args.factor,
        seed=args.seed,
    )
    print(json.dumps({"written": written}, ensure_ascii=False))


if __name__ == "__main__":
    main()
