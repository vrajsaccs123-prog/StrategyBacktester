"""Turn rule JSON into entry and exit flags, and into a position series."""

from __future__ import annotations

import copy
import re

import pandas as pd

from engine.indicators import INDICATOR_DEFS, output_names
from engine.models import ParamSpec

PRICE_COLUMNS = ("open", "high", "low", "close", "volume")
COMPARE_OPS = {">", "<", ">=", "<=", "=="}
CROSS_OPS = {"crosses_above", "crosses_below"}
RULE_GROUPS = ("entry", "exit", "entry_short", "exit_short")
_TOKEN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)(?:\[(\d+)\])?")


def default_spec() -> dict:
    return {
        "name": "SMA cross",
        "side": "long_only",
        "indicators": [
            {
                "id": "fast",
                "type": "sma",
                "params": {"length": {"value": 10, "min": 5, "max": 40, "step": 1, "kind": "int"}},
            },
            {
                "id": "slow",
                "type": "sma",
                "params": {"length": {"value": 30, "min": 20, "max": 120, "step": 5, "kind": "int"}},
            },
        ],
        "entry": {"all": [{"left": "fast", "op": "crosses_above", "right": "slow"}]},
        "exit": {"all": [{"left": "fast", "op": "crosses_below", "right": "slow"}]},
        "entry_short": {"all": []},
        "exit_short": {"all": []},
    }


def param_number(raw) -> float:
    if isinstance(raw, dict):
        return float(raw["value"])
    return float(raw)


def _is_search_dict(raw) -> bool:
    return isinstance(raw, dict) and "min" in raw and "max" in raw


def _infer_kind(raw: dict) -> str:
    if raw.get("kind") in {"int", "float"}:
        return raw["kind"]
    numbers = [raw.get("value", raw["min"]), raw["min"], raw["max"], raw.get("step", 1)]
    if all(float(number).is_integer() for number in numbers if number is not None):
        return "int"
    return "float"


def _spec_from_raw(raw: dict) -> ParamSpec | None:
    if not _is_search_dict(raw):
        return None
    kind = _infer_kind(raw)
    default = raw.get("value", raw["min"])
    step = raw.get("step", 1 if kind == "int" else None)
    return ParamSpec(float(default), float(raw["min"]), float(raw["max"]), None if step is None else float(step), kind)


def available_columns(spec: dict) -> list[str]:
    columns = list(PRICE_COLUMNS)
    for indicator in spec.get("indicators", []):
        columns.extend(output_names(indicator["id"], indicator["type"]))
    return columns


def validate_spec(spec: dict) -> list[str]:
    errors = []
    name = str(spec.get("name", "")).strip()
    if not name:
        errors.append("Strategy needs a name")
    if spec.get("side") not in {"long_only", "long_short"}:
        errors.append("Side must be long only or long/short")

    seen = set()
    columns = set(PRICE_COLUMNS)
    for indicator in spec.get("indicators", []):
        indicator_id = str(indicator.get("id", "")).strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", indicator_id):
            errors.append(f"Indicator id '{indicator_id}' must be a letter followed by letters or numbers")
            continue
        if indicator_id in seen:
            errors.append(f"Duplicate indicator id '{indicator_id}'")
        seen.add(indicator_id)
        if indicator.get("type") not in INDICATOR_DEFS:
            errors.append(f"Unknown indicator type '{indicator.get('type')}'")
            continue
        columns.update(output_names(indicator_id, indicator["type"]))

    if not any(spec.get("entry", {}).get(key) for key in ("all", "any")):
        errors.append("Add at least one entry condition")

    def check_node(node, label: str) -> None:
        if not isinstance(node, dict) or not node:
            return
        if "all" in node or "any" in node:
            key = "all" if "all" in node else "any"
            for index, child in enumerate(node[key]):
                check_node(child, f"{label} {index + 1}")
            return
        left = node.get("left")
        op = node.get("op")
        right = node.get("right")
        if left not in columns:
            errors.append(f"{label}: unknown left side '{left}'")
        if op not in COMPARE_OPS | CROSS_OPS:
            errors.append(f"{label}: unknown operator '{op}'")
        if isinstance(right, str):
            if right not in columns:
                errors.append(f"{label}: unknown right side '{right}'")
        elif isinstance(right, dict):
            if "value" not in right:
                errors.append(f"{label}: threshold needs a value")
        elif not isinstance(right, (int, float)):
            errors.append(f"{label}: right side must be a column or a number")

    for group in RULE_GROUPS:
        check_node(spec.get(group) or {"all": []}, group)
    return errors


def _series_for(frame: pd.DataFrame, ref) -> pd.Series:
    if isinstance(ref, str):
        if ref not in frame.columns:
            raise KeyError(f"Unknown column '{ref}'")
        return frame[ref].astype(float)
    number = param_number(ref)
    return pd.Series(number, index=frame.index, dtype=float)


def eval_condition(frame: pd.DataFrame, condition: dict) -> pd.Series:
    left = _series_for(frame, condition["left"])
    right = _series_for(frame, condition["right"])
    op = condition["op"]
    if op in COMPARE_OPS:
        flags = {"<": left < right, ">": left > right, "<=": left <= right, ">=": left >= right, "==": left == right}[op]
    elif op == "crosses_above":
        flags = (left > right) & (left.shift(1) <= right.shift(1))
    elif op == "crosses_below":
        flags = (left < right) & (left.shift(1) >= right.shift(1))
    else:
        raise ValueError(f"Unknown operator '{op}'")
    return flags.fillna(False)


def eval_rule(frame: pd.DataFrame, node: dict | None) -> pd.Series:
    index = frame.index
    if not node:
        return pd.Series(False, index=index)
    if "all" in node:
        children = node["all"]
        if not children:
            return pd.Series(False, index=index)
        flags = eval_rule(frame, children[0])
        for child in children[1:]:
            flags = flags & eval_rule(frame, child)
        return flags.fillna(False)
    if "any" in node:
        children = node["any"]
        if not children:
            return pd.Series(False, index=index)
        flags = eval_rule(frame, children[0])
        for child in children[1:]:
            flags = flags | eval_rule(frame, child)
        return flags.fillna(False)
    return eval_condition(frame, node)


def positions_from_flags(
    entry: pd.Series,
    exit_: pd.Series,
    start: int = 0,
    entry_short: pd.Series | None = None,
    exit_short: pd.Series | None = None,
    side: str = "long_only",
) -> pd.Series:
    """Walk flags from `start`, beginning flat. Earlier rows are indicator history only."""

    entry_flags = entry.fillna(False).to_numpy()
    exit_flags = exit_.fillna(False).to_numpy()
    short_entry = entry_short.fillna(False).to_numpy() if entry_short is not None else None
    short_exit = exit_short.fillna(False).to_numpy() if exit_short is not None else None
    position = 0
    values = []
    for i in range(start, len(entry_flags)):
        if side != "long_short":
            if position == 0 and entry_flags[i]:
                position = 1
            elif position == 1 and exit_flags[i]:
                position = 0
        else:
            want_long = bool(entry_flags[i])
            want_short = bool(short_entry[i]) if short_entry is not None else False
            if position > 0 and exit_flags[i]:
                position = 0
            elif position < 0 and short_exit is not None and short_exit[i]:
                position = 0
            if want_long and not want_short:
                position = 1
            elif want_short and not want_long:
                position = -1
        values.append(position)
    return pd.Series(values, index=entry.index[start:], dtype=int)


def _tokenize(path: str) -> list:
    tokens = []
    for part in path.split("."):
        match = _TOKEN.fullmatch(part)
        if not match:
            raise ValueError(f"Bad parameter path '{path}'")
        name, index = match.groups()
        tokens.append((name, int(index)) if index is not None else name)
    return tokens


def extract_param_space(spec: dict) -> dict[str, ParamSpec]:
    space: dict[str, ParamSpec] = {}
    for indicator in spec.get("indicators", []):
        definitions = INDICATOR_DEFS[indicator["type"]].params
        for param_name, raw in indicator.get("params", {}).items():
            found = _spec_from_raw(raw) if isinstance(raw, dict) else None
            if found is None:
                continue
            kind = definitions[param_name].kind if param_name in definitions else found.kind
            space[f"ind.{indicator['id']}.{param_name}"] = ParamSpec(
                found.default, found.min, found.max, found.step, kind
            )

    def walk(node, path: str) -> None:
        if not isinstance(node, dict):
            return
        if "all" in node or "any" in node:
            key = "all" if "all" in node else "any"
            for index, child in enumerate(node[key]):
                walk(child, f"{path}.{key}[{index}]")
            return
        right = node.get("right")
        found = _spec_from_raw(right) if isinstance(right, dict) else None
        if found is not None:
            space[f"{path}.right"] = found

    for group in RULE_GROUPS:
        walk(spec.get(group) or {}, group)
    return space


def apply_params(spec: dict, params: dict) -> dict:
    updated = copy.deepcopy(spec)
    indicators = {item["id"]: item for item in updated.get("indicators", [])}
    for path, value in params.items():
        if path.startswith("ind."):
            _, indicator_id, param_name = path.split(".", 2)
            target = indicators[indicator_id]["params"]
            current = target.get(param_name)
            if isinstance(current, dict) and "value" in current:
                current["value"] = value
            else:
                target[param_name] = value
            continue
        tokens = _tokenize(path)
        node = updated
        for token in tokens[:-1]:
            if isinstance(token, tuple):
                node = node[token[0]][token[1]]
            else:
                node = node[token]
        last = tokens[-1]
        current = node.get(last)
        if isinstance(current, dict) and "value" in current:
            current["value"] = value
        else:
            node[last] = value
    return updated
