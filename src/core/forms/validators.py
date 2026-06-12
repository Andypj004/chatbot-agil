from typing import Any, Dict, Tuple
import re


def _as_float(value: Any) -> Tuple[bool, float]:
    try:
        return True, float(value)
    except Exception:
        return False, 0.0


def _as_int(value: Any) -> Tuple[bool, int]:
    try:
        return True, int(value)
    except Exception:
        return False, 0


def validate_field(field: Dict[str, Any], value: Any) -> Tuple[bool, str]:
    if field.get("required") and (
        value is None or (isinstance(value, str) and value.strip() == "")
    ):
        return False, "value required"

    typ = field.get("type")
    if typ in ("int", "integer"):
        ok, _ = _as_int(value)
        if not ok:
            return False, "must be integer"

    if typ in ("float", "number", "decimal"):
        ok, _ = _as_float(value)
        if not ok:
            return False, "must be number"

    if typ == "bool":
        if not isinstance(value, bool):
            if isinstance(value, str) and value.lower() in ("true", "false"):
                return True, ""
            return False, "must be boolean"

    if typ in ("text", "string"):
        if value is not None and not isinstance(value, str):
            return False, "must be string"

    min_value = field.get("min")
    max_value = field.get("max")
    if min_value is not None or max_value is not None:
        ok, num = _as_float(value)
        if not ok:
            return False, "must be number"
        if min_value is not None and num < float(min_value):
            return False, f"must be >= {min_value}"
        if max_value is not None and num > float(max_value):
            return False, f"must be <= {max_value}"

    min_len = field.get("min_length")
    max_len = field.get("max_length")
    if (min_len is not None or max_len is not None) and isinstance(value, str):
        if min_len is not None and len(value) < int(min_len):
            return False, f"length must be >= {min_len}"
        if max_len is not None and len(value) > int(max_len):
            return False, f"length must be <= {max_len}"

    pattern = field.get("pattern")
    if pattern and isinstance(value, str):
        if not re.search(pattern, value):
            return False, "pattern mismatch"

    choices = field.get("choices")
    if choices is not None:
        if value not in choices:
            return False, f"must be one of: {choices}"

    return True, ""


def run_validator(
    validator_name: str, value: Any, field: Dict[str, Any]
) -> Tuple[bool, str]:
    if validator_name == "non_empty_str":
        if isinstance(value, str) and value.strip():
            return True, ""
        return False, "must be a non-empty string"

    return validate_field(field, value)


def validate_cross_fields(
    cross_validators: list[Dict[str, Any]], answers: Dict[str, Any]
) -> Tuple[bool, str]:
    for rule in cross_validators:
        name = rule.get("name")
        if name == "lte_field":
            left = answers.get(rule.get("field"))
            right = answers.get(rule.get("other"))
            if left is None or right is None:
                continue
            ok_left, left_val = _as_float(left)
            ok_right, right_val = _as_float(right)
            if not ok_left or not ok_right:
                return False, "cross validation expects numeric fields"
            if left_val > right_val:
                return (
                    False,
                    rule.get("message")
                    or f"{rule.get('field')} must be <= {rule.get('other')}",
                )
        elif name == "gte_field":
            left = answers.get(rule.get("field"))
            right = answers.get(rule.get("other"))
            if left is None or right is None:
                continue
            ok_left, left_val = _as_float(left)
            ok_right, right_val = _as_float(right)
            if not ok_left or not ok_right:
                return False, "cross validation expects numeric fields"
            if left_val < right_val:
                return (
                    False,
                    rule.get("message")
                    or f"{rule.get('field')} must be >= {rule.get('other')}",
                )
        elif name == "require_if":
            field = rule.get("field")
            when_field = rule.get("when_field")
            when_value = rule.get("when_value")
            if answers.get(when_field) == when_value:
                if answers.get(field) in (None, ""):
                    return False, rule.get("message") or f"{field} is required"

    return True, ""
