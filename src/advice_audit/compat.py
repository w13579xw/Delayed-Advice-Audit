"""Read-only compatibility for frozen pre-rename identifiers."""
LEGACY_METHODS = {"daci": "advice_audit", "daci_calibrated": "advice_audit_calibrated"}

def canonical_method(value):
    return LEGACY_METHODS.get(value, value)

def canonical_schema(value):
    if isinstance(value, str) and value.startswith("daci."):
        return "advice_audit." + value[len("daci."):]
    return value
