"""Select current values from the array-of-versions fields on `people` documents.

NSH stores member history as arrays (Names[], Employment[], Benefits[]). The current
value is the last entry with Deleted != True and Active != False (data contract A0 rule 4).
"""


def current_entry(versions):
    """Return the latest non-deleted, active entry of an array-of-versions field, or None."""
    if not isinstance(versions, list):
        return None
    for entry in reversed(versions):
        if not isinstance(entry, dict):
            continue
        if entry.get("Deleted", False) or entry.get("Active", True) is False:
            continue
        return entry
    return None


def current_value(versions, key, default=None):
    """Return `key` from the current entry of `versions`, or `default` if there is none."""
    entry = current_entry(versions)
    if entry is None:
        return default
    return entry.get(key, default)