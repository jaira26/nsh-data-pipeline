"""Join-key helpers shared by the generator and preprocessing (data contract A3)."""
import re


def make_nswid(last_name, ssn):
    """Build the member join key: lowercased first 7 chars of last name + last 4 SSN digits.

    Returns None when the last name is missing or the SSN has fewer than 4 digits.
    """
    if not isinstance(last_name, str) or not isinstance(ssn, str):
        return None
    name = last_name.strip()
    digits = re.sub(r"\D", "", ssn)
    if not name or len(digits) < 4:
        return None
    return name[:7].lower() + digits[-4:]