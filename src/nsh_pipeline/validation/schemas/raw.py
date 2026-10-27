"""Light schemas for the raw JSONL collections, plus file checks against manifest.json.

Raw data keeps the source quirks on purpose, so these schemas only check what preprocessing
relies on (plan section 8, stage 3): the fields it reads exist, keys are unique text, money and
dates are stored in the form the data contract says (A4), and the misspelled field names are
still misspelled. If someone "fixes" `needsSubstantion` upstream, preprocessing would silently
read nothing, so the correctly spelled name is a failure too.
"""
import hashlib
from pathlib import Path

import pandas as pd
from pandera.pandas import Check, DataFrameSchema

from nsh_pipeline.validation.schemas.checks import any_value, number, text

MDY = r"\d{1,2}/\d{1,2}/\d{4}"


def _no_column(name: str) -> Check:
    """Fail if the correctly spelled `name` appears (the contract field is misspelled on purpose)."""
    return Check(lambda df: name not in df.columns, name=f"no_correctly_spelled_{name}")


def _schema(name: str, columns: dict, checks=None) -> DataFrameSchema:
    return DataFrameSchema(columns, name=name, strict=False, checks=checks or [])


RAW_SCHEMAS: dict[str, DataFrameSchema] = {
    "applications": _schema("applications", {
        "_id": text(nullable=False, unique=True),
        "employer": any_value(nullable=False),
        "status": any_value(),
        "dnnPortalAlias": any_value(),
    }),
    "people": _schema("people", {
        "_id": text(nullable=False, unique=True),
        "Application": text(),
        "Names": any_value(),
        "PersonalData": any_value(),
        "Employment": any_value(),
        "Benefits": any_value(),
    }),
    "wex_claims": _schema("wex_claims", {
        "_id": text(nullable=False, unique=True),  # a String, not an ObjectId (contract A0 rule 8)
        "employeeId": text(),
        "clientId": text(),
        "date_of_service": text(),
        "amount": text(),  # numeric strings (A4); "N/A" passes here and is caught by anomalies
        "paid_amt": text(),
        "denied_amt": text(),
        "claim_status": text(),
        "claim_source_desc": text(),
        "needsSubstantion": any_value(),  # (sic)
        "ticketNumber": text(),
    }, checks=[_no_column("needsSubstantiation")]),
    "claims": _schema("claims", {
        "claimNumber": text(nullable=False, unique=True),
        "employeeId": text(),
        "clientId": text(),
        "requestedAmount": number(nullable=True),
        "amountApproved": number(nullable=True),
        "amountDenied": number(nullable=True),
        "status": text(),
        "createdAt": text(),
    }),
    "tickets": _schema("tickets", {
        "ticketNumber": text(nullable=False, unique=True),
        "employeeId": text(),
        "clientId": text(),
        "status": text(),
        "createdAt": text(),
    }),
    "bills": _schema("bills", {
        "_id": text(nullable=False, unique=True),
        "appId": text(),
        "billingDate": text(pattern=r"\d{4}-\d{2}"),
        "recurring": any_value(),
    }),
    "reserves": _schema("reserves", {
        "_id": text(nullable=False, unique=True),
        "client": text(),
        "year": number(whole=True),
        "month": number(ge=1, le=12, whole=True),
        "balance": number(nullable=True),
        "reservesBilled": number(nullable=True),
        "claimsPaid": number(nullable=True),
        "returnOfReserve": number(nullable=True),
    }),
    "nse_transaction_daily_logs": _schema("nse_transaction_daily_logs", {
        "_id": text(nullable=False, unique=True),
        "employerCode": text(),
        "employeeId": text(),
        "amount": number(whole=True),  # integer cents (A4)
        "effectiveDate": text(),
    }),
    "debit_card_status": _schema("debit_card_status", {
        "_id": text(nullable=False, unique=True),
        "employeeId": text(),
        "nswid": text(),
        "pariticpant_status": any_value(),  # (sic)
        "valid_start_dt": text(pattern=MDY),  # M/D/YYYY strings (A4)
        "valid_end_dt": text(pattern=MDY),
        "max_oop": number(nullable=True),
    }, checks=[_no_column("participant_status")]),
}

RAW_COLLECTIONS = tuple(RAW_SCHEMAS)


def sha256_of(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_files(raw_dir, manifest: dict | None) -> list[dict]:
    """Compare each expected raw file with manifest.json (exists, row count, SHA-256).

    Returns one error record per problem; an empty list means every file matches. Without a
    manifest only existence is checked (the caller reports the missing manifest).
    """
    raw_dir = Path(raw_dir)
    files = None if manifest is None else manifest.get("files", {})
    errors = []
    for name in RAW_COLLECTIONS:
        path = raw_dir / f"{name}.jsonl"
        if not path.exists():
            errors.append({"file": path.name, "check": "exists", "message": "file is missing"})
            continue
        if files is None:
            continue
        expected = files.get(path.name)
        if expected is None:
            errors.append({"file": path.name, "check": "manifest", "message": "not listed in manifest.json"})
            continue
        with path.open(encoding="utf-8") as f:
            rows = sum(1 for line in f if line.strip())
        if "rows" in expected and rows != expected["rows"]:
            errors.append({"file": path.name, "check": "row_count",
                           "message": f"{rows} rows, manifest says {expected['rows']}"})
        if "sha256" in expected and sha256_of(path) != expected["sha256"]:
            errors.append({"file": path.name, "check": "sha256", "message": "checksum differs from manifest"})
    return errors


def raw_frame(records: list[dict]) -> pd.DataFrame:
    """Top-level fields of raw documents as a DataFrame (nested arrays stay Python objects)."""
    return pd.DataFrame(records)
