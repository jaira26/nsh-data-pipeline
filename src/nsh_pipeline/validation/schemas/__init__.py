"""pandera schemas for the raw, processed and feature layers, and a helper that turns
pandera errors into a JSON-friendly result."""
import pandas as pd
from pandera.errors import SchemaError, SchemaErrors
from pandera.pandas import DataFrameSchema


def validate_frame(df: pd.DataFrame, schema: DataFrameSchema, table: str, sample: int = 10) -> dict:
    """Validate `df` lazily (every failure collected) and return a result record.

    Result: {"table", "status": "passed" | "failed", "rows", "errors": [{"column", "check",
    "count", "failure_cases"}]}. Never raises on invalid data.
    """
    result = {"table": table, "status": "passed", "rows": int(len(df)), "errors": []}
    try:
        schema.validate(df, lazy=True)
    except SchemaErrors as exc:
        result["status"] = "failed"
        result["errors"] = summarize_failures(exc.failure_cases, sample)
    except SchemaError as exc:  # pragma: no cover - lazy=True normally raises SchemaErrors
        result["status"] = "failed"
        result["errors"] = [{"column": None, "check": "schema", "count": 1, "failure_cases": [str(exc)]}]
    return result


def summarize_failures(failure_cases: pd.DataFrame, sample: int = 10) -> list[dict]:
    """Group pandera's failure-case rows by (column, check) with a count and sample values.

    Table-level failures (missing or extra columns, multi-column uniqueness) get column None.
    """
    cases = failure_cases.copy()
    column_level = cases["schema_context"].astype(str).eq("Column") & cases["column"].notna()
    cases["column"] = cases["column"].astype(object).where(column_level, "").astype(str)
    errors = []
    for (column, check), group in cases.groupby(["column", "check"], sort=True):
        values = [str(v) for v in group["failure_case"].tolist()[:sample]]
        errors.append({
            "column": column or None,
            "check": str(check),
            "count": int(len(group)),
            "failure_cases": values,
        })
    return errors


def failed_tables(results: list[dict]) -> list[str]:
    """Names of the tables whose result status is "failed"."""
    return [r["table"] for r in results if r.get("status") == "failed"]
