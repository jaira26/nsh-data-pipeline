"""Exclude NSH's non-client orgs (NS Demo, NSAI staff) from reporting (data contract A0 rule 2)."""
import pandas as pd

from nsh_pipeline.constants import EXCLUDED_ORG_IDS

EXCLUDED_ALIASES = frozenset({"demo", "nsais"})


def flag_excluded_orgs(
    clients: pd.DataFrame, id_col: str = "_id", alias_col: str = "dnnPortalAlias"
) -> pd.DataFrame:
    """Return a copy of `clients` with a boolean `is_excluded_org` column, matched by id or alias."""
    out = clients.copy()
    excluded = out[id_col].isin(list(EXCLUDED_ORG_IDS))
    if alias_col in out.columns:
        excluded |= out[alias_col].astype(str).str.lower().isin(list(EXCLUDED_ALIASES))
    out["is_excluded_org"] = excluded
    return out


def drop_excluded(df: pd.DataFrame, client_col: str, excluded_ids=EXCLUDED_ORG_IDS) -> pd.DataFrame:
    """Return the rows of `df` whose `client_col` is not an excluded org.

    Rows with no client id are kept; orphan checks belong to validation, not here.
    """
    keep = ~df[client_col].isin(list(excluded_ids))
    return df[keep].reset_index(drop=True)