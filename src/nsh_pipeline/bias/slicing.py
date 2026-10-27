"""Bias detection by data slicing (M5): swipe outcomes compared across client and member subgroups.

Reads the processed swipes, clients and members tables, attaches client and member attributes
to every swipe, and computes each slice's denial and substantiation rates (Fairlearn
MetricFrame). Slices that are worse off than overall by more than the thresholds in
params.yaml -> bias are flagged; flags that are only side effects of a stronger flag are marked
as proxies; the strongest flag is then mitigated by reweighing. Airflow and DVC call:
    python -m nsh_pipeline.bias.slicing --processed-dir data/processed --params params.yaml
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from nsh_pipeline.bias import report
from nsh_pipeline.bias.metrics import RATE_METRICS, group_rates, metric_outcome, one_sided_p
from nsh_pipeline.bias.mitigation import reweigh
from nsh_pipeline.config import load_params
from nsh_pipeline.preprocess.privacy import assert_no_phi

logger = logging.getLogger(__name__)

UNKNOWN = "Unknown"
THRESHOLD_KEYS = ("min_n", "abs_diff_pts", "rel_ratio", "alpha")
SLICE_COLUMNS = ["metric", "dimension", "group", "n", "rate", "overall", "rest_rate", "p_value"]


def load_tables(processed_dir) -> dict[str, pd.DataFrame]:
    """Read the processed tables bias analysis needs; fail clearly if one is missing."""
    processed_dir = Path(processed_dir)
    tables = {}
    for name in ("swipes", "clients", "members"):
        path = processed_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing processed table {name!r}: expected {path} (run preprocessing first)")
        tables[name] = pd.read_parquet(path)
    return tables


def join_swipes(swipes: pd.DataFrame, clients: pd.DataFrame, members: pd.DataFrame,
                client_dims: list[str], member_dims: list[str]) -> tuple[pd.DataFrame, list[str], dict]:
    """Attach client and member attributes to every swipe; return (joined, slice columns, join counts).

    Excluded orgs (NS Demo, NSAI) are dropped. Swipes whose client or member is missing cannot be
    sliced, so they are dropped and counted. Attributes whose name clashes with a swipe column
    (clients.status vs swipes.status) get a client_/member_ prefix. Missing attribute values become
    "Unknown" so they form their own slice instead of disappearing.
    """
    client_names = {c: f"client_{c}" if c in swipes.columns else c for c in client_dims}
    member_names = {c: f"member_{c}" if c in swipes.columns else c for c in member_dims}
    kept_clients = clients.loc[~clients["is_excluded_org"], ["client_id", *client_dims]].rename(columns=client_names)
    kept_members = members[["member_id", *member_dims]].rename(columns=member_names)

    has_client = swipes["client_id"].isin(kept_clients["client_id"])
    has_member = swipes["member_id"].isin(kept_members["member_id"])
    joined = (swipes[has_client & has_member]
              .merge(kept_clients, on="client_id", how="left", validate="many_to_one")
              .merge(kept_members, on="member_id", how="left", validate="many_to_one"))
    dims = [*client_names.values(), *member_names.values()]
    joined[dims] = joined[dims].astype("string").fillna(UNKNOWN).astype(str)

    counts = {
        "swipes": len(swipes),
        "analysed": len(joined),
        "dropped_no_client": int((~has_client).sum()),
        "orphan_swipe_ids": sorted(swipes.loc[has_client & ~has_member, "swipe_id"]),
        "unparseable_paid": int(joined["paid_usd"].isna().sum()),
    }
    if counts["orphan_swipe_ids"]:
        logger.warning("Dropped %d orphan swipes (member not in members table)", len(counts["orphan_swipe_ids"]))
    if counts["dropped_no_client"]:
        logger.warning("Dropped %d swipes whose client is unknown or excluded", counts["dropped_no_client"])
    return joined, dims, counts


def slice_table(swipes: pd.DataFrame, dims: list[str], metrics=RATE_METRICS) -> pd.DataFrame:
    """One row per (metric, dimension, group): row count, rate, overall rate, rest-of-data rate, p-value."""
    rows = []
    for metric in metrics:
        outcome = metric_outcome(swipes, metric)
        if outcome.empty:
            logger.warning("No rows for %s; skipped", metric)
            continue
        total_n, total_x = len(outcome), int(outcome.sum())
        for dim in dims:
            by_group, overall = group_rates(outcome, swipes.loc[outcome.index, dim])
            for group, row in by_group.iterrows():
                n, x = int(row["n"]), int(round(row["rate"] * row["n"]))
                rest_n, rest_x = total_n - n, total_x - x
                rows.append({
                    "metric": metric, "dimension": dim, "group": str(group), "n": n, "rate": row["rate"],
                    "overall": overall, "rest_rate": rest_x / rest_n if rest_n else np.nan,
                    "p_value": one_sided_p(x, n, rest_x, rest_n),
                })
    return pd.DataFrame(rows, columns=SLICE_COLUMNS)


def flag_slices(table: pd.DataFrame, min_n: int, abs_diff_pts: float, rel_ratio: float, alpha: float,
                n_tests: int | None = None) -> pd.DataFrame:
    """Add diff_pts, ratio and a verdict per slice: "flagged", "too small to judge" or "ok".

    Only the worse-off direction is flagged: a higher denial or substantiation rate is a burden
    on that group, a lower one is not. A flag also needs the gap to be significant. Dozens of
    slices are tested at once, so alpha is Bonferroni-corrected: divided by `n_tests`, by default
    the number of slices with at least min_n rows. Without it, clean data from 30 seeds produced
    15 chance flags; with it (and the exact test), 1.
    """
    out = table.copy()
    out["diff_pts"] = (out["rate"] - out["overall"]) * 100
    out["ratio"] = out["rate"] / out["overall"].replace(0, np.nan)
    out["excess"] = out["n"] * (out["rate"] - out["rest_rate"].fillna(out["overall"]))  # extra outcome-1 rows
    tested = out["n"].ge(min_n)
    out["worse"] = out["diff_pts"].gt(abs_diff_pts) | out["ratio"].gt(rel_ratio)
    significant = out["p_value"].lt(alpha / max(n_tests or int(tested.sum()), 1))
    out["verdict"] = np.select([~tested, out["worse"] & significant], ["too small to judge", "flagged"], default="ok")
    return out


def explain_proxies(swipes: pd.DataFrame, slices: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """Mark flags that are side effects of the strongest flag of the same metric.

    The root is the flag with the most excess outcomes (rows x gap to the rest), not the largest
    gap: a small slice inside the real cause can have a bigger gap but explains far fewer rows.
    Every flag on another dimension is re-tested on the swipes outside the root slice; if it is
    no longer flagged there (or nothing is left), it is a proxy, e.g. a client status whose
    clients all sit in the flagged state.
    """
    out = slices.assign(explained_by=pd.Series(pd.NA, index=slices.index, dtype="object"))
    n_tests = int(slices["n"].ge(thresholds["min_n"]).sum())  # same Bonferroni correction as the first pass
    for metric, flags in out[out["verdict"].eq("flagged")].groupby("metric"):
        root = flags.loc[flags["excess"].idxmax()]
        others = flags[flags["dimension"].ne(root["dimension"])]
        if others.empty:
            continue
        outside = swipes[swipes[root["dimension"]].ne(root["group"])]
        retest = flag_slices(slice_table(outside, list(others["dimension"].unique()), (metric,)), **thresholds,
                             n_tests=n_tests)
        still_flagged = set(retest.loc[retest["verdict"].eq("flagged"), ["dimension", "group"]].itertuples(index=False))
        for idx, row in others.iterrows():
            if (row["dimension"], row["group"]) not in still_flagged:
                out.loc[idx, "explained_by"] = f"{root['dimension']} = {root['group']}"
    return out


def paid_table(swipes: pd.DataFrame, dims: list[str]) -> pd.DataFrame:
    """Average paid per swipe for every slice. Descriptive only: amounts differ by plan, so not flagged."""
    frames = [swipes.groupby(dim)["paid_usd"].mean().rename("avg_paid_usd").rename_axis("group").reset_index()
              .assign(dimension=dim) for dim in dims]
    return pd.concat(frames, ignore_index=True)[["dimension", "group", "avg_paid_usd"]]


def top_flag(slices: pd.DataFrame):
    """The strongest flag that is not a proxy (most excess outcomes), or None if nothing is flagged."""
    roots = slices[slices["verdict"].eq("flagged") & slices["explained_by"].isna()]
    return None if roots.empty else roots.loc[roots["excess"].idxmax()]


def analyse(tables: dict[str, pd.DataFrame], settings: dict) -> dict:
    """Join, slice, flag and explain; mitigate the strongest flag by reweighing."""
    thresholds = {key: settings[key] for key in THRESHOLD_KEYS}
    swipes, dims, join = join_swipes(tables["swipes"], tables["clients"], tables["members"],
                                     settings["slices"]["clients"], settings["slices"]["members"])
    slices = explain_proxies(swipes, flag_slices(slice_table(swipes, dims), **thresholds), thresholds)
    root = top_flag(slices)
    mitigation = None
    weights = pd.Series(1.0, index=swipes.index, name="weight")
    if root is not None:
        outcome = metric_outcome(swipes, root["metric"])
        mitigation = reweigh(outcome, swipes.loc[outcome.index, root["dimension"]])
        mitigation.update(metric=root["metric"], dimension=root["dimension"], group=root["group"])
        weights.loc[outcome.index] = mitigation["weights"]
    return {"swipes": swipes, "dimensions": dims, "join": join, "slices": slices,
            "paid": paid_table(swipes, dims), "mitigation": mitigation, "weights": weights}


def summarize(result: dict) -> dict:
    """JSON-safe summary for the CLI, Airflow XCom and logs."""
    slices = result["slices"]
    flagged = slices[slices["verdict"].eq("flagged")]
    roots, proxies = flagged[flagged["explained_by"].isna()], flagged[flagged["explained_by"].notna()]
    mitigation = result["mitigation"]
    return {
        "swipes_analysed": result["join"]["analysed"],
        "orphan_swipes_dropped": len(result["join"]["orphan_swipe_ids"]),
        "flagged": [f"{r.metric}: {r.dimension} = {r.group}" for r in roots.itertuples()],
        "proxies": [f"{r.metric}: {r.dimension} = {r.group} (explained by {r.explained_by})"
                    for r in proxies.itertuples()],
        "too_small_to_judge": int(slices["verdict"].eq("too small to judge").sum()),
        "mitigation": None if mitigation is None else {
            "method": "reweighing", "metric": mitigation["metric"], "dimension": mitigation["dimension"],
            "max_gap_before_pts": round(mitigation["max_gap_before_pts"], 2),
            "max_gap_after_pts": round(mitigation["max_gap_after_pts"], 2),
            "effective_n": round(mitigation["effective_n"], 1), "n": mitigation["n"],
        },
    }


def run_bias(processed_dir, params: dict, report_dir="reports", out_dir="data/bias") -> dict:
    """Analyse `processed_dir`; write the report, slice table and charts to `report_dir` and
    per-swipe reweighing weights to `out_dir`. Flags are logged as warnings and never fail the run."""
    result = analyse(load_tables(processed_dir), params["bias"])
    report_path = report.write_report(result, params, report_dir)

    weights = pd.DataFrame({"swipe_id": result["swipes"]["swipe_id"], "weight": result["weights"]})
    assert_no_phi(weights, "swipe_weights")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights.to_parquet(out_dir / "swipe_weights.parquet", index=False)

    summary = {**summarize(result), "report": str(report_path)}
    for flag in summary["flagged"]:
        logger.warning("Bias flag: %s", flag)
    logger.info("Bias report written to %s", report_path)
    return summary


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description="Detect and mitigate bias in processed NSH data by slicing.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--report-dir", default="reports", help="Where bias_report.md and bias/ charts go")
    parser.add_argument("--out-dir", default="data/bias", help="Where swipe_weights.parquet goes")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    summary = run_bias(args.processed_dir, load_params(args.params), args.report_dir, args.out_dir)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":  # pragma: no cover
    main()
