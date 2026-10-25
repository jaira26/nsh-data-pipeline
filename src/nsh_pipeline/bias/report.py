"""Write the bias report: reports/bias_report.md, reports/bias/slices.csv and two charts.

Charts use matplotlib's object API (no pyplot), so they render without a display inside Airflow.
"""
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from nsh_pipeline.bias.metrics import METRIC_LABELS, metric_outcome
from nsh_pipeline.preprocess.flatten import GROUP_SIZE_LABELS

DECISIONS_URL = "https://github.com/jaira26/nsh-data-pipeline/blob/main/docs/decisions.md"

# Chart colors: two categorical slots, a reserved "critical" status color for flagged bars
# (always paired with a text label), a muted gray for too-small slices, and text/grid inks.
SERIES = ("#2a78d6", "#eb6834")
CRITICAL = "#d03b3b"
MUTED = "#a9a8a2"
SURFACE = "#fcfcfb"
TEXT, TEXT_2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def pct(rate: float) -> str:
    return f"{rate * 100:.1f}%"


def p_text(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def md_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def _axes(height_rows: int):
    fig = Figure(figsize=(7.5, 0.42 * height_rows + 1.4), dpi=150, facecolor=SURFACE)
    ax = fig.add_subplot()
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_2, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    return fig, ax


def _overall_line(ax, overall: float):
    ax.axvline(overall, color=TEXT_2, linestyle="--", linewidth=1.2, zorder=1)
    ax.text(overall, -0.75, f" overall {pct(overall)}", color=TEXT_2, fontsize=8, va="bottom")


def rate_chart(slices: pd.DataFrame, metric: str, dimension: str, path: Path) -> Path:
    """Horizontal bars of one metric across one dimension; flagged bars in red with a label."""
    data = slices[(slices["metric"] == metric) & (slices["dimension"] == dimension)].sort_values("rate")
    colors = {"flagged": CRITICAL, "too small to judge": MUTED, "ok": SERIES[0]}
    fig, ax = _axes(len(data))
    y = range(len(data))
    ax.barh(y, data["rate"], height=0.7, color=[colors[v] for v in data["verdict"]])
    ax.set_yticks(list(y), data["group"], color=TEXT, fontsize=9)
    for i, row in enumerate(data.itertuples()):
        note = {"flagged": "  ▲ flagged", "too small to judge": "  too small to judge"}.get(row.verdict, "")
        ax.annotate(f"{pct(row.rate)}  n={row.n:,}{note}", (row.rate, i), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, zorder=3, color=TEXT if row.verdict == "flagged" else TEXT_2,
                    fontweight="bold" if row.verdict == "flagged" else None,
                    bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.6})  # legible over the overall line
    _overall_line(ax, data["overall"].iloc[0])
    ax.set_xlim(0, max(data["rate"].max(), data["overall"].iloc[0]) * 1.45)
    ax.set_title(f"{METRIC_LABELS[metric]} by {dimension}", loc="left", color=TEXT, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    return path


def reweighing_chart(mitigation: dict, path: Path) -> Path:
    """Each group's rate before and after reweighing, side by side."""
    data = mitigation["comparison"].sort_values("before")
    fig, ax = _axes(len(data) * 1.15)
    y = pd.Series(range(len(data)), dtype=float)
    ax.barh(y + 0.19, data["before"], height=0.36, color=SERIES[0], label="Before reweighing")
    ax.barh(y - 0.19, data["after"], height=0.36, color=SERIES[1], label="After reweighing")
    ax.set_yticks(list(y), [str(g) for g in data.index], color=TEXT, fontsize=9)
    root = list(data.index).index(mitigation["group"])
    for value, offset in ((data["before"].iloc[root], 0.19), (data["after"].iloc[root], -0.19)):
        ax.annotate(pct(value), (value, root + offset), xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=8, color=TEXT, zorder=3, bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.6})
    _overall_line(ax, mitigation["overall"])
    ax.set_xlim(0, data["before"].max() * 1.3)
    ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=TEXT_2)
    title = f"{METRIC_LABELS[mitigation['metric']]} by {mitigation['dimension']}, before and after reweighing"
    ax.set_title(title, loc="left", color=TEXT, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    return path


def _roots(slices: pd.DataFrame) -> dict:
    """Strongest non-proxy flag per metric."""
    flags = slices[slices["verdict"].eq("flagged") & slices["explained_by"].isna()]
    return {metric: rows.loc[rows["excess"].idxmax()] for metric, rows in flags.groupby("metric")}


def _ordered(slices: pd.DataFrame) -> pd.DataFrame:
    """Group-size buckets in size order (1-9 ... 250+) instead of alphabetical."""
    rank = {label: i for i, label in enumerate(GROUP_SIZE_LABELS)}
    key = slices.apply(lambda r: rank.get(r["group"], len(rank)) if r["dimension"] == "group_size_bucket" else 0,
                       axis=1)
    return slices.assign(_key=key).sort_values(["_key", "group"], kind="stable").drop(columns="_key")


def _slice_name(row) -> str:
    return f"{row['dimension']} = {row['group']}"


def _summary_section(result: dict, params: dict) -> list[str]:
    join, slices, settings = result["join"], result["slices"], params["bias"]
    overall = slices.groupby("metric")["overall"].first()
    paid_debit = len(metric_outcome(result["swipes"], "substantiation_rate"))
    lines = [
        "## Summary", "",
        f"- **{join['analysed']:,} swipes analysed** out of {join['swipes']:,}. "
        f"Dropped: {len(join['orphan_swipe_ids'])} orphan swipes (member not in `members`) and "
        f"{join['dropped_no_client']} with an unknown or excluded client. "
        f"{join['unparseable_paid']} swipes have an unparseable paid amount and are left out of the average paid.",
        f"- Overall denial rate **{pct(overall.get('denial_rate', 0))}**; substantiation rate "
        f"**{pct(overall.get('substantiation_rate', 0))}** of {paid_debit:,} paid debit-card swipes.",
    ]
    flagged = slices[slices["verdict"].eq("flagged")]
    roots = flagged[flagged["explained_by"].isna()]
    if roots.empty:
        lines.append(f"- **No slice is flagged.** Every slice with at least {settings['min_n']} swipes is within "
                     f"{settings['abs_diff_pts']} points and {settings['rel_ratio']}× of the overall rate, "
                     "or its gap is not significant.")
    for row in roots.to_dict("records"):
        lines.append(f"- **Flagged: {_slice_name(row)}** ({METRIC_LABELS[row['metric']].lower()} {pct(row['rate'])} "
                     f"vs {pct(row['overall'])} overall, {row['diff_pts']:+.1f} pts, {row['ratio']:.1f}×, "
                     f"n = {row['n']:,}, p {p_text(row['p_value'])}).")
    for row in flagged[flagged["explained_by"].notna()].to_dict("records"):
        lines.append(f"- Proxy, not separate bias: {_slice_name(row)} "
                     f"({pct(row['rate'])}, n = {row['n']:,}) is explained by {row['explained_by']}.")
    small = slices[slices["verdict"].eq("too small to judge") & slices["metric"].eq("denial_rate")]
    if not small.empty:
        names = ", ".join(f"{_slice_name(r)} (n = {r['n']})" for r in small.to_dict("records"))
        lines.append(f"- Too small to judge (fewer than {settings['min_n']} swipes): {names}.")

    inject = params.get("inject_issues", {})
    skew = inject.get("denial_skew")
    if inject.get("enabled") and skew:
        found = any(r["dimension"] == "state" and r["group"] == skew["state"] and r["metric"] == "denial_rate"
                    for r in roots.to_dict("records"))
        lines.append(f"- **Planted bias check:** injected issues are on, and `denial_skew` flipped {skew['rate']:.0%} "
                     f"of approved swipes in state {skew['state']} to Denied. "
                     + ("Detected." if found else "**Not detected.**"))
    return lines


def _method_section(settings: dict, dims: list[str]) -> list[str]:
    return [
        "## Method", "",
        f"- **Slices:** {', '.join(f'`{d}`' for d in dims)}. Client attributes come from `clients` "
        "(NS Demo and NSAI removed), member attributes from `members`; every swipe is joined to its client and member. "
        "Missing values form their own `Unknown` slice. `client_type` is not sliced because it is mostly null.",
        "- **Metrics per slice**, computed with Fairlearn's `MetricFrame`: denial rate (share of swipes with status "
        "Denied), substantiation rate (share of paid debit-card swipes that need substantiation) and average paid per "
        "swipe (descriptive only, since amounts depend on the plan).",
        "- **Flag rule** (`params.yaml → bias`): a slice is flagged when its rate is higher than the overall rate "
        f"by more than {settings['abs_diff_pts']} percentage points or {settings['rel_ratio']}×, it has at least "
        f"{settings['min_n']} swipes, and a one-sided Fisher exact test against all other swipes gives "
        f"p < {settings['alpha']} divided by the number of slices tested (Bonferroni correction: dozens of slices are "
        "tested at once, so without it chance gaps get flagged). A lower-than-overall rate is not flagged: it is no "
        "burden on the group.",
        "- **Proxy check:** the flag with the most excess outcomes (swipes × gap to the rest) is the root candidate. "
        "Every flag on another dimension is re-tested without the root slice's swipes; a flag that disappears is "
        "reported as a proxy, not as separate bias.",
        "- There is no model in this pipeline yet, so these outcome rates stand in for model performance per slice. "
        "The same slicing can be re-run on model predictions later.",
    ]


def _all_slices_table(result: dict) -> str:
    slices, paid = _ordered(result["slices"]), result["paid"]
    wide = slices.pivot_table(index=["dimension", "group"], columns="metric",
                              values=["n", "rate"], aggfunc="first", sort=False)
    verdicts = slices.set_index(["metric", "dimension", "group"])["verdict"]
    avg_paid = paid.set_index(["dimension", "group"])["avg_paid_usd"]
    marks = {"flagged": " ▲ flagged", "too small to judge": " (too small)", "ok": ""}
    rows = []
    for dim, group in wide.index:
        cells = [f"{dim} = {group}"]
        for metric in ("denial_rate", "substantiation_rate"):
            n = wide.loc[(dim, group)].get(("n", metric))
            if pd.isna(n):
                cells += ["0", "–"]
                continue
            rate = wide.loc[(dim, group), ("rate", metric)]
            cells += [f"{int(n):,}", pct(rate) + marks[verdicts[(metric, dim, group)]]]
        paid_value = avg_paid.get((dim, group))
        cells.append("–" if pd.isna(paid_value) else f"${paid_value:,.2f}")
        rows.append(cells)
    return md_table(["Slice", "Swipes", "Denial rate", "Paid debit swipes", "Substantiation rate", "Avg paid"], rows)


def _results_section(result: dict, charts: dict) -> list[str]:
    slices = result["slices"]
    lines = ["## Results", ""]
    lines += [f"![{title}]({path})" for title, path in charts.items() if not title.startswith("reweighing")]
    flagged = slices[slices["verdict"].eq("flagged")]
    lines += ["", "### Flags", ""]
    if flagged.empty:
        lines.append("None.")
    else:
        rows = [[METRIC_LABELS[r["metric"]], _slice_name(r), f"{r['n']:,}", pct(r["rate"]), pct(r["overall"]),
                 f"{r['diff_pts']:+.1f}", f"{r['ratio']:.1f}×", p_text(r["p_value"]),
                 "flagged" if pd.isna(r["explained_by"]) else f"proxy of {r['explained_by']}"]
                for r in flagged.sort_values("diff_pts", ascending=False).to_dict("records")]
        lines.append(md_table(["Metric", "Slice", "Swipes", "Rate", "Overall", "Gap (pts)", "Ratio", "p-value",
                               "Verdict"], rows))
    lines += ["", "### All slices", "", _all_slices_table(result)]
    return lines


def _interpretation_section(result: dict) -> list[str]:
    slices, swipes = result["slices"], result["swipes"]
    roots = _roots(slices)
    lines = ["## Interpretation", ""]
    if not roots:
        return lines + ["No subgroup is treated measurably worse on denials or substantiation. Slices below the "
                        "minimum size are listed as too small to judge, not as fair."]
    for metric, root in roots.items():
        label = METRIC_LABELS[metric].lower()
        lines.append(f"- **{_slice_name(root)}** has a {label} of {pct(root['rate'])} against {pct(root['rest_rate'])} "
                     f"for all other swipes ({root['ratio']:.1f}× the overall rate, p {p_text(root['p_value'])}). "
                     "It is the flag that stands on its own.")
        outcome = metric_outcome(swipes, metric)
        rows = swipes.loc[outcome.index]
        in_root = rows[root["dimension"]].eq(root["group"])
        same_metric = slices["metric"].eq(metric) & slices["dimension"].ne(root["dimension"])
        proxies = slices[same_metric & slices["explained_by"].eq(_slice_name(root))]
        elevated = slices[same_metric & slices["verdict"].eq("ok") & slices["worse"]]
        for kind, rows_ in (("is flagged only because", proxies), ("looks high but is not significant after the "
                                                                    "correction, and", elevated)):
            for other in rows_.to_dict("records"):
                in_other = rows[other["dimension"]].eq(other["group"])
                outside = outcome[in_other & ~in_root]
                where = (f"{(in_other & in_root).sum() / in_other.sum():.0%} of its swipes are in {_slice_name(root)}; "
                         f"outside it the rate is {pct(outside.mean())} (n = {len(outside):,})" if len(outside)
                         else f"all of its swipes are in {_slice_name(root)}")
                lines.append(f"- **{_slice_name(other)}** ({pct(other['rate'])}, n = {other['n']:,}) {kind} {where}.")
    lines += ["", "A gap like this in real data is a question for operations before it is a fairness finding: the "
              "cause could be carrier or plan rules in that slice, a claims-processing fault, or unequal treatment. "
              "Either way, anything trained on this data would learn that these members are denied more often, "
              "which is why the gap is mitigated below."]
    return lines


def _mitigation_section(result: dict, settings: dict, charts: dict) -> list[str]:
    mitigation, slices = result["mitigation"], result["slices"]
    small = slices[slices["verdict"].eq("too small to judge") & slices["metric"].eq("denial_rate")]
    small_names = ", ".join(_slice_name(r) for r in small.to_dict("records")) or "none in this run"
    min_n_rule = (f"Slices with fewer than {settings['min_n']} swipes are reported as too small to judge, never as "
                  f"fair or unfair, so a handful of swipes cannot raise or hide a flag ({small_names}).")
    lines = ["## Mitigation", ""]
    if mitigation is None:
        return lines + ["Nothing is flagged, so no reweighing is needed: every swipe keeps weight 1.0 in "
                        "`data/bias/swipe_weights.parquet`.", "", min_n_rule]

    dim, group, overall = mitigation["dimension"], mitigation["group"], mitigation["overall"]
    root = mitigation["comparison"].loc[group]
    rows = [[g, f"{int(r.n):,}", pct(r.before), pct(r.after)] for g, r in mitigation["comparison"].iterrows()]
    lines += [
        f"### 1. Reweighing on `{dim}`", "",
        "Each swipe gets the weight P(group) × P(outcome) / P(group, outcome) (Kamiran & Calders, 2012), written to "
        "`data/bias/swipe_weights.parquet` (`swipe_id`, `weight`) for any model trained on swipes to use as "
        "`sample_weight`. Rows and outcomes are not changed.", "",
        f"![reweighing]({charts['reweighing']})", "",
        md_table([dim, "Swipes", "Before", "After"], rows), "",
        f"- Largest gap between {dim} groups: **{mitigation['max_gap_before_pts']:.1f} pts → "
        f"{mitigation['max_gap_after_pts']:.1f} pts**. The overall rate stays {pct(overall)}.",
        f"- **Trade-off, sample size:** the effective sample size drops from {mitigation['n']:,} to "
        f"{mitigation['effective_n']:,.0f} ({mitigation['effective_n'] / mitigation['n']:.0%}). Weights range from "
        f"{mitigation['weight_min']:.2f} to {mitigation['weight_max']:.2f}; in {group}, outcome-1 swipes count "
        f"{overall / root.before:.2f}× and the others {(1 - overall) / (1 - root.before):.2f}×.",
        f"- **Trade-off, pooled rate:** every {dim} group is brought to the pooled rate of {pct(overall)}, and that "
        f"pooled rate is itself raised by {group}. Groups that were below it count their outcome-1 swipes more, so the "
        "weighted data no longer shows how low their rates really are.",
        f"- **Trade-off, accuracy:** reweighing hides the gap from a model; it does not remove its cause. If {group}'s "
        "higher rate is real (for example a stricter carrier), a model trained with these weights will under-predict "
        "it there.", "",
        "### 2. Minimum-size reporting rule", "", min_n_rule, "",
        "### 3. Next steps", "",
        f"- Find the cause of the {_slice_name({'dimension': dim, 'group': group})} gap with operations before any "
        "model uses this data.",
        f"- If the data feeds a model, train with the weights above or stratify the train/test split by `{dim}`, "
        "then re-run this slicing on the model's predictions.",
        "- The DAG re-runs this check on every pipeline run, so a new skew shows up as a warning in the `bias` "
        "task log.",
    ]
    return lines


def _assumptions_section(settings: dict) -> list[str]:
    return [
        "## Assumptions and limitations", "",
        f"- **Employee status buckets are our own rule** ([decision #3]({DECISIONS_URL})): COBRA with an active "
        "benefit → COBRA Enrolled; terminated → Other; active benefit → Enrolled; otherwise Not Enrolled. "
        "A different rule would move members between `employee_status` slices.",
        f"- **Group-size buckets are our choice** ([decision #10]({DECISIONS_URL})): 1-9, 10-24, 25-49, 50-99, "
        "100-249, 250+. Other cut-offs would merge or split the `group_size_bucket` slices.",
        f"- **The 40-hour ticket SLA is an assumption** ([decision #4]({DECISIONS_URL})). This report slices swipes, "
        "not tickets, so it does not use the SLA; a future fairness view of SLA breaches would inherit it.",
        f"- The flag thresholds ({settings['abs_diff_pts']} pts, {settings['rel_ratio']}×, n ≥ {settings['min_n']}, "
        f"p < {settings['alpha']} before Bonferroni) are our defaults ([decisions #16–#20]({DECISIONS_URL})); "
        "change them in `params.yaml → bias`.",
        "- The data is synthetic: rates reflect the generator, so this report demonstrates the method and says nothing "
        "about real NSH members.",
        "- Dimensions are tested one at a time, plus the proxy check. Intersections such as state × age band are not "
        "tested, because most would fall below the minimum size at this data scale.",
    ]


def write_report(result: dict, params: dict, report_dir) -> Path:
    """Write bias_report.md, bias/slices.csv and the charts under `report_dir`; return the report path."""
    report_dir = Path(report_dir)
    chart_dir = report_dir / "bias"
    chart_dir.mkdir(parents=True, exist_ok=True)
    settings, slices, mitigation = params["bias"], result["slices"], result["mitigation"]
    slices.to_csv(chart_dir / "slices.csv", index=False)

    if mitigation is not None:
        metric, dimension = mitigation["metric"], mitigation["dimension"]
    else:
        metric, dimension = "denial_rate", "state" if "state" in result["dimensions"] else result["dimensions"][0]
    charts = {}
    name = f"{metric}_by_{dimension}.png"
    rate_chart(slices, metric, dimension, chart_dir / name)
    charts[f"{METRIC_LABELS[metric]} by {dimension}"] = f"bias/{name}"
    if mitigation is not None:
        name = f"reweighing_{dimension}.png"
        reweighing_chart(mitigation, chart_dir / name)
        charts["reweighing"] = f"bias/{name}"

    lines = [
        "# Bias report: swipe outcomes by subgroup", "",
        f"*Generated by `python -m nsh_pipeline.bias.slicing` from the processed tables, as of "
        f"{params['reference_date']}. Thresholds come from `params.yaml → bias`.*", "",
        *_summary_section(result, params), "",
        *_method_section(settings, result["dimensions"]), "",
        *_results_section(result, charts), "",
        *_interpretation_section(result), "",
        *_mitigation_section(result, settings, charts), "",
        *_assumptions_section(settings), "",
    ]
    path = report_dir / "bias_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
