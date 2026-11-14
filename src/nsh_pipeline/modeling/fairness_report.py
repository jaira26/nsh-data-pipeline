"""Model-bias report files: reports/model/bias/ (bias_report.md, slice_metrics.csv and PNG charts).

Charts use matplotlib's object API (no pyplot) and the phase-1 bias report's palette, so both
reports look alike and render without a display in Airflow and CI.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from nsh_pipeline.bias.report import CRITICAL, DECISIONS_URL, GRID, MUTED, SERIES, SURFACE, TEXT, TEXT_2, md_table
from nsh_pipeline.modeling.fairness_metrics import METRIC_LABELS, RATE_METRICS
from nsh_pipeline.preprocess.flatten import GROUP_SIZE_LABELS

REPORT_MD = "bias_report.md"
SLICE_CSV = "slice_metrics.csv"
SCATTER_PNG = "fairness_vs_performance.png"
SHORT = {"selection_rate": "selection", "false_positive_rate": "FPR", "recall": "recall", "precision": "precision"}
SUPPORT = {"selection_rate": "count", "false_positive_rate": "negatives", "recall": "positives",
           "precision": "predicted_positive"}


def pct(value) -> str:
    return "–" if value is None or pd.isna(value) else f"{value * 100:.1f}%"


def pts(value) -> str:
    return "–" if value is None or pd.isna(value) else f"{value:.1f} pts"


def p_text(value) -> str:
    return "–" if value is None or pd.isna(value) else ("< 0.001" if value < 0.001 else f"{value:.3f}")


def gated_metrics(settings: dict) -> list[str]:
    return [metric for metric in RATE_METRICS if settings["max_gap_pts"].get(metric) is not None]


def ordered_groups(column: str, groups) -> list[str]:
    """Group-size buckets in size order (1-9 ... 250+); everything else alphabetical."""
    groups = [str(group) for group in groups]
    if column == "group_size_bucket":
        rank = {label: i for i, label in enumerate(GROUP_SIZE_LABELS)}
        return sorted(groups, key=lambda g: (rank.get(g, len(rank)), g))
    return sorted(groups)


def _style(ax, percent_axis: str | None = "x"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_2, length=0, labelsize=8)
    if percent_axis:
        ax.grid(axis=percent_axis, color=GRID, linewidth=0.8)
        getattr(ax, f"{percent_axis}axis").set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_axisbelow(True)


def _rate_axis(ax, upper: float):
    """x axis from 0 to `upper` (room for labels), with ticks only up to 100%: a rate never exceeds it."""
    ax.set_xlim(0, upper)
    ax.set_xticks([tick / 10 for tick in range(0, 11, 2 if upper > 0.6 else 1) if tick / 10 <= min(upper, 1.0)])


def _verdicts(flags: pd.DataFrame, column: str, metric: str) -> dict:
    rows = flags[(flags["slice"] == column) & (flags["metric"] == metric)]
    return {str(r["group"]): ("flagged" if r["flagged"] else "ok" if r["judged"] else "too small")
            for r in rows.to_dict("records")}


def slice_chart(audit: dict, column: str, settings: dict, title: str, path: Path) -> Path:
    """2 x 2 panels, one per metric (selection rate, FPR, recall, precision): bars per group, overall dashed.

    Flagged groups are red with a "▲ flagged" label; groups whose denominator is too small to judge are
    gray. Each panel title says whether the metric is gated (with its limit) or only reported.
    """
    table = audit["table"][audit["table"]["slice"] == column].set_index("group")
    groups = ordered_groups(column, table.index)
    metrics = list(RATE_METRICS)
    fig = Figure(figsize=(10.4, 2 * (0.4 * len(groups) + 1.25) + 0.5), dpi=150, facecolor=SURFACE)
    axes = fig.subplots(2, 2, sharey=True).ravel()
    colors = {"flagged": CRITICAL, "too small": MUTED, "ok": SERIES[0]}
    for ax, metric in zip(axes, metrics):
        _style(ax)
        verdicts = _verdicts(audit["flags"], column, metric)
        for group in groups:  # metrics that are not gated have no flag rows: judge size from the support
            if group not in verdicts and table.loc[group, SUPPORT[metric]] < settings["min_support"]:
                verdicts[group] = "too small"
        values = [float(table.loc[g, metric]) if pd.notna(table.loc[g, metric]) else 0.0 for g in groups]
        y = list(range(len(groups)))
        ax.barh(y, values, height=0.68, color=[colors[verdicts.get(g, "ok")] for g in groups])
        ax.set_yticks(y, groups, color=TEXT, fontsize=9)
        overall = audit["overall"][metric]
        ax.axvline(overall, color=TEXT_2, linestyle="--", linewidth=1.1, zorder=1)
        ax.text(overall, len(groups) - 0.35, f" overall {pct(overall)}", color=TEXT_2, fontsize=7.5, va="bottom")
        for i, group in enumerate(groups):
            verdict = verdicts.get(group, "ok")
            note = {"flagged": "  ▲ flagged", "too small": "  too small"}.get(verdict, "")
            support = int(table.loc[group, SUPPORT[metric]])
            ax.annotate(f"{pct(table.loc[group, metric])}  n={support:,}{note}", (values[i], i), xytext=(3, 0),
                        textcoords="offset points", va="center", fontsize=7.5, zorder=3,
                        color=TEXT if verdict == "flagged" else TEXT_2,
                        fontweight="bold" if verdict == "flagged" else None,
                        bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.5})
        _rate_axis(ax, max(max(values), overall, 0.05) * 1.6)
        limit = settings["max_gap_pts"].get(metric)
        status = f"gated: flagged if > {limit:g} pts worse" if limit is not None else "reported, not gated"
        ax.set_title(f"{METRIC_LABELS[metric]}\n{status}", loc="left", color=TEXT, fontsize=9, fontweight="bold")
    fig.suptitle(title, x=0.01, ha="left", color=TEXT, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    return path


def scatter_chart(report: dict, settings: dict, path: Path) -> Path:
    """Every candidate on validation: largest gated gap (x) against PR-AUC (y), with the gate limit.

    Shape = original or reweighed; color = passes (blue) or fails (red) the gate; every point is labeled.
    """
    split, metric = report["selection_split"], settings["selection_metric"]
    fig = Figure(figsize=(7.5, 4.6), dpi=150, facecolor=SURFACE)
    ax = fig.add_subplot()
    _style(ax, percent_axis=None)
    ax.grid(axis="both", color=GRID, linewidth=0.8)
    xs = []
    for name, candidate in report["candidates"].items():
        summary = candidate[split]
        gaps = [w["gap_pts"] for w in summary["worst_gap_pts"].values() if w["gap_pts"] is not None]
        x, y = max(gaps) if gaps else 0.0, summary[metric]
        xs.append(x)
        color = SERIES[0] if summary["passed"] else CRITICAL
        marker = "D" if candidate["mitigation"] else "o"
        ax.scatter([x], [y], s=70, marker=marker, color=color, edgecolors=SURFACE, linewidths=1.5, zorder=3)
        label = name + ("  ★ selected" if name == report["selected_model"] else "")
        ax.annotate(label, (x, y), xytext=(7, 4), textcoords="offset points", fontsize=8, color=TEXT,
                    fontweight="bold" if name == report["selected_model"] else None)
    limits = sorted({limit for limit in settings["max_gap_pts"].values() if limit is not None})
    ax.set_xlim(0, max(xs + limits) * 1.35)
    low, high = ax.get_ylim()
    ax.set_ylim(low - (high - low) * 0.08, high)  # room under the lowest point for the limit label
    for limit in limits:
        ax.axvline(limit, color=TEXT_2, linestyle="--", linewidth=1.1, zorder=1)
        ax.text(limit, ax.get_ylim()[0], f" gate limit {limit:g} pts", color=TEXT_2, fontsize=7.5, va="bottom",
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.5})
    ax.set_xlabel(f"Largest gap of a group vs overall ({', '.join(SHORT[m] for m in gated_metrics(settings))}), "
                  "percentage points", color=TEXT_2, fontsize=8.5)
    ax.set_ylabel(f"Validation {metric.replace('_', '-').upper()}", color=TEXT_2, fontsize=8.5)
    handles = [Line2D([], [], marker="o", linestyle="", color=SERIES[0], label="passes gate"),
               Line2D([], [], marker="o", linestyle="", color=CRITICAL, label="fails gate"),
               Line2D([], [], marker="o", linestyle="", color=TEXT_2, label="original"),
               Line2D([], [], marker="D", linestyle="", color=TEXT_2, label="reweighed")]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=7.5, labelcolor=TEXT_2)
    ax.set_title(f"Fairness against performance, {split} split", loc="left", color=TEXT, fontsize=11,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    return path


def mitigation_chart(before: dict, after: dict, column: str, metrics: list[str], min_support: int, title: str,
                     path: Path) -> Path:
    """Each judged group's metric before (blue) and after (orange) mitigation, for the reweighed slice.

    Only groups with enough support to be judged are drawn (the note under each panel counts the rest),
    each bar is labeled, and the overall value before and after is in the panel title.
    """
    t_before = before["table"][before["table"]["slice"] == column].set_index("group")
    t_after = after["table"][after["table"]["slice"] == column].set_index("group")
    panels = []
    for metric in metrics:
        judged = [g for g in ordered_groups(column, t_before.index) if t_before.loc[g, SUPPORT[metric]] >= min_support
                  and pd.notna(t_before.loc[g, metric]) and g in t_after.index and pd.notna(t_after.loc[g, metric])]
        if judged:
            panels.append((metric, judged))
    if not panels:  # nothing large enough to compare: fall back to the first metric, every group
        panels = [(metrics[0], ordered_groups(column, t_before.index))]
    rows = max(len(groups) for _, groups in panels)
    fig = Figure(figsize=(4.4 * len(panels) + 1.0, 0.62 * rows + 2.0), dpi=150, facecolor=SURFACE)
    axes = fig.subplots(1, len(panels), squeeze=False)[0]
    for ax, (metric, groups) in zip(axes, panels):
        _style(ax)
        y = pd.Series(range(len(groups)), dtype=float)
        b = [float(t_before.loc[g, metric]) if pd.notna(t_before.loc[g, metric]) else 0.0 for g in groups]
        a = [float(t_after.loc[g, metric]) if pd.notna(t_after.loc[g, metric]) else 0.0 for g in groups]
        ax.barh(y + 0.19, b, height=0.36, color=SERIES[0], label="before")
        ax.barh(y - 0.19, a, height=0.36, color=SERIES[1], label="after reweighing")
        for values, offset in ((b, 0.19), (a, -0.19)):
            for i, value in enumerate(values):
                ax.annotate(pct(value), (value, i + offset), xytext=(3, 0), textcoords="offset points", va="center",
                            fontsize=7.5, color=TEXT_2, bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.4})
        ax.set_yticks(list(y), groups, color=TEXT, fontsize=9)
        _rate_axis(ax, max(b + a + [0.05]) * 1.3)
        ax.set_title(f"{METRIC_LABELS[metric]}\noverall {pct(before['overall'][metric])} → "
                     f"{pct(after['overall'][metric])}", loc="left", color=TEXT, fontsize=9, fontweight="bold")
        hidden = len(t_before) - len(groups)
        if hidden:
            ax.set_xlabel(f"{hidden} group(s) with fewer than {min_support} rows not shown", color=TEXT_2,
                          fontsize=7.5)
        ax.legend(loc="lower right", frameon=False, fontsize=7.5, labelcolor=TEXT_2)
    fig.suptitle(title, x=0.01, ha="left", color=TEXT, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    return path


def slice_csv(results: dict, path: Path) -> Path:
    """Every candidate x split x slice group, with the gated metrics each group is flagged on."""
    frames = []
    for name, result in results.items():
        for split in ("validation", "test"):
            audit = result[split]["audit"]
            flagged = audit["flags"][audit["flags"]["flagged"]].groupby(["slice", "group"])["metric"].apply(
                lambda m: ";".join(sorted(m)))
            table = audit["table"].copy()
            table["flagged_metrics"] = [flagged.get((s, g), "") for s, g in zip(table["slice"], table["group"])]
            table.insert(0, "split", split)
            table.insert(0, "candidate", name)
            frames.append(table)
    pd.concat(frames, ignore_index=True).to_csv(path, index=False, float_format="%.4f")
    return path


def _verdict_text(passed: bool) -> str:
    return "passed" if passed else "**FAILED**"


def markdown(report: dict, results: dict, settings: dict, charts: dict) -> str:
    gate, metrics = report["gate"], gated_metrics(settings)
    selected, audited = report["selected_model"], report["audited_model"]
    test_perf = report["performance"][report["audit_split"]]
    lines = ["# Model bias report", "",
             f"Data `{report['data_version']}` · code `{str(report['git_sha'])[:12]}` · generated "
             f"{report['generated_at']}", "", "## Verdict", ""]
    if selected:
        mitigation = report["mitigation_applied"]
        how = (f", reweighed on `{mitigation['slice']}`" if mitigation else ", no mitigation needed")
        lines.append(f"- **Fairness gate {'passed' if report['passed'] else 'FAILED'}.** Selected model: "
                     f"**`{selected}`** (base `{report['base_model']}`{how}), decision threshold "
                     f"{report['threshold']:.3f}. Saved in `{report['selected_model_dir']}`.")
    else:
        lines.append("- **Fairness gate FAILED: no candidate passes**, even after mitigation. The audit below is of "
                     f"the best-performing candidate, `{audited}`.")
    lines += [f"- {report['selection_reason']}",
              f"- Test split ({report['audit_split']}, never used to choose): PR-AUC {test_perf['pr_auc']:.3f}, "
              f"ROC-AUC {test_perf['roc_auc']:.3f}, recall {pct(test_perf['recall'])}, precision "
              f"{pct(test_perf['precision'])}."]
    if report["flags"]:
        for flag in report["flags"]:
            lines.append(f"- **Flagged on test: {flag['slice']} = {flag['group']}**, {METRIC_LABELS[flag['metric']]} "
                         f"{pct(flag['value'])} vs {pct(flag['overall'])} overall ({flag['gap_pts']:+.1f} pts, "
                         f"n = {flag['support']:,}, p {p_text(flag['p_value'])}).")
    else:
        lines.append(f"- No group is flagged on the test split ({gate['n_tests']} group × metric tests).")
    planted = settings.get("planted_skew")
    if planted:
        column, group = "state", str(planted.get("state"))
        caught = []
        for name, res in results.items():
            flagged = res["validation"]["audit"]["flagged"]
            if res["mitigation"] is None and (flagged["slice"].eq(column) & flagged["group"].eq(group)).any():
                caught.append(name)
        lines.append(f"- **Planted bias check:** `inject_issues.denial_skew` is on (state {group}, rate "
                     f"{planted.get('rate', 0):.0%}). Flagged on validation for "
                     + (", ".join(f"`{n}`" for n in caught) if caught else "**no candidate (not detected)**") + ".")

    lines += ["", "## Candidates", "",
              f"Ranked by validation {settings['selection_metric'].replace('_', '-').upper()}; the selected model "
              "is the first one that passes the gate on the validation split. Gaps are the largest "
              "worse-than-overall gap of any judged group.", ""]
    rows = []
    for rank, name in enumerate(report["ranking"], start=1):
        candidate = report["candidates"][name]
        val, test = candidate["validation"], candidate["test"]
        mitigation = candidate["mitigation"]
        rows.append([rank, f"`{name}`" + (" ★" if name == selected else ""),
                     f"reweighing on `{mitigation['slice']}`" if mitigation else "–",
                     f"{val['pr_auc']:.3f}",
                     *[pts((val["worst_gap_pts"].get(m) or {}).get("gap_pts")) for m in metrics],
                     _verdict_text(val["passed"]), f"{test['pr_auc']:.3f}", _verdict_text(test["passed"])])
    lines.append(md_table(["#", "Candidate", "Mitigation", "Val PR-AUC",
                           *[f"Val worst {SHORT[m]} gap" for m in metrics], "Val gate", "Test PR-AUC", "Test gate"],
                          rows))
    lines += ["", f"![Fairness against performance]({SCATTER_PNG})", ""]
    flagged_rows = [[f"`{name}`", f"{f['slice']} = {f['group']}", SHORT[f["metric"]], pct(f["value"]),
                     pct(f["overall"]), f"{f['gap_pts']:+.1f}", f"{f['support']:,}", p_text(f["p_value"])]
                    for name in report["ranking"] for f in report["candidates"][name]["validation"]["flags"]]
    if flagged_rows:
        lines += ["Flags on the validation split (these decided which candidates need mitigation):", "",
                  md_table(["Candidate", "Group", "Metric", "Group value", "Overall", "Gap (pts)", "n", "p"],
                           flagged_rows), ""]

    mitigated = [name for name in report["ranking"] if report["candidates"][name]["mitigation"]]
    lines += ["## Mitigation", ""]
    if not mitigated:
        lines.append("No candidate failed the gate on validation, so no mitigation was needed.")
    for name in mitigated:
        candidate, info = report["candidates"][name], report["candidates"][name]["mitigation"]
        base = report["candidates"][info["base_model"]]
        lines += [f"### `{info['base_model']}` → `{name}` (reweighing on `{info['slice']}`)", "",
                  f"- Training rows reweighed so the denial rate is equal in every `{info['slice']}` group: train gap "
                  f"{info['train_denial_gap_before_pts']:.1f} → {info['train_denial_gap_after_pts']:.1f} pts. "
                  f"Weights {info['weight_min']:.2f}–{info['weight_max']:.2f}; effective sample size "
                  f"{info['effective_n']:,.0f} of {info['train_rows']:,} rows ({info['effective_share']:.0%}).",
                  "- Same tuned hyperparameters, refit with the weights; threshold re-chosen on validation.", ""]
        rows = []
        for split in ("validation", "test"):
            for key in ("pr_auc", "recall", "precision"):
                rows.append([split, key.replace("_", "-").upper() if key == "pr_auc" else key,
                             f"{base[split][key]:.3f}", f"{candidate[split][key]:.3f}"])
            fpr = [results[n][split]["audit"]["overall"]["false_positive_rate"] for n in (info["base_model"], name)]
            rows.append([split, "overall FPR", pct(fpr[0]), pct(fpr[1])])
            for metric in metrics:
                rows.append([split, f"worst {SHORT[metric]} gap",
                             pts((base[split]["worst_gap_pts"].get(metric) or {}).get("gap_pts")),
                             pts((candidate[split]["worst_gap_pts"].get(metric) or {}).get("gap_pts"))])
            rows.append([split, "fairness gate", _verdict_text(base[split]["passed"]),
                         _verdict_text(candidate[split]["passed"])])
        lines += [md_table(["Split", "Metric", "Before", "After"], rows), ""]
        if name in charts:
            lines += [f"![{name} before and after]({charts[name]})", ""]
    if mitigated:
        lines += ["**Trade-offs.** Reweighing keeps every row and needs nothing at prediction time, but (1) it costs "
                  "effective sample size, (2) a model that may no longer use the slice's denial pattern gives up any "
                  "real signal in it, which shows as lower PR-AUC, and (3) it treats the symptom: if the skew comes "
                  "from an unfair denial process, that process still needs fixing. Group-specific thresholds "
                  "(Fairlearn `ThresholdOptimizer`) were not used: they need the group at prediction time and give "
                  "randomized decisions, which breaks the model contract.", ""]

    lines += [f"## Per-slice results: `{audited}`, {report['audit_split']} split", "",
              f"Decision threshold {report['threshold']:.3f}. Overall: selection "
              f"{pct(report['overall']['selection_rate'])}, "
              f"FPR {pct(report['overall']['false_positive_rate'])}, recall {pct(report['overall']['recall'])}, "
              f"precision {pct(report['overall']['precision'])}, {report['overall']['count']:,} swipes with "
              f"{report['overall']['positives']:,} denials.", ""]
    test_audit = results[audited][report["audit_split"]]["audit"]
    for column in gate["slice_columns"]:
        table = test_audit["table"][test_audit["table"]["slice"] == column].set_index("group")
        rows = []
        for group in ordered_groups(column, table.index):
            row = table.loc[group]
            record = next(r for r in report["per_slice_metrics"] if r["slice"] == column and r["group"] == group)
            note = ("▲ flagged: " + ", ".join(SHORT[m] for m in record["flagged_metrics"])
                    if record["flagged_metrics"] else
                    "too small for " + ", ".join(SHORT[m] for m in record["too_small_for"])
                    if record["too_small_for"] else "ok")
            rows.append([group, f"{int(row['count']):,}", f"{int(row['positives']):,}", pct(row["selection_rate"]),
                         pct(row["false_positive_rate"]), pct(row["recall"]), pct(row["precision"]), note])
        lines += [f"### {column}", "", md_table(["Group", "Swipes", "Denials", "Selection", "FPR", "Recall",
                                                 "Precision", "Verdict"], rows), ""]
        disparity = report["max_disparity"][column]
        gaps = [f"{SHORT[m]} {disparity[m]['gap_pts']:.1f} pts ({disparity[m]['worst_group']} vs "
                f"{disparity[m]['best_group']})" for m in RATE_METRICS if disparity[m]["gap_pts"] is not None]
        if gaps:
            lines += ["Largest gap between two judged groups: " + "; ".join(gaps) + ".", ""]
        if column in charts:
            lines += [f"![{column}]({charts[column]})", ""]

    limits = ", ".join(f"{SHORT[m]} {settings['max_gap_pts'][m]:g} pts" for m in metrics)
    lines += ["## Method", "",
              "- **Decisions:** score ≥ threshold = predicted denied. Threshold = best F1 on the validation split, "
              "the same rule as training and model validation.",
              "- **Metrics per group** (Fairlearn `MetricFrame` on each of "
              f"{', '.join(f'`{c}`' for c in gate['slice_columns'])}): "
              "selection rate, false-positive rate (approved swipes flagged as denials), recall (denials caught), "
              "precision, counts.",
              "- **Gate** (`params.yaml → model_bias`): a group is flagged when it is worse off than overall by more "
              f"than its metric's limit ({limits}), the metric's denominator has at least {gate['min_support']} rows, "
              f"and a one-sided Fisher exact test against all other rows gives p < {gate['alpha']} ÷ number of tests "
              "(Bonferroni). Selection rate and precision are reported, not gated: they also move with legitimate "
              "differences in denial risk between groups.",
              "- **Selection after bias:** candidates are ranked by validation PR-AUC; the first that passes the gate "
              "on the validation split is selected. The test split confirms the choice and is never used to make it.",
              f"- Decisions behind these rules: [docs/decisions.md]({DECISIONS_URL}) (rows on model bias).", ""]
    return "\n".join(lines)


def write_all(report: dict, results: dict, settings: dict, report_dir) -> dict:
    """Write bias_report.md, slice_metrics.csv and every chart into report_dir; return {name: path}."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    for old in report_dir.glob("*.png"):
        old.unlink()
    metrics = gated_metrics(settings)
    audited = report["audited_model"]
    test_audit = results[audited][report["audit_split"]]["audit"]
    charts = {}
    for column in report["gate"]["slice_columns"]:
        file = f"slices_{column}.png"
        slice_chart(test_audit, column, settings, f"{audited} by {column}, {report['audit_split']} split",
                    report_dir / file)
        charts[column] = file
    for name, result in results.items():
        info = result["mitigation"]
        if not info:
            continue
        file = f"mitigation_{name}.png"
        mitigation_chart(results[info["base_model"]][report["audit_split"]]["audit"],
                         result[report["audit_split"]]["audit"], info["slice"], metrics, settings["min_support"],
                         f"{info['base_model']} by {info['slice']}, before and after reweighing "
                         f"({report['audit_split']} split)", report_dir / file)
        charts[name] = file
    scatter_chart(report, settings, report_dir / SCATTER_PNG)
    slice_csv(results, report_dir / SLICE_CSV)
    (report_dir / REPORT_MD).write_text(markdown(report, results, settings, charts) + "\n", encoding="utf-8")
    return {**charts, "scatter": SCATTER_PNG, "csv": SLICE_CSV, "report": REPORT_MD}
