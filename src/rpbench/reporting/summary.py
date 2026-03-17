"""Generate markdown summary report."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_summary(
    records: list[dict[str, Any]],
    output_path: str | Path,
    ols_figure_name: str,
    covariance_figure_name: str,
    table_name: str,
) -> None:
    """Write a markdown summary of the demo run."""

    n_total = len(records)
    mechanisms = sorted({r["mechanism"] for r in records if r["mechanism"] != "NonPrivate"})
    epsilons = sorted({r["epsilon"] for r in records if r["epsilon"] is not None})
    seeds = sorted({r["seed"] for r in records if r["seed"] is not None})

    delta_val = None
    dataset = None
    for r in records:
        if r["delta"] is not None:
            delta_val = r["delta"]
        if r["dataset"]:
            dataset = r["dataset"]

    np_mse = None
    for r in records:
        if r["mechanism"] == "NonPrivate":
            np_mse = r["downstream_metrics"]["test_mse"]

    lines = [
        "# WF-001 Demo Run Summary",
        "",
        f"**Dataset:** {dataset}  ",
        f"**Mechanisms:** {', '.join(mechanisms)}  ",
        f"**Task:** OLSFromRelease  ",
        f"**Epsilon grid:** {epsilons}  ",
        f"**Delta:** {delta_val:.2e}  " if delta_val else "**Delta:** 1/n^2  ",
        f"**Seeds:** {seeds}  ",
        f"**Total records:** {n_total}  ",
        "",
    ]

    if np_mse is not None:
        lines.append(f"**Non-private baseline test MSE:** {np_mse:.6f}")
        lines.append("")

    lines.extend([
        "## Results",
        "",
        f"See [{table_name}]({table_name}) for the aggregate table.",
        "",
        "### OLS Downstream Task",
        "",
        f"![OLS Downstream Utility Plot]({ols_figure_name})",
        "",
        "### Covariance Release Quality",
        "",
        f"![Covariance Release Quality Plot]({covariance_figure_name})",
        "",
        "## Outputs",
        "",
        "- Row-level results: `data/outputs/runs/demo_results.jsonl`",
        f"- Summary table: `{table_name}`",
        f"- OLS figure: `{ols_figure_name}`",
        f"- Covariance figure: `{covariance_figure_name}`",
        "",
    ])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
