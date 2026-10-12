"""Plot an existing benchmark report; matplotlib is needed only for this script."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(source: Path, output: Path):
    report = json.loads(source.read_text())
    scenarios = list(dict.fromkeys(row["scenario"] for row in report["rows"]))
    labels = {
        "random": "Random",
        "fisher": "Fisher baseline",
        "fisher_family": "Fisher + family limit",
        "gain": "Information gain",
        "gain_explore": "Gain + exploration",
        "gain_family": "Gain + family limit",
        "gain_family_explore": "Gain + family + exploration",
    }
    titles = {
        "matched_model": "Scoring model matches the data",
        "family_dependence": "Related items share an unmodeled effect",
        "calibration_shift": "True difficulty differs from stored difficulty",
    }
    fig, axes = plt.subplots(len(scenarios), 2, figsize=(15, 11), squeeze=False)
    fig.patch.set_facecolor("#f4f7fb")
    for index, scenario in enumerate(scenarios):
        rows = [row for row in report["rows"] if row["scenario"] == scenario]
        colors = ["#3766bf" if row["policy"] == "fisher" else "#168879" if row["policy"] == "fisher_family" else "#8795a8" for row in rows]
        for column, metric in enumerate(("rmse", "coverage_95")):
            ax = axes[index][column]
            values = [row["metrics"][metric]["mean"] for row in rows]
            errors = [row["metrics"][metric]["sd_across_seeds"] for row in rows]
            ax.barh(range(len(rows)), values, xerr=errors, color=colors, capsize=3, height=0.65)
            ax.set_yticks(range(len(rows)))
            ax.set_yticklabels([labels[row["policy"]] for row in rows], fontsize=9)
            ax.invert_yaxis()
            ax.set_xlim(0, 1.05 if column else 0.9)
            ax.set_xlabel("95% interval coverage" if column else "Ability RMSE · lower is better", fontsize=10)
            ax.set_title(titles[scenario], fontsize=11, loc="left", pad=12)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="x", alpha=0.2)
            ax.set_axisbelow(True)
            if column:
                ax.axvline(0.95, color="#bd4c43", linestyle="--", linewidth=1.3)
    fig.suptitle("Adaptive testing: useful tradeoffs, no universal winner", fontsize=20, x=0.04, ha="left")
    fig.text(0.04, 0.925, f"Synthetic data · {len(report['seeds'])} seeds × {report['learners_per_seed']} learners · {report['answers_per_learner']} answers per policy · error bars = SD across seeds", fontsize=11)
    fig.text(0.04, 0.02, "Family limits help these stress cases but cost accuracy under the matched model. Coverage still falls below 95% under correlated responses.", fontsize=11)
    fig.tight_layout(rect=(0.02, 0.045, 0.99, 0.91), h_pad=2.5, w_pad=3.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("docs/measurement_results.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/measurement_results.png"))
    args = parser.parse_args()
    plot(args.input, args.output)
