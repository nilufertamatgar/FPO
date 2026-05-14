import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_ROOT = Path("/dapustor/nilufer/ADFLIP/INFERENCE/Results_train")
OUT_CSV = RESULTS_ROOT / "model_metrics_summary.csv"
OUT_PLOTS_DIR = RESULTS_ROOT / "metric_plots"


def extract_float(pattern: str, text: str):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def parse_accuracy_log(path: Path):
    text = path.read_text(encoding="utf-8")
    return {
        "pairwise_accuracy": extract_float(r"Pairwise accuracy \(GT > Neg\):\s*([0-9.]+)", text),
    }


def parse_sr_crr_log(path: Path):
    text = path.read_text(encoding="utf-8")
    return {
        "mean_sequence_recovery": extract_float(r"Mean sequence recovery:\s*([0-9.]+)", text),
        "mean_critical_recovery": extract_float(r"Mean critical recovery:\s*([0-9.]+)", text),
        "mean_perplexity": extract_float(r"Mean perplexity overall:\s*([0-9.]+)", text),
    }


def build_model_colors(models):
    cmap = plt.get_cmap("tab20")
    return {model: cmap(i % 20) for i, model in enumerate(models)}


def add_value_labels(ax, bars, values, as_percent=True):
    for bar, val in zip(bars, values):
        if pd.isna(val):
            continue
        label = f"{val * 100:.2f}%" if as_percent else f"{val:.2f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )


def make_bar_plot(df, metric_col, ylabel, title, out_path, model_order, model_colors, as_percent=True):
    plot_df = df.set_index("model").reindex(model_order).reset_index()
    plot_df = plot_df.dropna(subset=[metric_col])

    if plot_df.empty:
        print(f"Skipping plot for {metric_col}: no valid data")
        return

    colors = [model_colors[m] for m in plot_df["model"]]

    plt.figure(figsize=(max(10, len(model_order) * 0.7), 6))
    ax = plt.gca()
    bars = ax.bar(plot_df["model"], plot_df[metric_col], color=colors)

    ax.set_xticks(range(len(plot_df["model"])))
    ax.set_xticklabels(plot_df["model"], rotation=90)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    add_value_labels(ax, bars, plot_df[metric_col].tolist(), as_percent=as_percent)

    ymax = plot_df[metric_col].max()
    ax.set_ylim(0, max(ymax * 1.15, 0.05))

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    OUT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for model_dir in sorted(RESULTS_ROOT.iterdir()):
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        accuracy_log = model_dir / "accuracy.log"
        sr_crr_log = model_dir / "SR_CRR.log"

        row = {
            "model": model_name,
            "pairwise_accuracy": None,
            "mean_sequence_recovery": None,
            "mean_critical_recovery": None,
            "mean_perplexity": None,
            "has_accuracy_log": accuracy_log.exists(),
            "has_sr_crr_log": sr_crr_log.exists(),
        }

        if accuracy_log.exists():
            row.update(parse_accuracy_log(accuracy_log))

        if sr_crr_log.exists():
            row.update(parse_sr_crr_log(sr_crr_log))

        rows.append(row)

    df = pd.DataFrame(rows)

    # Keep one fixed model order across all plots.
    model_order = df["model"].tolist()

    # Keep one fixed color per model across all plots.
    model_colors = build_model_colors(model_order)

    df.to_csv(OUT_CSV, index=False)

    make_bar_plot(
        df,
        "pairwise_accuracy",
        "Pairwise Accuracy",
        "Pairwise Accuracy (GT > Neg)",
        OUT_PLOTS_DIR / "pairwise_accuracy.png",
        model_order,
        model_colors,
        as_percent=True,
    )

    make_bar_plot(
        df,
        "mean_sequence_recovery",
        "Sequence Recovery",
        "Mean Sequence Recovery",
        OUT_PLOTS_DIR / "mean_sequence_recovery.png",
        model_order,
        model_colors,
        as_percent=True,
    )

    make_bar_plot(
        df,
        "mean_critical_recovery",
        "Critical Recovery",
        "Mean Critical Recovery",
        OUT_PLOTS_DIR / "mean_critical_recovery.png",
        model_order,
        model_colors,
        as_percent=True,
    )

    make_bar_plot(
        df,
        "mean_perplexity",
        "Perplexity",
        "Mean Perplexity Overall",
        OUT_PLOTS_DIR / "mean_perplexity.png",
        model_order,
        model_colors,
        as_percent=False,
    )

    print(f"Wrote summary CSV to: {OUT_CSV}")
    print(f"Wrote plots to: {OUT_PLOTS_DIR}")


if __name__ == "__main__":
    main()
