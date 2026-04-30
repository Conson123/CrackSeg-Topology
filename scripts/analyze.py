"""Post-hoc analysis: load K-fold summaries, compare experiments,
generate charts and LaTeX tables."""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def beautify_exp_name(exp_name):
    name = exp_name.replace("finetune_", "").replace("Unet_", "").replace("efficientnet-b4", "")
    if "_loss_" not in name and "_att_" not in name and "_post_" not in name:
        return "Baseline"
    reps = {
        "_loss_focal": "Focal Loss", "_loss_tversky": "Tversky Loss",
        "_loss_lovasz": "Lovasz Loss", "_loss_lovasz_softmax": "Lovasz-Softmax",
        "_att_cbam": "CBAM", "_att_se": "SE-Net", "_att_eca": "ECA-Net",
        "_post_morphological": "Morphological", "_post_crf": "CRF",
    }
    for old, new in reps.items():
        if old in name:
            name = name.replace(old, f" + {new}" if name.count("_") > 1 else new)
    return name.strip("_ ")


def load_all_results(results_root="run_results"):
    print("Loading experiment results ...")
    files = glob.glob(os.path.join(results_root, "*", "kfold_summary.csv"))
    if not files:
        print("No result files found.")
        return None
    results = {}
    for fp in files:
        exp = beautify_exp_name(os.path.basename(os.path.dirname(fp)))
        try:
            df = pd.read_csv(fp)
            row = df.iloc[-1]
            d = {
                "IoU": row.get("val_iou_score", 0),
                "F1": row.get("val_f1_score", 0),
                "Precision": row.get("val_precision", 0),
                "Recall": row.get("val_recall", 0),
                "Accuracy": row.get("val_accuracy", 0),
                "Length_MAE(cm)": row.get("val_mae_cm", 0),
                "Loss": row.get("val_loss", 0),
            }
            if "val_width_mae_cm" in row:
                d["Width_MAE(cm)"] = row.get("val_width_mae_cm", 0)
            results[exp] = d
            print(f"  Loaded: {exp}")
        except Exception as e:
            print(f"  Failed: {fp} ({e})")
    print(f"Loaded {len(results)} experiments.\n")
    return results


def create_comparison_table(results):
    df = pd.DataFrame(results).T.sort_values("IoU", ascending=False)
    print("=" * 100)
    print("Experiment Comparison Table")
    print("=" * 100)
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))
    df.to_csv("experiment_comparison.csv", float_format="%.4f")
    print("\nSaved to experiment_comparison.csv")
    return df


def find_best_methods(df):
    print("\n" + "=" * 80)
    print("Best Method per Metric")
    print("=" * 80)
    for col in df.columns:
        if col in ("Length_MAE(cm)", "Width_MAE(cm)", "Loss"):
            idx = df[col].idxmin()
        else:
            idx = df[col].idxmax()
        print(f"  {col:25s}: {idx:40s} = {df.loc[idx, col]:.4f}")


def plot_metrics_comparison(df, save_path="metrics_comparison.png"):
    metrics = [m for m in ["IoU", "F1", "Precision", "Recall",
                           "Length_MAE(cm)", "Width_MAE(cm)"] if m in df.columns]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Experimental Results Comparison", fontsize=16, fontweight="bold")
    axes = axes.flatten()
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        ascending = "MAE" in metric or "Error" in metric
        data = df[metric].sort_values(ascending=ascending)
        bars = ax.barh(range(len(data)), data.values)
        for i, name in enumerate(data.index):
            bars[i].set_color("lightcoral" if "Baseline" in name else "skyblue")
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(data.index, fontsize=9)
        ax.set_xlabel(metric, fontsize=11, fontweight="bold")
        ax.set_title(f"{metric} Comparison", fontsize=12, fontweight="bold")
        for i, v in enumerate(data.values):
            ax.text(v, i, f" {v:.4f}", va="center", fontsize=8)
        ax.grid(axis="x", alpha=0.3)
    for i in range(len(metrics), len(axes)):
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Comparison chart saved to: {save_path}")
    plt.close()


def plot_radar_chart(df, save_path="radar_comparison.png"):
    top5 = df.head(5)
    metrics = ["IoU", "F1", "Precision", "Recall", "Accuracy"]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))
    for name, row in top5.iterrows():
        vals = row[metrics].tolist() + [row[metrics[0]]]
        ax.plot(angles, vals, "o-", linewidth=2, label=name)
        ax.fill(angles, vals, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("Performance Radar (Top 5)", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Radar chart saved to: {save_path}")
    plt.close()


def generate_latex_table(df, save_path="latex_table.txt"):
    metrics = [m for m in ["IoU", "F1", "Precision", "Recall", "MAE(cm)"] if m in df.columns]
    with open(save_path, "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Experimental Results Comparison}\n\\label{tab:results}\n")
        f.write("\\begin{tabular}{l" + "c" * len(metrics) + "}\n\\hline\n")
        f.write("Method & " + " & ".join(metrics) + " \\\\\n\\hline\n")
        for name, row in df.iterrows():
            f.write(name.replace("_", "\\_"))
            for m in metrics:
                if m in row:
                    v = row[m]
                    is_best = ((m == "MAE(cm)" and v == df[m].min()) or
                               (m != "MAE(cm)" and v == df[m].max()))
                    f.write(f" & \\textbf{{{v:.4f}}}" if is_best else f" & {v:.4f}")
            f.write(" \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    print(f"LaTeX table saved to: {save_path}")


def run_analysis():
    print("\n" + "=" * 80)
    print("Experiment Analysis")
    print("=" * 80 + "\n")
    results = load_all_results()
    if results is None:
        return
    df = create_comparison_table(results)
    find_best_methods(df)
    plot_metrics_comparison(df)
    plot_radar_chart(df)
    generate_latex_table(df)
    print("\nAnalysis complete.")
