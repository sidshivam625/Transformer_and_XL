import csv
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt


def read_training_log(log_path):
    path = Path(log_path)
    if not path.exists():
        raise FileNotFoundError(f"Training log not found: {log_path}")

    records = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            records.append({
                "epoch": int(row["Epoch"]),
                "train_loss": float(row["Train_Loss"]),
                "train_ppl": float(row["Train_PPL"]),
                "val_loss": float(row["Val_Loss"]),
                "val_ppl": float(row["Val_PPL"]),
            })
    return records


def read_eval_metrics(metrics_path):
    path = Path(metrics_path)
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    return data


def write_eval_metrics(metrics_path, metrics):
    path = Path(metrics_path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def plot_training_curves(records, output_dir="plots"):
    if not records:
        raise ValueError("No training records found to plot.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    epochs = [record["epoch"] for record in records]
    train_loss = [record["train_loss"] for record in records]
    val_loss = [record["val_loss"] for record in records]
    train_ppl = [record["train_ppl"] for record in records]
    val_ppl = [record["val_ppl"] for record in records]

    plt.style.use("seaborn-v0_8-darkgrid")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, train_loss, marker="o", label="Train Loss")
    axes[0].plot(epochs, val_loss, marker="o", label="Val Loss")
    axes[0].set_title("Loss vs Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, train_ppl, marker="o", label="Train PPL")
    axes[1].plot(epochs, val_ppl, marker="o", label="Val PPL")
    axes[1].set_title("Perplexity vs Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Perplexity")
    axes[1].legend()

    fig.tight_layout()
    loss_plot = output_path / "training_metrics.png"
    fig.savefig(loss_plot, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return str(loss_plot)


def plot_evaluation_summary(eval_metrics, output_dir="plots"):
    if not eval_metrics:
        return None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    labels = []
    values = []
    if "loss" in eval_metrics:
        labels.append("Eval Loss")
        values.append(eval_metrics["loss"])
    if "ppl" in eval_metrics:
        labels.append("Eval PPL")
        values.append(eval_metrics["ppl"])

    if not labels:
        return None

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values, color=["#4c78a8", "#f58518"][: len(labels)])
    ax.set_title("Evaluation Summary")
    ax.set_ylabel("Value")

    for index, value in enumerate(values):
        ax.text(index, value, f"{value:.4f}", ha="center", va="bottom")

    fig.tight_layout()
    summary_plot = output_path / "evaluation_summary.png"
    fig.savefig(summary_plot, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return str(summary_plot)


def summarize_training(records):
    if not records:
        return {}

    best_epoch = min(records, key=lambda item: item["val_loss"])
    final_epoch = records[-1]

    return {
        "epochs": len(records),
        "best_epoch": best_epoch["epoch"],
        "best_val_loss": best_epoch["val_loss"],
        "best_val_ppl": best_epoch["val_ppl"],
        "final_train_loss": final_epoch["train_loss"],
        "final_val_loss": final_epoch["val_loss"],
        "final_train_ppl": final_epoch["train_ppl"],
        "final_val_ppl": final_epoch["val_ppl"],
    }


def main(log_path="training_log.txt", eval_metrics_path="evaluation_metrics.json", output_dir="plots"):
    records = read_training_log(log_path)
    train_summary = summarize_training(records)
    training_plot = plot_training_curves(records, output_dir=output_dir)

    eval_metrics = read_eval_metrics(eval_metrics_path)
    eval_plot = plot_evaluation_summary(eval_metrics, output_dir=output_dir)

    summary = {
        "training": train_summary,
        "evaluation": eval_metrics,
        "training_plot": training_plot,
        "evaluation_plot": eval_plot,
    }

    summary_path = Path(output_dir) / "metrics_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved plots to: {output_dir}")
    print(f"Saved summary to: {summary_path}")
    return summary


if __name__ == "__main__":
    main()