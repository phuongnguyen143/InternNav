import json
from pathlib import Path

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# Add 1–3 evaluation log files here.
# ============================================================
FILES = {
    "S2+S1-Default": "/home/khangnh11/VR/InternNav/checkpoints/InternVLA-N1-w-NavDP/Eval-bkhn2/eval_metrics.jsonl",
    "S2+S1-Default+BKHN1": "/home/khangnh11/VR/InternNav/checkpoints/InternVLA-N1-DualVLN-train-from-internvla-n1-w-navdp-v2/Eval/eval_metrics.jsonl",
    "S2+S1-Scratch": "/home/khangnh11/VR/InternNav/checkpoints/InternVLA-N1-DualVLN-train-only-30deg-scratch-navdp-v1/Eval/eval_metrics.jsonl",
}
OUTPUT_DIR = Path("eval_plots")


# Global publication-style font settings
plt.rcParams.update({
    "font.size": 16,
    "font.weight": "bold",
    "axes.labelsize": 20,
    "axes.labelweight": "bold",
    "axes.titlesize": 22,
    "axes.titleweight": "bold",
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})


def load_eval_log(file_path: str) -> dict[str, list[float]]:
    steps = []
    losses = []
    avg_losses = []

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"Warning: skipping malformed JSON in "
                    f"{file_path.name}, line {line_number}"
                )
                continue

            if record.get("event") != "eval_step":
                continue

            step = record.get("step")
            loss = record.get("loss")
            avg_loss = record.get("avg_loss")

            if step is None or loss is None:
                continue

            steps.append(int(step))
            losses.append(float(loss))

            if avg_loss is None:
                avg_loss = sum(losses) / len(losses)

            avg_losses.append(float(avg_loss))

    if not steps:
        raise ValueError(f"No eval_step records found in {file_path}")

    ordered = sorted(
        zip(steps, losses, avg_losses),
        key=lambda item: item[0],
    )

    return {
        "step": [item[0] for item in ordered],
        "loss": [item[1] for item in ordered],
        "avg_loss": [item[2] for item in ordered],
    }


def plot_metric(
    all_results: dict[str, dict[str, list[float]]],
    metric: str,
    title: str,
    y_label: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 8))

    for model_name, result in all_results.items():
        ax.plot(
            result["step"],
            result[metric],
            linewidth=4.0,       # Bold/thick line
            label=model_name,
        )

    ax.set_xlabel(
        "Evaluation Step",
        fontsize=20,
        fontweight="bold",
        labelpad=10,
    )

    ax.set_ylabel(
        y_label,
        fontsize=20,
        fontweight="bold",
        labelpad=10,
    )

    ax.set_title(
        title,
        fontsize=24,
        fontweight="bold",
        pad=18,
    )

    # Bold tick labels
    for tick in ax.get_xticklabels():
        tick.set_fontsize(16)
        tick.set_fontweight("bold")

    for tick in ax.get_yticklabels():
        tick.set_fontsize(16)
        tick.set_fontweight("bold")

    ax.grid(
        True,
        linestyle="--",
        linewidth=1.2,
        alpha=0.5,
    )

    legend = ax.legend(
        loc="best",
        fontsize=16,
        frameon=True,
        framealpha=1.0,
        edgecolor="black",
        fancybox=False,
        borderpad=0.8,
        handlelength=3,
    )

    # Bold legend text
    for text in legend.get_texts():
        text.set_fontweight("bold")

    # Make plot border thicker
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)

    ax.tick_params(
        axis="both",
        width=1.8,
        length=7,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Saved: {output_path}")

    plt.show()
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for model_name, file_path in FILES.items():
        result = load_eval_log(file_path)
        all_results[model_name] = result

        print(
            f"{model_name}: "
            f"{len(result['step'])} steps, "
            f"final loss={result['loss'][-1]:.6f}, "
            f"final avg loss={result['avg_loss'][-1]:.6f}"
        )

    plot_metric(
        all_results=all_results,
        metric="loss",
        title="Evaluation Loss Comparison",
        y_label="Loss",
        output_path=OUTPUT_DIR / "step_vs_loss.png",
    )

    plot_metric(
        all_results=all_results,
        metric="avg_loss",
        title="Average Evaluation Loss Comparison",
        y_label="Average Loss",
        output_path=OUTPUT_DIR / "step_vs_avg_loss.png",
    )


if __name__ == "__main__":
    main()