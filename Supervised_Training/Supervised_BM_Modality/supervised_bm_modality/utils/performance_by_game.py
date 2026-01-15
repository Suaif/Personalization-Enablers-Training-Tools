import pandas as pd
import matplotlib.pyplot as plt
import shutil
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

def get_performance_by_game(data_dir, exp_dir, split_df, split_predictions, show_plot=False):
    """
    Compute performance metrics for each game in the split.
    Generate plot and save to CSV.

    - data_dir: path to the data directory
    - exp_dir: path to the experiment directory
    - split_df: name of the split CSV file
    - split_predictions: name of the split predictions CSV file
    - show_plot: whether to show the plot
    """
    
    # Load data
    df = pd.read_csv(f"{data_dir}/{split_df}")
    predictions = pd.read_csv(f"{exp_dir}/{split_predictions}_predictions.csv")

    df["prediction"] = predictions["predicted_label"]
    df["true_label"] = predictions["true_label"]

    # Group by game
    grouped_df = df.groupby("game")

    def compute_metrics(x):
        y_true = x["true_label"]
        y_pred = x["prediction"]

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1-score-macro": f1_score(y_true, y_pred, average="macro"),
            "f1-score-micro": f1_score(y_true, y_pred, average="micro"),
            "f1-score-weighted": f1_score(y_true, y_pred, average="weighted"),
            "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        }

    performance_by_game = grouped_df.apply(compute_metrics)

    performance_df = pd.DataFrame(
        performance_by_game.tolist(),
        index=performance_by_game.index
    )

    performance_df.to_csv(f"{exp_dir}/performance_by_game_{split_predictions}.csv")

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1-score-macro",
        "f1-score-micro",
        "f1-score-weighted",
    ]

    # Plotting
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    exp_name = exp_dir.split("/")[-1]
    fig.suptitle(f"{split_predictions} Performance Metrics by Game - \n{exp_name}", fontsize=16)
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        performance_df[metric].plot(kind="bar", ax=ax)
        ax.set_title(f"{metric} by Game")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(f"{exp_dir}/performance_by_game_{split_predictions}.png")

    if show_plot:
        plt.show()
    
    plt.close(fig)
