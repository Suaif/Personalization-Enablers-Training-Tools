import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import f1_score, balanced_accuracy_score


def apply_inter_subject_split(data_dir, default_splits):
    """
    Applies inter-subject splitting logic if enabled in the configuration.
    Moves 50% of data from one test subject to the train set.
    
    Args:
        data_dir (str): Path to the data directory containing train.csv, val.csv, test.csv.
        
    Returns:
        dict: A dictionary mapping split names ('train', 'val', 'test') to their corresponding CSV filenames.
    """
    
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    train_df = pd.read_csv(train_path, index_col=0)
    test_df = pd.read_csv(test_path, index_col=0)
            
    test_subjects = test_df['subject'].unique()
    
    if len(test_subjects) > 0:

        subject_to_split = test_subjects[0]
        print(f"Moving 50% of data for subject {subject_to_split} from Test to Train (Inter_subject mode)")
        
        subject_rows = test_df[test_df['subject'] == subject_to_split]
        
        rows_to_move = subject_rows.sample(frac=0.5)
        
        test_df = test_df.drop(rows_to_move.index)
        
        train_df = pd.concat([train_df, rows_to_move])
        
        new_train_filename = "train_intersubject.csv"
        new_test_filename = "test_intersubject.csv"
        
        train_df.to_csv(os.path.join(data_dir, new_train_filename))
        test_df.to_csv(os.path.join(data_dir, new_test_filename))
        
        print(f"Saved modified splits to {new_train_filename} and {new_test_filename}")
        
        return {
            'train': new_train_filename,
            'val': "val.csv",
            'test': new_test_filename
        }
    
    return default_splits

def get_performance_by_subject(data_dir, exp_dir, split_paths, show_plot=False):
    """
    Generates a 3x2 grid plot for Train, Val, and Test splits.
    Columns: F1-Weighted, Balanced Accuracy.
    Calculates metrics for each subject and saves to a JSON file.
    """
    exp_name = os.path.basename(exp_dir)
    
    all_subjects = set()
    split_data = {}
    
    for split_name, split_path in split_paths.items():
        data_path = os.path.join(data_dir, f"{split_path}")
        pred_path = os.path.join(exp_dir, f"{split_name}_predictions.csv")
        
        if os.path.exists(data_path) and os.path.exists(pred_path):
            data_df = pd.read_csv(data_path)
            pred_df = pd.read_csv(pred_path)
             # Ensure lengths match
            min_len = min(len(data_df), len(pred_df))
            data_df = data_df.iloc[:min_len]
            pred_df = pred_df.iloc[:min_len]
            
            merged_df = data_df.copy()
            merged_df["prediction"] = pred_df["predicted_label"].values
            merged_df["true_label"] = pred_df["true_label"].values
                 
            subjects = merged_df["subject"].unique()
            all_subjects.update(subjects)
            split_data[split_name] = merged_df
            
    # Plotting
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle(f"Performance by Subject \n {exp_name}", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(hspace=0.4)

    sorted_subjects = sorted(list(all_subjects))  
    palette = sns.color_palette("husl", len(sorted_subjects))
    subject_color_map = dict(zip(sorted_subjects, palette))
    
    # Store all metrics to save later
    all_metrics = {}

    for i, split in enumerate(split_paths.keys()):
        if split not in split_data:
            continue
        
        merged_df = split_data[split]
        subject_metrics = []
        
        # Calculate metrics for each subject present in this split
        for subj in sorted_subjects:
            subj_data = merged_df[merged_df["subject"] == subj]
            if len(subj_data) == 0:
                continue
                
            y_true = subj_data["true_label"]
            y_pred = subj_data["prediction"]
            
            f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            
            subject_metrics.append({
                "subject": str(subj),
                "f1-score-weighted": f1_w,
                "balanced-accuracy": bal_acc
            })
            
        metrics_df = pd.DataFrame(subject_metrics)
        all_metrics[split] = metrics_df.to_dict(orient="records")
        
        if metrics_df.empty:
            continue

        # Plotting
        ax_f1 = axes[i, 0]
        ax_ba = axes[i, 1]

        subjects_in_split = [s for s in sorted_subjects if s in metrics_df["subject"].values]
        
        sns.barplot(
            data=metrics_df, 
            x="subject", 
            y="f1-score-weighted", 
            ax=ax_f1, 
            hue="subject", 
            palette=subject_color_map, 
            legend=False,
            order=subjects_in_split
        )
        ax_f1.set_title(f"{split.upper()} - F1 Weighted")
        ax_f1.set_ylim(0, 1.05)
        ax_f1.set_xlabel("")
        ax_f1.tick_params(axis='x', rotation=45, labelsize=8)
        
        sns.barplot(
            data=metrics_df, 
            x="subject", 
            y="balanced-accuracy", 
            ax=ax_ba, 
            hue="subject", 
            palette=subject_color_map, 
            legend=False,
            order=subjects_in_split
        )
        ax_ba.set_title(f"{split.upper()} - Balanced Accuracy")
        ax_ba.set_ylim(0, 1.05)
        ax_ba.set_xlabel("")
        ax_ba.tick_params(axis='x', rotation=45, labelsize=8)

    # Save metrics to JSON
    json_path = os.path.join(exp_dir, "performance_by_subject.json")
    with open(json_path, "w") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"Performance by subject metrics saved to {json_path}")
        
    # Save plot
    plot_path = os.path.join(exp_dir, "performance_by_subject.png")
    plt.savefig(plot_path)
    print(f"Performance by subject plot saved to {plot_path}")
    
    if show_plot:
        plt.show()
    
    plt.close(fig)
