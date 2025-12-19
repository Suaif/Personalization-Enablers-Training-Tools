import json
import os
import argparse
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Configuration
# Using absolute paths as per the user's setup
BASE_DIR = r'c:\Users\ismas\Documents\X2Learn\Personalization-Enablers-Training-Tools'
DATA_DIR = os.path.join(BASE_DIR, 'outputs\XRoom\shimmer')
FEATURE_DIR = os.path.join(DATA_DIR, 'eda_features')
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
VAL_CSV = os.path.join(DATA_DIR, 'val.csv')
TEST_CSV = os.path.join(DATA_DIR, 'test.csv')

class ShimmerDataset(Dataset):
    def __init__(self, csv_file, feature_dir, label_encoder=None):
        self.feature_dir = feature_dir
        self.data_frame = pd.read_csv(csv_file)
        
        # Encode labels
        if label_encoder:
            self.label_encoder = label_encoder
            # Handle unseen labels by assigning a default -1
            le_dict = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
            self.data_frame['encoded_labels'] = self.data_frame['labels'].apply(lambda x: le_dict.get(x, -1))
        else:
            self.label_encoder = LabelEncoder()
            self.data_frame['encoded_labels'] = self.label_encoder.fit_transform(self.data_frame['labels'])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.data_frame.iloc[idx]['files']
        label = self.data_frame.iloc[idx]['encoded_labels']
        
        file_path = os.path.join(self.feature_dir, filename)
        
        # Handling potential filename mismatches
        if not os.path.exists(file_path):
             if filename.startswith('Participants_'):
                 alt_name = 'Participant_' + filename[len('Participants_'):]
                 alt_path = os.path.join(self.feature_dir, alt_name)
                 if os.path.exists(alt_path):
                     file_path = alt_path
             elif filename.startswith('Participant_'):
                 alt_name = 'Participants_' + filename[len('Participant_'):]
                 alt_path = os.path.join(self.feature_dir, alt_name)
                 if os.path.exists(alt_path):
                     file_path = alt_path
        
        try:
            features = np.load(file_path)
        except Exception as e:
            features = np.zeros(100, dtype=np.float32) 
            
        if features.ndim == 1:
            features = features[np.newaxis, :]
            
        return torch.from_numpy(features).float(), torch.tensor(label).long()

class Simple1DCNN(nn.Module):
    def __init__(self, num_classes, input_channels=1):
        super(Simple1DCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def calculate_metrics(y_true, y_pred, average='macro'):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1-score-macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1-score-micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'f1-score-weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0)
    }
    return metrics

def get_predictions(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # Filter out invalid labels (-1)
            valid_mask = labels != -1
            if not valid_mask.any():
                continue
            
            inputs = inputs[valid_mask]
            labels = labels[valid_mask]
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)

def get_numpy_data(dataset):
    """Extracts features and labels from dataset into numpy arrays, filtering -1 labels."""
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    all_X = []
    all_y = []
    for X, y in loader:
        X = X.numpy()
        y = y.numpy()
        # Squeeze channel dim: (Batch, 1, Features) -> (Batch, Features)
        if X.ndim == 3 and X.shape[1] == 1:
            X = X.squeeze(axis=1)
            
        mask = y != -1
        if mask.any():
            all_X.append(X[mask])
            all_y.append(y[mask])
            
    return np.concatenate(all_X), np.concatenate(all_y)

def save_plots(train_labels, train_preds, val_labels, val_preds, test_labels, test_preds, 
               label_encoder, exp_name, model_name, results_folder):
    """Generates and saves the comparison histograms."""
    # --- DARK MODE TOGGLE ---
    DARK_MODE = False
    if DARK_MODE:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    sets_data = [
        (train_labels, train_preds, 'Train Set'),
        (val_labels, val_preds, 'Validation Set'),
        (test_labels, test_preds, 'Test Set')
    ]

    for ax, (labels, preds, title) in zip(axes, sets_data):
        if len(labels) == 0:
            ax.text(0.5, 0.5, 'No Data', ha='center')
            ax.set_title(title)
            continue
            
        pred_names = label_encoder.inverse_transform(labels) # Use labels to get classes safely? No, preds are indices
        # Safe inverse transform
        try:
            pred_names_str = label_encoder.inverse_transform(preds)
            true_names_str = label_encoder.inverse_transform(labels)
        except Exception:
             # Fallback if raw integers
             pred_names_str = preds
             true_names_str = labels

        df_pred = pd.DataFrame({'Class': pred_names_str, 'Type': 'Predictions'})
        df_true = pd.DataFrame({'Class': true_names_str, 'Type': 'True Labels'})
        combined_df = pd.concat([df_true, df_pred], ignore_index=True)
        
        # Determine order from classes if possible
        order = list(label_encoder.classes_) if hasattr(label_encoder, 'classes_') else None
        
        sns.countplot(data=combined_df, x='Class', hue='Type', ax=ax, order=order)
        ax.set_title(title)
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)

    fig.suptitle(f"Predictions Histogram - {model_name}\n{exp_name}")
    plt.tight_layout()
    filename = f'predictions_histogram_{model_name}.png'
    plt.savefig(os.path.join(results_folder, filename))
    plt.close(fig)
    print(f"Saved {filename}")

def run_classification(exp_name, base_dir, data_dir, feature_dir, train_csv, val_csv, test_csv, results_folder):
    # Determine results folder
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    else:
        print(f"{results_folder} already exists. Removing all files")
        shutil.rmtree(results_folder)
        os.makedirs(results_folder)
    
    print(f"Results will be saved to: {results_folder}")

    # 1. Train
    train_dataset = ShimmerDataset(train_csv, feature_dir)
    label_encoder = train_dataset.label_encoder
    print(f"Classes: {label_encoder.classes_}")
    
    # 2. Val & Test
    val_dataset = ShimmerDataset(val_csv, feature_dir, label_encoder=label_encoder)
    test_dataset = ShimmerDataset(test_csv, feature_dir, label_encoder=label_encoder)
    
    # DataLoaders for PyTorch
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Numpy Arrays for Sklearn/XGBoost
    X_train, y_train = get_numpy_data(train_dataset)
    X_val, y_val = get_numpy_data(val_dataset)
    X_test, y_test = get_numpy_data(test_dataset)

    # Data Statistics
    print("Train data statistics:")
    print(pd.DataFrame(X_train).describe())
    print("Val data statistics:")
    print(pd.DataFrame(X_val).describe())
    print("Test data statistics:")
    print(pd.DataFrame(X_test).describe())

    # Check for nans
    if np.isnan(X_train).any() or np.isnan(X_val).any() or np.isnan(X_test).any():
        print("Xtrain: ", X_train)
        print("Xval: ", X_val)
        print("Xtest: ", X_test)
        raise ValueError("NaN values found in the data")
    
    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")

    # Normalize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- DEFINE MODELS ---
    
    all_results = {}
    metrics_data = []

    def record_metrics(model_name, split_name, y_true, y_pred):
        m = calculate_metrics(y_true, y_pred)
        # Add to nested dict
        if model_name not in all_results:
            all_results[model_name] = {}
        all_results[model_name][split_name] = m
        
        # Add to flat list for DataFrame
        for metric_name, value in m.items():
            metrics_data.append({
                'Model': model_name,
                'Split': split_name,
                'Metric': metric_name,
                'Value': value
            })
        return m
    
    # 1. 1DCNN
    print("\n--- Training 1DCNN ---")
    input_channels = train_dataset[0][0].shape[0] if len(train_dataset) > 0 else 1
    
    cnn_model = Simple1DCNN(num_classes=len(label_encoder.classes_), input_channels=input_channels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
    
    # Train 1DCNN
    num_epochs = 20
    for epoch in range(num_epochs):
        cnn_model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # Filter out -1 labels if they exist (ShimmerDataset returns -1 for unknown classes)
            valid_mask = labels != -1
            if not valid_mask.any():
                continue
            inputs = inputs[valid_mask]
            labels = labels[valid_mask]

            optimizer.zero_grad()
            outputs = cnn_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    # Evaluate 1DCNN
    cnn_train_labels, cnn_train_preds = get_predictions(cnn_model, train_loader, device)
    cnn_val_labels, cnn_val_preds = get_predictions(cnn_model, val_loader, device)
    cnn_test_labels, cnn_test_preds = get_predictions(cnn_model, test_loader, device)
    
    record_metrics('1DCNN', 'Train', cnn_train_labels, cnn_train_preds)
    record_metrics('1DCNN', 'Validation', cnn_val_labels, cnn_val_preds)
    record_metrics('1DCNN', 'Test', cnn_test_labels, cnn_test_preds)

    save_plots(cnn_train_labels, cnn_train_preds, cnn_val_labels, cnn_val_preds, 
               cnn_test_labels, cnn_test_preds, label_encoder, exp_name, '1DCNN', results_folder)


    # 2. SVM
    print("\n--- Training SVM ---")
    svm_model = SVC(kernel='rbf', probability=True) # RBF is standard
    svm_model.fit(X_train, y_train)
    
    svm_train_preds = svm_model.predict(X_train)
    svm_val_preds = svm_model.predict(X_val)
    svm_test_preds = svm_model.predict(X_test)
    
    record_metrics('SVM', 'Train', y_train, svm_train_preds)
    record_metrics('SVM', 'Validation', y_val, svm_val_preds)
    record_metrics('SVM', 'Test', y_test, svm_test_preds)

    save_plots(y_train, svm_train_preds, y_val, svm_val_preds, y_test, svm_test_preds, 
                label_encoder, exp_name, 'SVM', results_folder)

    # 3. XGBoost
    print("\n--- Training XGBoost ---")
    xgb_model = XGBClassifier(eval_metric='mlogloss')
    xgb_model.fit(X_train, y_train)
    
    xgb_train_preds = xgb_model.predict(X_train)
    xgb_val_preds = xgb_model.predict(X_val)
    xgb_test_preds = xgb_model.predict(X_test)
    
    record_metrics('XGBoost', 'Train', y_train, xgb_train_preds)
    record_metrics('XGBoost', 'Validation', y_val, xgb_val_preds)
    record_metrics('XGBoost', 'Test', y_test, xgb_test_preds)

    save_plots(y_train, xgb_train_preds, y_val, xgb_val_preds, y_test, xgb_test_preds, 
            label_encoder, exp_name, 'XGBoost', results_folder)
               

    # --- SAVE ALL METRICS ---
    # 1. JSON
    metrics_path_json = os.path.join(results_folder, 'all_metrics.json')
    with open(metrics_path_json, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nSaved all metrics (JSON) to {metrics_path_json}")

    # 2. CSV DataFrame
    metrics_path_csv = os.path.join(results_folder, 'all_metrics.csv')
    df_metrics = pd.DataFrame(metrics_data)
    
    # Pivot to wide format: Model, Split, Accuracy, Precision, ...
    df_pivot = df_metrics.pivot_table(index=['Model', 'Split'], columns='Metric', values='Value').reset_index()
    df_pivot.to_csv(metrics_path_csv, index=False)
    print(f"Saved all metrics (CSV) to {metrics_path_csv}")
    print("\nMetrics DataFrame:")
    print(df_pivot)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='multi_model_experiment')
    args = parser.parse_args()

    # Configuration
    # Using absolute paths as per the user's setup
    BASE_DIR = r'c:\Users\ismas\Documents\X2Learn\Personalization-Enablers-Training-Tools'
    DATA_DIR = os.path.join(BASE_DIR, 'outputs', 'XRoom', 'shimmer')
    FEATURE_DIR = os.path.join(DATA_DIR, 'eda_features')
    TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
    VAL_CSV = os.path.join(DATA_DIR, 'val.csv')
    TEST_CSV = os.path.join(DATA_DIR, 'test.csv')
    results_folder = os.path.join(base_dir, 'outputs', 'new_classifier_results', args.exp_name)

    run_classification(args.exp_name, BASE_DIR, DATA_DIR, FEATURE_DIR, TRAIN_CSV, VAL_CSV, TEST_CSV, results_folder)

if __name__ == "__main__":
    main()