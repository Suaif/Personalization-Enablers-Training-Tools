import sys
import os
import json
import pandas as pd
import argparse
import re
import shutil

# Add path for preprocessing. Assuming the script runs from 'Personalization-Enablers-Training-Tools' root
sys.path.append(os.path.join(os.getcwd(), 'Pre_processing', 'Pre_Processing_BM_Modality', 'pre_processing_bm_modality'))

from preprocessing_utils import process_dataset

import extract_features
import alternative_classifiers

def run_preprocessing(data_path, modality_folder, pp_config):
    """
    Run preprocessing on the dataset -> Generates train, val, test splits.
    """
    print(f"--- 1. PREPROCESSING ---")

    all_subject_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    print(f"Found {len(all_subject_dirs)} subjects in {data_path}")
   
    print("Running process_dataset...")
    (train_split, val_split, test_split, stats, ssl_train, ssl_val, ssl_test) = process_dataset(
        full_dataset_path=data_path,
        all_subjects_dirs=all_subject_dirs,
        pre_processing_cfg=pp_config,
        outputs_folder=modality_folder,
        seq_len=pp_config.get("seq_len", 5),
        overlap=pp_config.get("overlap", 0.),
        frequency=pp_config.get("frequency", 10),
        resample_freq=pp_config.get("resample_freq", pp_config.get("frequency", 10)),
        use_sensors=pp_config.get("use_sensors", ["gsr"]),
        borders=pp_config.get("borders", None)
    )
    
    # Save CSVs
    print("Saving Split CSVs...")
    os.makedirs(modality_folder, exist_ok=True)
    
    train_df = pd.DataFrame.from_dict(train_split)
    val_df = pd.DataFrame.from_dict(val_split)
    test_df = pd.DataFrame.from_dict(test_split)

    train_df.to_csv(os.path.join(modality_folder, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(modality_folder, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(modality_folder, 'test.csv'), index=False)
    
    print(f"Saved: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples.")
    
    if stats:
        stats_df = pd.DataFrame(stats).sort_values(by=["subject", "session"])
        stats_df.to_csv(os.path.join(modality_folder, "stats_biomeasurements.csv"), index=False)


def run_feature_extraction(modality_folder, pp_config):
    """
    Run feature extraction on the preprocessed data.
    """
    print(f"\n--- 2. FEATURE EXTRACTION ---")
    
    process_type = pp_config.get('process', 'standardize')
    npy_data_dir = os.path.join(modality_folder, process_type)
    feature_output_dir = os.path.join(modality_folder, 'eda_features')
    
    print(f"Reading .npy files from: {npy_data_dir}")
    print(f"Saving features to: {feature_output_dir}")
    
    extract_features.generate_and_save_features(npy_data_dir, feature_output_dir, 
    sampling_rate=pp_config.get("frequency", 10), 
    signals=pp_config.get("use_sensors", ["gsr"]))
    return feature_output_dir

def run_classifier_training(exp_name, base_dir, modality_folder, feature_output_dir, results_folder):
    """
    Run classifier training on the feature extracted data.
    """
    print(f"\n--- 3. CLASSIFICATION ---")
    
    train_csv_path = os.path.join(modality_folder, 'train.csv')
    val_csv_path = os.path.join(modality_folder, 'val.csv')
    test_csv_path = os.path.join(modality_folder, 'test.csv')
    
    alternative_classifiers.run_classification(
        exp_name=exp_name,
        base_dir=base_dir,
        data_dir=modality_folder,
        feature_dir=feature_output_dir,
        train_csv=train_csv_path,
        val_csv=val_csv_path,
        test_csv=test_csv_path,
        results_folder=results_folder
    )
    
    # Copy dataset CSVs to results folder
    shutil.copy(train_csv_path, os.path.join(results_folder, 'train.csv'))
    shutil.copy(val_csv_path, os.path.join(results_folder, 'val.csv'))
    shutil.copy(test_csv_path, os.path.join(results_folder, 'test.csv'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='automated_experiment_1', help='Name of the experiment')
    args = parser.parse_args()
    
    print(f"Starting Experiment: {args.exp_name}")
    
    # Paths
    base_dir = os.getcwd() # Should be Personalization-Enablers-Training-Tools
    
    # Load Configuration
    config_path = 'configuration.json'

    def strip_json_comments(text):
        """Strip C-style comments from JSON text."""
        pattern = r'//.*?$|/\*.*?\*/'
        return re.sub(pattern, '', text, flags=re.DOTALL | re.MULTILINE)

    with open(config_path, 'r') as f:
        content = f.read()
        cleaned_content = strip_json_comments(content)
        config = json.loads(cleaned_content)
    
    # Load configs
    dataset_config = config.get("dataset_config", {})
    dataset_name = dataset_config.get("dataset_name", "XRoom")
    modality = dataset_config.get("modality", "shimmer")
    pp_config = config[modality].get("pre_processing_config", {})

    # Paths
    datasets_folder = os.path.join(base_dir, 'datasets')
    data_path = os.path.join(datasets_folder, dataset_name)
    outputs_folder = os.path.join(base_dir, 'outputs')
    modality_folder = os.path.join(outputs_folder, dataset_name, modality)
    feature_output_dir = os.path.join(modality_folder, 'eda_features')
    results_folder = os.path.join(base_dir, 'outputs', 'alternative_classifiers_results', args.exp_name)
    
    # --- PIPELINE STEPS ---
    
    # 1. Preprocessing
    run_preprocessing(data_path, modality_folder, pp_config)
    
    # 2. Feature Extraction
    run_feature_extraction(modality_folder, pp_config)
    
    # 3. Classification
    run_classifier_training(args.exp_name, base_dir, modality_folder, feature_output_dir, results_folder)

    # Save configuration file in the results folder
    config_path = os.path.join(results_folder, 'configuration.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print("\n--- PIPELINE COMPLETE ---")

if __name__ == "__main__":
    main()
