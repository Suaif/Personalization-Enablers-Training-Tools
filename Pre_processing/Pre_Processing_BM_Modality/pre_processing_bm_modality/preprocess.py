import os
import shutil
import pandas as pd

from conf import (
    CUSTOM_SETTINGS,
    DATA_PATH,
    MODALITY,
    MODALITY_FOLDER,
    EXPERIMENT_RESULTS_FOLDER
)

from preprocessing_utils import process_dataset


def preprocess():
    print('Pre Processing BM Modality')

    all_subject_dirs = os.listdir(DATA_PATH)
    print(f"Found a total of {len(all_subject_dirs)} under {DATA_PATH}.")

    (train_split, val_split, test_split, 
     stats, ssl_train_split, ssl_val_split, ssl_test_split) = process_dataset(
        DATA_PATH,
        all_subject_dirs,
        CUSTOM_SETTINGS[MODALITY]["pre_processing_config"],
        MODALITY_FOLDER,
        seq_len=CUSTOM_SETTINGS[MODALITY]["pre_processing_config"].get("seq_len", 5),  # in seconds
        overlap=CUSTOM_SETTINGS[MODALITY]["pre_processing_config"].get("overlap", 0.),  # between 0 and 1 (proportion)
        frequency=CUSTOM_SETTINGS[MODALITY]["pre_processing_config"].get("frequency", 10),  # in Hz
        resample_freq=CUSTOM_SETTINGS[MODALITY]["pre_processing_config"].get(
            "resample_freq",
            CUSTOM_SETTINGS[MODALITY]["pre_processing_config"].get("frequency", 10)
        ),  # resampling if needed
        use_sensors=CUSTOM_SETTINGS[MODALITY]["pre_processing_config"].get("use_sensors", ["gsr"]),
        borders=CUSTOM_SETTINGS[MODALITY]["pre_processing_config"].get("borders", None)
    )

    if stats is not None:
        stats_df = (
            pd.DataFrame(stats)
            .sort_values(by=["subject", "session"])
        )

        stats_df.to_csv(os.path.join(MODALITY_FOLDER, "stats_biomeasurements.csv"), index=None)

    print('Writing CSV files containing the splits to storage')
    # Save filtered datasets (without 0.5 values)
    train_df = pd.DataFrame.from_dict(train_split)
    val_df = pd.DataFrame.from_dict(val_split)
    test_df = pd.DataFrame.from_dict(test_split)
    
    train_df.to_csv(
        os.path.join(
            MODALITY_FOLDER,
            'train.csv'
        )
    )
    val_df.to_csv(
        os.path.join(
            MODALITY_FOLDER,
            'val.csv'
        )
    )
    test_df.to_csv(
        os.path.join(
            MODALITY_FOLDER,
            'test.csv'
        )
    )

    # Print detailed statistics
    print('\n' + '='*80)
    print('PREPROCESSING SUMMARY')
    print('='*80)
    
    print('\n📊 SUPERVISED DATASETS (Labeled data):')
    print('-' * 80)
    print(f'    train.csv:      {len(train_df):5d} samples')
    print(f'    val.csv:        {len(val_df):5d} samples')
    print(f'    test.csv:       {len(test_df):5d} samples')
    print(f'    TOTAL:          {len(train_df) + len(val_df) + len(test_df):5d} samples')

    if (
        "get_ssl" in CUSTOM_SETTINGS[MODALITY]["pre_processing_config"] and
        CUSTOM_SETTINGS[MODALITY]["pre_processing_config"]["get_ssl"]
    ):
        print('Writing CSV files containing the SSL splits to storage')
        ssl_train_df = pd.DataFrame.from_dict(ssl_train_split)
        ssl_val_df = pd.DataFrame.from_dict(ssl_val_split)
        ssl_test_df = pd.DataFrame.from_dict(ssl_test_split)
        
        ssl_train_df.to_csv(
            os.path.join(
                MODALITY_FOLDER,
                'ssl_train.csv'
            )
        )
        ssl_val_df.to_csv(
            os.path.join(
                MODALITY_FOLDER,
                'ssl_val.csv'
            )
        )
        ssl_test_df.to_csv(
            os.path.join(
                MODALITY_FOLDER,
                'ssl_test.csv'
            )
        )

        print('\n📊 SSL DATASETS (Labeled + Unlabeled data):')
        print('-' * 80)

        print(f'    ssl_train.csv:  {len(ssl_train_df):5d} samples')
        print(f'    ssl_val.csv:    {len(ssl_val_df):5d} samples')
        print(f'    ssl_test.csv:   {len(ssl_test_df):5d} samples')
        print(f'    TOTAL:          {len(ssl_train_df) + len(ssl_val_df) + len(ssl_test_df):5d} samples')
        
        print(f'\n  Comparison (SSL vs Supervised):')
        print(f'    train (SSL - Sup):       {len(ssl_train_df) - len(train_df):5d} unlabeled samples')
        print(f'    val   (SSL - Sup):       {len(ssl_val_df) - len(val_df):5d} unlabeled samples')
        print(f'    test  (SSL - Sup):       {len(ssl_test_df) - len(test_df):5d} unlabeled samples')
    
    print('\n' + '='*80)

    # Clean EXPERIMENT_RESULTS_FOLDER
    if os.path.exists(EXPERIMENT_RESULTS_FOLDER):
        print(f"Experiment folder {EXPERIMENT_RESULTS_FOLDER} already exists. Overwriting...")
        shutil.rmtree(EXPERIMENT_RESULTS_FOLDER)
    
    os.makedirs(EXPERIMENT_RESULTS_FOLDER)

if __name__ == '__main__':
    preprocess()
