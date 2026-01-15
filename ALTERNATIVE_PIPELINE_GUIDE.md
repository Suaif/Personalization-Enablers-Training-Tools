
# Alternative classfiers pipeline with feature extraction

1. Prepare the `configuration.json file` for the preprocessing settings
2. Run `run_alternative_pipeline.py --exp_name=[my_experiment_name]`

## Pipeline
1. **Preprocessing**: Same preprocessing as in the original pipeline
2. **Feature extraction**: Extract features from the preprocessed data with `extract_features.py`. It does not work with the `ppg` channel.
3. **Classification**: Uses `alternative_classifiers.py`:
    * **1DCNN**
    * **SVM**
    * **XGBoost**

## Outputs (outputs/XRoom/shimmer/)
* `standardize\`: Folder containing the `.npy` files for each sample for supervised learning.
* `ssl_standardize\`: Folder containing the `.npy` files for each sample for SSL.
* `eda_features\`: Folder containing the `.npy` files for each sample extracted features.
* `ssl_[split].csv`: Files containing the train/val/test split of the samples for SSL.
* `[split].csv`: Files containing the train/val/test split of the samples for supervised learning.

## Experiment Results (on outputs/alternative_classifier_results/EXPERIMENT_ID/)
* `configuration.json`: The configuration file used for the experiment.
* `all_metrics.json` and `all_metrics.csv`: Performance for each split and model.
* `Split csv files`: Files containing the train/val/test split of the samples
* `Split csv predictions`: The predictions for the train/val/test split.
* `prediction_histogram_[model].png`: The distribution of the predictions for each split and model.
* `Performance per game per split`: The performance metrics in `.json` and `.png` format.

