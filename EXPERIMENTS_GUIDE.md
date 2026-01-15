
# Running a single experiment

1. Prepare the `configuration.json file` and update the `EXPERIMENT_ID` in the `.env` file.
2. Run `run_all_dockers-bm.sh` or `run_all_dockers-bm-gpu.sh`.

## New configuration settings
* `filter_05`: If true, remove from the train/val/test splits the 0.5 and 0.50001 default values
* `stratify`: If true, assure that each split contains all labels
* `inter_subject`: If true and `split_by="subject"`, it selectes one subject from the test split and moves 50% of the data to the train split. `train_intersubject.csv` and `test_intersubject.csv` will be created with the new splits.

## Outputs (outputs/XRoom/shimmer/)
* `standardize\`: Folder containing the `.npy` files for each sample for supervised learning.
* `ssl_standardize\`: Folder containing the `.npy` files for each sample for SSL.
* `ssl_[split].csv`: Files containing the train/val/test split of the samples for SSL.
* `[split].csv`: Files containing the train/val/test split of the samples for supervised learning.

## Experiment Results (on outputs/individual_experiments/EXPERIMENT_ID/)
* `configuration.json`: The configuration file used for the experiment.
* `_model.ckpt` and `_classifier.pt`: Encoder and classifier models.
* `_test_metrics_supervised.json`: The metrics for the test set.
* `Split csv files`: Files containing the train/val/test split of the samples
* `Split csv predictions`: The predictions for the train/val/test split.
* `prediction_histogram.png`: The distribution of the predictions for each split.
* `Performance per game per split`: The performance metrics in `.json` and `.png` format.
* `Performance per subject per split`: The performance metrics in `.json` and `.png` format.

# Running a list of experiments with ClearML logging
## Configuration

### Configure ClearML
```bash
clearml-init
# Enter credentials from app.clear.ml
```

### Create Experiment Config File

List of `.json` configurations with the parameters to uptdate from the original configuration file

All single experiments will be stored in /exp_group/experiment_id/

```json
[
    {
        "experiment_id": "experiment_1",
        "experiment_group": "my_experiments",
        "description": "",
        "config_updates": {
            "dataset_config.number_of_labels": 2,
            "shimmer.pre_processing_config.split_by": "subject",
            "shimmer.pre_processing_config.borders": [
                0.5,
                0.5
            ]
        }
    },
    {
        "experiment_id": "experiment_2",
        "experiment_group": "my_experiments",
        "description": "",
        "config_updates": {
            "dataset_config.number_of_labels": 2,
            "shimmer.pre_processing_config.split_by": "subject",
            "shimmer.pre_processing_config.borders": [
                0.5,
                0.5
            ]
        }
    }
]
```

### ClearML Dashboard

Each experiment automatically logs:
- **Scalars**: `ssl/*` and `supervised/*` metrics
- **Artifacts**: Config files, predictions
- **Console**: Training logs
- **Configuration**: Hyperparameters
- **Status**: completed/failed

---

## Usage


```bash

python run_experiments.py --config experiments_config.json

## Without sleaning the shimmer folder
python run_experiments.py --config experiments_config.json --no-clean-shimmer

```
