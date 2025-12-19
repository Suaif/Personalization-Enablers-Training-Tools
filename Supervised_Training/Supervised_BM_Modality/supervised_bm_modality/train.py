# Python code here
import json
import os

import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from callbacks.setup_callbacks import setup_callbacks
from classification_model import SupervisedModel
from classifiers.linear import LinearClassifier
from conf import CUSTOM_SETTINGS, MODALITY, MODALITY_FOLDER, COMPONENT_OUTPUT_FOLDER, EXPERIMENT_ID, LABEL_TO_ID, EXPERIMENT_RESULTS_FOLDER
from supervised_dataset import SupervisedDataModule, SupervisedTorchDataset
from utils.init_utils import (init_augmentations, init_transforms, init_encoder)


def run_supervised_training():
    print(json.dumps(CUSTOM_SETTINGS, indent=4))
    splith_paths = {'train': "train.csv", 'val': "val.csv", 'test': "test.csv"}

    train_transforms = {}
    test_transforms = {}

    if 'transforms' in CUSTOM_SETTINGS[MODALITY].keys():
        train_transforms, test_transforms = init_transforms(CUSTOM_SETTINGS[MODALITY]['transforms'])

    # for now, don't use augmentations during supervised training
    augmentations = None
    if (CUSTOM_SETTINGS[MODALITY]['sup_config']['use_augmentations_in_sup']) and (
            'augmentations' in CUSTOM_SETTINGS[MODALITY].keys()):
        augmentations = init_augmentations(CUSTOM_SETTINGS[MODALITY]['augmentations'])
        print("Augmentations loaded successfully")
    else:
        print("No augmentations loaded")

    label_mapping = LABEL_TO_ID[CUSTOM_SETTINGS['dataset_config']['dataset_name']]

    datamodule = SupervisedDataModule(
        path=MODALITY_FOLDER,
        input_type=CUSTOM_SETTINGS[MODALITY]['sup_config']['input_type'],
        batch_size=CUSTOM_SETTINGS[MODALITY]['sup_config']['batch_size'],
        split=splith_paths,
        label_mapping=label_mapping,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        augmentations=augmentations,
    )

    ckpt_name = (
        f"{EXPERIMENT_ID}_"
        f"{CUSTOM_SETTINGS['dataset_config']['dataset_name']}_"
        f"{MODALITY}_"
        f"{CUSTOM_SETTINGS[MODALITY]['sup_config']['input_type']}_"
        f"{CUSTOM_SETTINGS[MODALITY]['encoder_config']['class_name']}"
    )

    if "pretrained_path" in CUSTOM_SETTINGS[MODALITY]['encoder_config'].keys():
        ckpt_path = CUSTOM_SETTINGS[MODALITY]['encoder_config']['pretrained_path']
    elif (
        "pretrained_same_experiment" in CUSTOM_SETTINGS[MODALITY]['encoder_config'].keys() and
        CUSTOM_SETTINGS[MODALITY]['encoder_config']["pretrained_same_experiment"]
    ):
        ckpt_path = os.path.join(MODALITY_FOLDER, "ssl_training", f"{ckpt_name}_encoder.pt")
    else:
        ckpt_path = None

    # initialise encoder
    encoder = init_encoder(
        model_cfg=CUSTOM_SETTINGS[MODALITY]["encoder_config"],
        ckpt_path=ckpt_path
    )

    # add classification head to encoder
    num_classes = CUSTOM_SETTINGS['dataset_config'].get("number_of_labels", 3)
    if isinstance(num_classes, dict):
        num_classes = num_classes.get(MODALITY, 3)
    classifier = LinearClassifier(encoder.out_size, num_classes)
    model = SupervisedModel(encoder=encoder, classifier=classifier, **CUSTOM_SETTINGS[MODALITY]['sup_config']['kwargs'])

    checkpoint_filename = f'{ckpt_name}_model'

    # by default lightning does not overwrite checkpoints, but rather creates different versions (v1, v2, etc.)
    # for the sample checkpoint_filename. Thus, in order to enable overwriting, we delete checkpoint if it exists.
    if os.path.exists(os.path.join(EXPERIMENT_RESULTS_FOLDER, checkpoint_filename + '.ckpt')):
        os.remove(os.path.join(EXPERIMENT_RESULTS_FOLDER, checkpoint_filename + '.ckpt'))

    # initialize callbacks
    callbacks = setup_callbacks(
        early_stopping_metric="val_loss",
        no_ckpt=False,
        num_classes=num_classes,
        patience=50,
        dirpath=EXPERIMENT_RESULTS_FOLDER,
        monitor=CUSTOM_SETTINGS[MODALITY]['sup_config']['monitor'] if 'monitor' in CUSTOM_SETTINGS[MODALITY]['sup_config'] else "val_loss",
        checkpoint_filename=checkpoint_filename
    )

    # initialize Pytorch-Lightning Trainer
    trainer = Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        deterministic=True,
        default_root_dir=os.path.join(EXPERIMENT_RESULTS_FOLDER),
        callbacks=callbacks,
        max_epochs=CUSTOM_SETTINGS[MODALITY]['sup_config']['epochs']
    )

    if os.path.exists(EXPERIMENT_RESULTS_FOLDER):
        print(f"Experiment folder {EXPERIMENT_RESULTS_FOLDER} already exists. Overwriting...")
    else:
        os.makedirs(EXPERIMENT_RESULTS_FOLDER)
    
    # train model and report metrics
    # the model checkpoints (best and last if provided) will be saved in
    # /COMPONENT_OUTPUT_FOLDER/{EXPERIMENT_ID}_model_lightning.ckpt
    trainer.fit(model, datamodule)

    # evaluate model on the test set, by default the best model
    trainer.test(model, datamodule, ckpt_path="best")
    
    # save weights of the classifier independently for future use with SSL features
    torch.save(
        classifier.state_dict(),
        os.path.join(EXPERIMENT_RESULTS_FOLDER, f'{ckpt_name}_classifier.pt')
    )
    # Save a copy of the configuration file
    with open(os.path.join(EXPERIMENT_RESULTS_FOLDER, 'configuration.json'), 'w') as f:
        json.dump(CUSTOM_SETTINGS, f, indent=4)
    
    # Generate and save predictions to CSV files using the best model
    print("\nGenerating predictions using the best model...")
    # Load the best model checkpoint
    best_model_path = os.path.join(EXPERIMENT_RESULTS_FOLDER, f'{checkpoint_filename}.ckpt')
    model = SupervisedModel.load_from_checkpoint(best_model_path, encoder=encoder, classifier=classifier)
    model.eval()
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Manually create dataloaders for prediction (setup was only for training/testing stages)
    train_dataset = SupervisedTorchDataset(
        datamodule.path,
        datamodule.input_type,
        datamodule.split['train'],
        label_mapping=datamodule.label_mapping,
        transforms=datamodule.train_transforms,
        augmentations=None  # No augmentations for prediction
    )
    
    val_dataset = SupervisedTorchDataset(
        datamodule.path,
        datamodule.input_type,
        datamodule.split['val'],
        label_mapping=datamodule.label_mapping,
        transforms=datamodule.test_transforms,
        augmentations=None
    )
    
    test_dataset = SupervisedTorchDataset(
        datamodule.path,
        datamodule.input_type,
        datamodule.split['test'],
        label_mapping=datamodule.label_mapping,
        transforms=datamodule.test_transforms,
        augmentations=None
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=datamodule.batch_size,
        shuffle=False,
        num_workers=0  # Use 0 workers for simplicity
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=datamodule.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=datamodule.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    def generate_predictions(model, dataloader):
        """Generate predictions for a given dataloader"""
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                X, Y = batch[0], batch[1]
                X = X.to(model.device)
                Y = Y.to(model.device)
                
                out = model(X)
                preds = torch.argmax(out, dim=1)
                
                all_predictions.append(preds.cpu())
                all_labels.append(Y.cpu())
        
        # Concatenate all batches
        predictions = torch.cat(all_predictions)
        labels = torch.cat(all_labels)
        
        return [{"preds": predictions, "labels": labels}]
    
    # Generate predictions for train set
    train_predictions = generate_predictions(model, train_loader)
    SupervisedModel.save_predictions_csv(
        train_predictions,
        os.path.join(EXPERIMENT_RESULTS_FOLDER, f'train_predictions.csv'),
        split_name='train'
    )
    
    # Generate predictions for validation set
    val_predictions = generate_predictions(model, val_loader)
    SupervisedModel.save_predictions_csv(
        val_predictions,
        os.path.join(EXPERIMENT_RESULTS_FOLDER, f'val_predictions.csv'),
        split_name='val'
    )
    
    # Generate predictions for test set
    test_predictions = generate_predictions(model, test_loader)
    SupervisedModel.save_predictions_csv(
        test_predictions,
        os.path.join(EXPERIMENT_RESULTS_FOLDER, f'test_predictions.csv'),
        split_name='test'
    )

    # --- Plotting Prediction Histograms ---
    
    # Invert label mapping
    inv_label_mapping = {v: k for k, v in label_mapping.items()}
    class_order = ['BORED', 'ENGAGED', 'FRUSTRATED'] # Enforce specific order if desired, or use sorted(inv_label_mapping.values())
    
    # Helper to extract data
    def extract_data(pred_list):
        preds = pred_list[0]['preds'].cpu().numpy()
        labels = pred_list[0]['labels'].cpu().numpy()
        return preds, labels

    train_preds_np, train_labels_np = extract_data(train_predictions)
    val_preds_np, val_labels_np = extract_data(val_predictions)
    test_preds_np, test_labels_np = extract_data(test_predictions)

    # Dark Mode Toggle
    DARK_MODE = False
    if DARK_MODE:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    def plot_comparison(labels, preds, ax, title):
        # Convert to string labels
        pred_names = [inv_label_mapping.get(p, "UNKNOWN") for p in preds]
        true_names = [inv_label_mapping.get(l, "UNKNOWN") for l in labels]
        
        df_pred = pd.DataFrame({'Class': pred_names, 'Type': 'Predictions'})
        df_true = pd.DataFrame({'Class': true_names, 'Type': 'True Labels'})
        combined_df = pd.concat([df_true, df_pred], ignore_index=True)
        
        sns.countplot(data=combined_df, x='Class', hue='Type', ax=ax, order=class_order)
        ax.set_title(title)
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)

    plot_comparison(train_labels_np, train_preds_np, axes[0], 'Train Set')
    plot_comparison(val_labels_np, val_preds_np, axes[1], 'Validation Set')
    plot_comparison(test_labels_np, test_preds_np, axes[2], 'Test Set')
    
    fig.suptitle(f"Predictions Histogram \n {EXPERIMENT_ID}")
    plt.tight_layout()
    output_plot_path = os.path.join(EXPERIMENT_RESULTS_FOLDER, 'predictions_histogram.png')
    plt.savefig(output_plot_path)
    
    print(f"Experiment {EXPERIMENT_ID} finished, results saved to: {EXPERIMENT_RESULTS_FOLDER}")
    
if __name__ == '__main__':
    run_supervised_training()
