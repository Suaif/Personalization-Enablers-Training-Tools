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
import shutil

from callbacks.setup_callbacks import setup_callbacks
from classification_model import SupervisedModel
from classifiers.linear import LinearClassifier
from conf import CUSTOM_SETTINGS, MODALITY, MODALITY_FOLDER, COMPONENT_OUTPUT_FOLDER, EXPERIMENT_ID, LABEL_TO_ID, EXPERIMENT_RESULTS_FOLDER, INTER_SUBJECT_SPLIT
from supervised_dataset import SupervisedDataModule, SupervisedTorchDataset
from utils.init_utils import (init_augmentations, init_transforms, init_encoder)
from utils.performance_by_game import get_performance_by_game
from utils.inter_subject import apply_inter_subject_split, get_performance_by_subject
from utils.predictions import generate_all_predictions_and_plot

def run_supervised_training():
    print(json.dumps(CUSTOM_SETTINGS, indent=4))
    
    split_paths = {'train': "train.csv", 'val': "val.csv", 'test': "test.csv"}
    
    if INTER_SUBJECT_SPLIT and CUSTOM_SETTINGS[MODALITY]['pre_processing_config']['split_by']=="subject":
        split_paths = apply_inter_subject_split(MODALITY_FOLDER, split_paths)

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
        split=split_paths,
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

    generate_all_predictions_and_plot(
        model_class=SupervisedModel,
        checkpoint_path=best_model_path,
        datamodule=datamodule,
        dataset_class=SupervisedTorchDataset,
        output_dir=EXPERIMENT_RESULTS_FOLDER,
        experiment_id=EXPERIMENT_ID,
        encoder=encoder,
        classifier=classifier
    )

    # Compute performance by game
    for split_name, split_path in split_paths.items():
        get_performance_by_game(
            data_dir=MODALITY_FOLDER,
            exp_dir=EXPERIMENT_RESULTS_FOLDER,
            split_df=split_path,
            split_predictions=split_name,
            show_plot=False
        )

        # Copy train, val and test.csv to exp_dir
        shutil.copyfile(f"{MODALITY_FOLDER}/{split_path}", f"{EXPERIMENT_RESULTS_FOLDER}/{split_path}")

    # Compute performance by subject
    get_performance_by_subject(
        data_dir=MODALITY_FOLDER,
        exp_dir=EXPERIMENT_RESULTS_FOLDER,
        split_paths=split_paths,
        show_plot=False
    )
    
    print(f"Experiment {EXPERIMENT_ID} finished, results saved to: {EXPERIMENT_RESULTS_FOLDER}")
    
if __name__ == '__main__':
    run_supervised_training()
