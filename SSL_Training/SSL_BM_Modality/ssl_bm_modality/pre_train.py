import os
import json

import torch
from pytorch_lightning import Trainer

from conf import (
    CUSTOM_SETTINGS,
    MODALITY,
    MODALITY_FOLDER,
    EXPERIMENT_ID,
    COMPONENT_OUTPUT_FOLDER,
    OUTPUTS_FOLDER,
    EXPERIMENT_RESULTS_FOLDER
)
from ssl_dataset import SSLDataModule
from callbacks.setup_callbacks import setup_callbacks
from utils.init_utils import (init_augmentations, init_transforms,
                              setup_ssl_model, init_encoder)


def run_pre_training():
    """
    Pre-train encoder and save checkpoints.
    """
    print(CUSTOM_SETTINGS)

    if (
        'get_ssl' in CUSTOM_SETTINGS[MODALITY]['pre_processing_config'] and
        CUSTOM_SETTINGS[MODALITY]['pre_processing_config']['get_ssl']
    ):
        prefix = "ssl_"
    else:
        prefix = ""

    splith_paths = {'train': f"{prefix}train.csv", 'val': f"{prefix}val.csv", 'test': f"{prefix}test.csv"}

    train_transforms = None
    test_transforms = None
    if 'transforms' in CUSTOM_SETTINGS[MODALITY].keys():
        train_transforms, test_transforms = init_transforms(CUSTOM_SETTINGS[MODALITY]['transforms'])

    if 'augmentations' in CUSTOM_SETTINGS[MODALITY].keys():
        augmentations = init_augmentations(CUSTOM_SETTINGS[MODALITY]['augmentations'])
        print(augmentations)

    datamodule = SSLDataModule(
        path=MODALITY_FOLDER,
        input_type=prefix + CUSTOM_SETTINGS[MODALITY]['ssl_config']['input_type'],
        batch_size=CUSTOM_SETTINGS[MODALITY]['ssl_config']['batch_size'],
        split=splith_paths,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        n_views=2,
        num_workers=0,
        augmentations=augmentations
    )

    # initialise encoder
    encoder = init_encoder(
        CUSTOM_SETTINGS[MODALITY]["encoder_config"],
        CUSTOM_SETTINGS[MODALITY]['encoder_config']['pretrained'] if (
            "pretrained" in CUSTOM_SETTINGS[MODALITY]['encoder_config'].keys()
            ) else None
    )
    print(encoder)

    ssl_model = setup_ssl_model(encoder, model_cfg=CUSTOM_SETTINGS[MODALITY]['ssl_config'])
    print(ssl_model)

    checkpoint_filename = (
        f"{EXPERIMENT_ID}_"
        f"{CUSTOM_SETTINGS['dataset_config']['dataset_name']}_"
        f"{MODALITY}_"
        f"{CUSTOM_SETTINGS[MODALITY]['ssl_config']['input_type']}_"
        f"{CUSTOM_SETTINGS[MODALITY]['encoder_config']['class_name']}"
    )

    ssl_checkpoint = checkpoint_filename + "ssl_model"

    # by default lightning does not overwrite checkpoints, but rather creates different versions (v1, v2, etc.)
    # for the sample checkpoint_filename. Thus, in order to enable overwriting, we delete checkpoint if it exists.
    if os.path.exists(os.path.join(COMPONENT_OUTPUT_FOLDER, ssl_checkpoint + '.ckpt')):
        os.remove(os.path.join(COMPONENT_OUTPUT_FOLDER, ssl_checkpoint + '.ckpt'))

    # initialize callbacks
    callbacks = setup_callbacks(
        early_stopping_metric="val_loss",
        no_ckpt=False,
        patience=100,
        dirpath=COMPONENT_OUTPUT_FOLDER,
        monitor="val_loss",
        save_last=True,
        save_top_k=1,
        checkpoint_filename=ssl_checkpoint
    )

    # initialize Pytorch-Lightning Training
    trainer = Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        deterministic=True,
        default_root_dir=COMPONENT_OUTPUT_FOLDER,
        callbacks=callbacks,
        max_epochs=CUSTOM_SETTINGS[MODALITY]['ssl_config']['epochs'],
        log_every_n_steps=10,
    )

    trainer.fit(ssl_model, datamodule)
    metrics = trainer.test(ssl_model, datamodule, ckpt_path='best')
    print(metrics)
    
    # Convert tensors to floats for JSON serialization
    if isinstance(metrics, list) and len(metrics) > 0:
        metrics = metrics[0]
    metrics_to_save = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}
    
    output_dir = EXPERIMENT_RESULTS_FOLDER
    if os.path.exists(output_dir):
        print(f"Experiment folder {output_dir} already exists. Overwriting...")
    else:
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f"{EXPERIMENT_ID}_test_metrics_ssl.json")
    with open(output_file, "w") as f:
        json.dump(metrics_to_save, f, indent=4)
    print(f"Test metrics saved to {output_file}")

    # if save_last_encoder, the weights of encoder will be taken and saved from the last epoch of ssl pre-training
    if "save_last_encoder" in CUSTOM_SETTINGS[MODALITY]["ssl_config"] and CUSTOM_SETTINGS[MODALITY]["ssl_config"]["save_last_encoder"]:
        ssl_model = ssl_model.__class__.load_from_checkpoint(
            os.path.join(EXPERIMENT_RESULTS_FOLDER, checkpoint_filename + "_last.ckpt"),
            encoder=encoder
        )

    torch.save(
        ssl_model.encoder.state_dict(),
        os.path.join(
            COMPONENT_OUTPUT_FOLDER,
            f'{checkpoint_filename}_encoder.pt'
        )
    )
    torch.save(
        ssl_model.encoder.state_dict(),
        os.path.join(
            EXPERIMENT_RESULTS_FOLDER,
            f'{checkpoint_filename}_encoder.pt'
        )
    )


if __name__ == '__main__':
    run_pre_training()
