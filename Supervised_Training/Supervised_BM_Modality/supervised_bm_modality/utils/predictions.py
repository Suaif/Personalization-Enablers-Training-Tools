import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader


def generate_predictions_for_split(model, dataloader):
    """Generate predictions for a given dataloader
    
    Args:
        model: The trained model
        dataloader: DataLoader for the split
        
    Returns:
        List containing dict with 'preds' and 'labels' tensors
    """
    all_predictions = []
    all_labels = []
    
    model.eval()
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


def create_dataloader(datamodule, split_name, dataset_class):
    """Create a dataloader for a specific split
    
    Args:
        datamodule: The data module containing dataset configuration
        split_name: One of 'train', 'val', or 'test'
        dataset_class: The dataset class to instantiate
        
    Returns:
        DataLoader for the specified split
    """
    # Use appropriate transforms based on split
    transforms = datamodule.train_transforms if split_name == 'train' else datamodule.test_transforms
    
    dataset = dataset_class(
        datamodule.path,
        datamodule.input_type,
        datamodule.split[split_name],
        label_mapping=datamodule.label_mapping,
        transforms=transforms,
        augmentations=None
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=datamodule.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return dataloader


def plot_predictions_histogram(train_predictions, val_predictions, test_predictions, 
                                label_mapping, experiment_id, output_path):
    """Plot histogram comparing predictions vs true labels for all splits
    
    Args:
        train_predictions: Predictions for train set
        val_predictions: Predictions for validation set
        test_predictions: Predictions for test set
        label_mapping: Dictionary mapping label names to indices
        experiment_id: Experiment identifier for plot title
        output_path: Path to save the plot
    """
    # Invert label mapping
    inv_label_mapping = {v: k for k, v in label_mapping.items()}
    class_order = ['BORED', 'ENGAGED', 'FRUSTRATED']
    
    # Helper to extract data
    def extract_data(pred_list):
        preds = pred_list[0]['preds'].cpu().numpy()
        labels = pred_list[0]['labels'].cpu().numpy()
        return preds, labels

    train_preds_np, train_labels_np = extract_data(train_predictions)
    val_preds_np, val_labels_np = extract_data(val_predictions)
    test_preds_np, test_labels_np = extract_data(test_predictions)

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
    
    fig.suptitle(f"Predictions Histogram\n{experiment_id}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generate_all_predictions_and_plot(model_class, checkpoint_path, datamodule, dataset_class, 
                             output_dir, experiment_id, encoder=None, classifier=None):
    """Generate predictions for all splits and save to CSV files
    
    Args:
        model_class: The model class (e.g., SupervisedModel)
        checkpoint_path: Path to the model checkpoint
        datamodule: DataModule containing dataset configuration
        dataset_class: Dataset class to use for creating dataloaders
        output_dir: Directory to save prediction CSVs and plots
        experiment_id: Experiment identifier for plot titles
        encoder: Encoder to pass to model (if required)
        classifier: Classifier to pass to model (if required)
        
    Returns:
        Tuple of (train_predictions, val_predictions, test_predictions)
    """
    print("\nGenerating predictions using the best model...")
    
    # Load the best model checkpoint
    model = model_class.load_from_checkpoint(
        checkpoint_path, 
        encoder=encoder, 
        classifier=classifier
    )
    model.eval()
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataloaders for all splits
    train_loader = create_dataloader(datamodule, 'train', dataset_class)
    val_loader = create_dataloader(datamodule, 'val', dataset_class)
    test_loader = create_dataloader(datamodule, 'test', dataset_class)
    
    # Generate predictions for each split
    print("Generating train predictions...")
    train_predictions = generate_predictions_for_split(model, train_loader)
    model_class.save_predictions_csv(
        train_predictions,
        os.path.join(output_dir, 'train_predictions_new.csv'),
        split_name='train'
    )
    
    print("Generating validation predictions...")
    val_predictions = generate_predictions_for_split(model, val_loader)
    model_class.save_predictions_csv(
        val_predictions,
        os.path.join(output_dir, 'val_predictions_new.csv'),
        split_name='val'
    )
    
    print("Generating test predictions...")
    test_predictions = generate_predictions_for_split(model, test_loader)
    model_class.save_predictions_csv(
        test_predictions,
        os.path.join(output_dir, 'test_predictions_new.csv'),
        split_name='test'
    )
    
    # Plot predictions histogram
    print("Plotting predictions histogram...")
    plot_predictions_histogram(
        train_predictions, 
        val_predictions, 
        test_predictions,
        datamodule.label_mapping,
        experiment_id,
        os.path.join(output_dir, 'predictions_histogram_new.png')
    )
    
    print(f"Predictions saved to {output_dir}")
    
    return train_predictions, val_predictions, test_predictions