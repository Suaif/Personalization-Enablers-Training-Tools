#!/usr/bin/env python3

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
import shutil
from datetime import datetime
import argparse
from clearml import Task

class ExperimentRunner:
    def __init__(self, base_dir: str = None):
        """Initialize the experiment runner.
        
        Args:
            base_dir: Base directory of the project. If None, uses current directory.
        """
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent
        self.env_file = self.base_dir / ".env"
        self.config_file = self.base_dir / "configuration.json"
        self.config_backup = self.base_dir / "configuration.json.backup"
        self.env_backup = self.base_dir / ".env.backup"
        self.run_script = self.base_dir / "run_all_dockers-bm-gpu.sh"
        self.experiments_config = self.base_dir / "experiments_config.json"
        
    def backup_files(self):
        """Create backups of .env and configuration.json"""
        if self.config_file.exists():
            shutil.copy(self.config_file, self.config_backup)
            print(f"✓ Backed up configuration.json")
        if self.env_file.exists():
            shutil.copy(self.env_file, self.env_backup)
            print(f"✓ Backed up .env")
    
    def restore_files(self):
        """Restore original .env and configuration.json from backups"""
        if self.config_backup.exists():
            shutil.copy(self.config_backup, self.config_file)
            self.config_backup.unlink()
            print(f"✓ Restored configuration.json")
        if self.env_backup.exists():
            shutil.copy(self.env_backup, self.env_file)
            self.env_backup.unlink()
            print(f"✓ Restored .env")
    
    def load_base_config(self) -> Dict[str, Any]:
        """Load the base configuration from backup."""
        with open(self.config_backup, 'r') as f:
            return json.load(f)
    
    def update_env(self, experiment_id: str):
        """Update the EXPERIMENT_ID in .env file.
        
        Args:
            experiment_id: New experiment ID to set
        """
        env_lines = []
        updated = False
        
        if self.env_file.exists():
            with open(self.env_file, 'r') as f:
                env_lines = f.readlines()
        
        # Update or add EXPERIMENT_ID
        for i, line in enumerate(env_lines):
            if line.strip().startswith('EXPERIMENT_ID='):
                env_lines[i] = f'EXPERIMENT_ID={experiment_id}\n'
                updated = True
                break
        
        if not updated:
            env_lines.append(f'EXPERIMENT_ID={experiment_id}\n')
        
        with open(self.env_file, 'w') as f:
            f.writelines(env_lines)
        
        print(f"✓ Updated .env with EXPERIMENT_ID={experiment_id}")
    
    def update_config(self, config_updates: Dict[str, Any]):
        """Update configuration.json with new values.
        
        Args:
            config_updates: Dictionary with configuration updates.
                           Use dot notation for nested keys, e.g.:
                           {'ssl_config.epochs': 100, 'sup_config.batch_size': 32}
        """
        # Load base config (from backup)
        config = self.load_base_config()
        
        # Apply updates
        for key_path, value in config_updates.items():
            keys = key_path.split('.')
            current = config
            
            # Navigate to the nested key
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # Set the value
            current[keys[-1]] = value
            print(f"  - {key_path}: {value}")
        
        # Save updated config
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Updated configuration.json")
    
    def clean_shimmer_directory(self):
        """Clean the outputs/XRoom/shimmer directory."""
        shimmer_dir = self.base_dir / "outputs" / "XRoom" / "shimmer"
        print(f"Cleaning {shimmer_dir}...")
        if shimmer_dir.exists():
            try:
                for item in shimmer_dir.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
            except Exception as e:
                print(f"Warning: Failed to clean {shimmer_dir}: {e}")
        else:
            shimmer_dir.mkdir(parents=True, exist_ok=True)
            
    def save_results(self, experiment_id: str, experiment_group: str = None):
        """Save experiment results to outputs/exp_results/[experiment_group]/experiment_id/.
        
        Args:
            experiment_id: Unique identifier for this experiment
            experiment_group: Optional subdirectory to organize related experiments
        """
        if experiment_group:
            exp_results_dir = self.base_dir / "outputs" / "exp_results" / experiment_group / experiment_id
        else:
            exp_results_dir = self.base_dir / "outputs" / "exp_results" / experiment_id

        print(f"Saving results to {exp_results_dir}...")

        if exp_results_dir.exists():
            print("Warning: Overwriting existing results directory.")
            shutil.rmtree(exp_results_dir)
        
        exp_results_dir.mkdir(parents=True, exist_ok=True)
        
        if self.config_file.exists():
            shutil.copy(self.config_file, exp_results_dir / "configuration.json")
        
        # Load current configuration to get dataset name and modality
        with open(self.config_file, 'r') as f:
            config = json.load(f)
        
        dataset_name = config.get("dataset_config", {}).get("dataset_name", "default_dataset")
        modality = config.get("dataset_config", {}).get("modality", "default_modality")
        
        if isinstance(modality, list) and "shimmer" in modality:
            modality = "shimmer"
        
        outputs_folder = self.base_dir / "outputs"
        modality_folder = outputs_folder / dataset_name / modality
        component_output_folder = modality_folder / "supervised_training"

        # Results
        results_dir = outputs_folder / "results" / experiment_id
        ssl_metrics_file = results_dir.parent / f"{experiment_id}_test_metrics_ssl.json"
        supervised_metrics_file = results_dir.parent / f"{experiment_id}_test_metrics_supervised.json"
        
        merged_results = {}
        if ssl_metrics_file.exists():
            with open(ssl_metrics_file, "r") as f:
                merged_results["ssl_metrics"] = json.load(f)
        else:
            print(f"Warning: SSL metrics file not found: {ssl_metrics_file}")
        
        if supervised_metrics_file.exists():
            with open(supervised_metrics_file, "r") as f:
                merged_results["supervised_metrics"] = json.load(f)
        else:
            print(f"Warning: Supervised metrics file not found: {supervised_metrics_file}")
        
        if merged_results:
            merged_results_path = exp_results_dir / "merged_metrics.json"
            with open(merged_results_path, "w") as f:
                json.dump(merged_results, f, indent=2)

        # Predictions
        prediction_files = ["train_predictions.csv", "val_predictions.csv", "test_predictions.csv"]
        for pred_file in prediction_files:
            src_path = component_output_folder / pred_file
            if src_path.exists():
                shutil.copy(src_path, exp_results_dir / pred_file)
            else:
                print(f"  ⚠ Warning: Prediction file not found: {src_path}")
        
        return exp_results_dir

    def run_experiment(self, experiment_id: str, config_updates: Dict[str, Any] = None, description: str = "", experiment_group: str = None, clean_shimmer: bool = True):
        """Run a single experiment with ClearML tracking.
        
        Args:
            experiment_id: Unique identifier for this experiment
            config_updates: Optional configuration updates for this experiment
            description: Optional description of the experiment
            experiment_group: Optional subdirectory to organize related experiments (e.g., 'batch_size_exps')
            clean_shimmer: Optional flag to clean the shimmer directory before running the experiment
        """
        print(f"\n{'='*60}")
        print(f"Experiment: {experiment_id}")
        if description:
            print(f"Description: {description}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        
        # Initialize ClearML Task
        task = Task.init(
            project_name="Personalization-Enablers", 
            task_name=experiment_id,
            reuse_last_task_id=False
        )
        
        # Update .env
        self.update_env(experiment_id)
        
        # Update configuration if needed
        if config_updates:
            print(f"Configuration updates:")
            self.update_config(config_updates)
            
        # Connect configuration to ClearML
        full_config = self.load_base_config()
        # Apply updates to the loaded config object for logging purposes (the file is already updated)
        if config_updates:
            with open(self.config_file, 'r') as f:
                full_config = json.load(f)
        
        task.connect(full_config)
        if description:
            task.set_comment(description)
        
        if experiment_group:
            task.add_tags([experiment_group])
            
        if clean_shimmer:
            self.clean_shimmer_directory()
        
        print(f"\nRunning experiment pipeline...")
        success = False
        try:
            script_name = self.run_script.name
            
            result = subprocess.run(
                ['bash', script_name],
                cwd=str(self.base_dir),
                check=True,
                env={**os.environ, 'EXPERIMENT_ID': experiment_id}
            )
            
            print(f"\n✓ Experiment {experiment_id} completed successfully!")
            success = True
            
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Experiment {experiment_id} failed with error {e}")
            success = False
            
        if success:
            exp_results_dir = self.save_results(experiment_id, experiment_group)
            
            # Load and log metrics and artifacts to ClearML
            merged_metrics_path = exp_results_dir / "merged_metrics.json"
            if merged_metrics_path.exists():
                with open(merged_metrics_path, 'r') as f:
                    metrics = json.load(f)
                
                if "ssl_metrics" in metrics:
                    for metric_name, metric_value in metrics["ssl_metrics"].items():
                        task.get_logger().report_single_value(
                            name=f"ssl/{metric_name}",
                            value=metric_value
                        )
                
                if "supervised_metrics" in metrics:
                    for metric_name, metric_value in metrics["supervised_metrics"].items():
                        task.get_logger().report_single_value(
                            name=f"supervised/{metric_name}",
                            value=metric_value
                        )
            else:
                print(f"Warning: No merged metrics file found at {merged_metrics_path}")
            
            task.upload_artifact('configuration', artifact_object=exp_results_dir / "configuration.json")
            
            for item in exp_results_dir.iterdir():
                if item.is_file():
                    task.upload_artifact(item.name, artifact_object=item)
                elif item.is_dir():
                    shutil.make_archive(str(item), 'zip', item)
                    task.upload_artifact(item.name, artifact_object=Path(str(item) + '.zip'))
                    Path(str(item) + '.zip').unlink()

            task.mark_completed()
        else:
            task.mark_failed()
            
        task.close()
        return success
    
    def load_experiments_from_file(self, filepath: Path = None) -> List[Dict[str, Any]]:
        """Load experiment configurations from JSON file.
        
        Args:
            filepath: Path to experiments config file. If None, uses default.
            
        Returns:
            List of experiment configurations
        """
        if filepath is None:
            filepath = self.experiments_config
        
        if not filepath.exists():
            print(f"Error: Experiments config file not found: {filepath}")
            sys.exit(1)
        
        with open(filepath, 'r') as f:
            experiments = json.load(f)
        
        return experiments
    
    def run_experiments(self, experiments: List[Dict[str, Any]], clean_shimmer: bool = True):
        """Run multiple experiments sequentially.
        
        Args:
            experiments: List of experiment configurations. Each should have:
                        - 'experiment_id': experiment identifier
                        - 'config_updates': dict of config updates
                        - 'description': optional description
            clean_shimmer: If True, clean the shimmer directory before running the experiment
        """
        
        start_time = time.time()
        
        # Backup original files
        self.backup_files()
        
        total = len(experiments)
        successful = 0
        failed = 0
        failed_experiments = []
        
        try:
            for i, exp in enumerate(experiments, 1):
                exp_id = exp['experiment_id']
                config_updates = exp.get('config_updates', {})
                description = exp.get('description', '')
                experiment_group = exp.get('experiment_group', None)
                
                print(f"\n\n{'#'*60}")
                print(f"# Experiment {i}/{total}")
                print(f"{'#'*60}")
                
                success = self.run_experiment(exp_id, config_updates, description, experiment_group, clean_shimmer)
                
                if success:
                    successful += 1
                else:
                    failed += 1
                    failed_experiments.append(exp_id)
                        
        finally:
            # Restore original files
            print(f"\n\nRestoring original configuration files...")
            self.restore_files()
        
        # Print summary
        print(f"\n\n{'='*60}")
        print(f"EXPERIMENT SUITE SUMMARY")
        print(f"{'='*60}")
        print(f"Experiment time: {time.time() - start_time}")
        print(f"Total experiments: {total}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        if failed_experiments:
            print(f"Failed experiments: {', '.join(failed_experiments)}")
        print(f"{'='*60}\n")
        
        return successful, failed


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Run multiple experiments with different configurations'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='experiments_config.json',
        help='Path to experiments configuration file (default: experiments_config.json)'
    )
    parser.add_argument(
        '--clean-shimmer',
        default=True,
        action=argparse.BooleanOptionalAction,
        help='Clean the shimmer directory before running the experiment'
    )
    
    args = parser.parse_args()
    
    # Create runner
    runner = ExperimentRunner()
    
    # Load experiments
    config_path = Path(args.config)
    experiments = runner.load_experiments_from_file(config_path)
    
    # List experiments
    print(f"\nFound {len(experiments)} experiments in {config_path}:\n")
    for i, exp in enumerate(experiments, 1):
        print(f"{i}. {exp['experiment_id']}")
        if 'description' in exp:
            print(f"   {exp['description']}")
        print()
    
    # Run experiments
    print(f"\nLoaded {len(experiments)} experiments from {config_path}")
    
    successful, failed = runner.run_experiments(experiments, args.clean_shimmer)
    
    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)

if __name__ == '__main__':
    main()
