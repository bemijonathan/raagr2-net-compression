# MLflow Experiment Tracking

This directory contains utilities for tracking machine learning experiments with MLflow. MLflow helps with reproducibility, visualization, and comparison of different experiments.

## MLflow Utilities

### Main MLflow Utility (`mlflow.py`)

The main utility file provides a class `MLFlowExperiment` for tracking experiments and helper functions for common operations.

Features:
- Track model parameters, hyperparameters, and configurations
- Log metrics during training and evaluation
- Save model checkpoints and artifacts
- Visualize predictions and results
- Compare different models and pruning techniques

### Training with MLflow (`training_with_mlflow.py`)

An example script that shows how to integrate MLflow with the brain segmentation model training process.

### Pruning with MLflow (`pruning_with_mlflow.py`)

An example script that demonstrates how to track model pruning experiments with MLflow, supporting both magnitude-based and DeGraph pruning methods.

## Usage Examples

### Basic Usage

```python
from utils.mlflow import MLFlowExperiment

# Initialize an experiment
experiment = MLFlowExperiment(
    experiment_name="my_experiment",
    model=my_model,
    run_name="first_run"
)

# Log parameters
experiment.log_params({
    "learning_rate": 0.001,
    "batch_size": 32,
    "model_type": "DLUNet"
})

# Log metrics
experiment.log_metrics({
    "train_loss": 0.5,
    "train_dice": 0.8
}, step=1)

# Log the model
experiment.log_model(registered_model_name="my_model")

# End the experiment
experiment.end_run()
```

### Using the Helper Functions

```python
from utils.mlflow import setup_experiment, log_epoch_metrics, log_best_model

# Set up an experiment with configuration
config = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "num_epochs": 10
}
experiment = setup_experiment(
    experiment_name="brain_segmentation",
    model=model,
    config=config
)

# Log metrics for each epoch
for epoch in range(num_epochs):
    # ... training and validation ...
    
    log_epoch_metrics(
        experiment=experiment,
        train_metrics={"loss": train_loss, "dice": train_dice},
        val_metrics={"loss": val_loss, "dice": val_dice},
        epoch=epoch
    )

# Log the best model
log_best_model(
    experiment=experiment,
    model=model,
    model_path="model/best_model.pth",
    metrics={"loss": best_loss, "dice": best_dice}
)
```

### Training a Model with MLflow Tracking

```bash
# Run the training script with MLflow tracking
python utils/training_with_mlflow.py
```

### Pruning a Model with MLflow Tracking

```bash
# Run magnitude-based pruning with MLflow tracking
python utils/pruning_with_mlflow.py magnitude

# Run DeGraph pruning with MLflow tracking
python utils/pruning_with_mlflow.py degraph
```

## Viewing Experiments

After running experiments, you can view the results using the MLflow UI:

```bash
# Start the MLflow UI
mlflow ui
```

Then open your browser and navigate to http://127.0.0.1:5000 to see your experiments, runs, and metrics.

## Best Practices

1. **Experiment Names**: Use consistent experiment names for related runs (e.g., "brain_segmentation" for all training experiments).

2. **Run Names**: Use descriptive run names that include key information (e.g., "dlunet_lr0.001_batch32").

3. **Parameters**: Log all relevant hyperparameters and configuration values at the beginning of each run.

4. **Metrics**: Log training and validation metrics for each epoch to track progress over time.

5. **Artifacts**: Save model checkpoints, visualizations, and other artifacts to reproduce results.

6. **Tagging**: Use tags to categorize runs (e.g., "baseline", "pruned", "fine-tuned").

7. **Comparison**: Use the MLflow UI to compare different runs and identify the best models.

## Customization

You can customize the MLflow tracking by:

1. Setting a different tracking URI for a remote MLflow server.
2. Creating nested runs for more complex experiments.
3. Adding custom visualization functions for your specific models and datasets.
4. Implementing custom metrics and artifacts for your specific use case. 