"""
Training and evaluation pipelines for comparing different model architectures
for predicting cancer dependency prediction.
"""

import numpy as np
from .train import CancerTrainer, train_test_split
from .model import MultimodalCancerNet, SimpleConcatNet, DeepMLP, BaselineNet
from torch.utils.data import DataLoader
import torch


def evaluate_models(data_dict, val_size=0.2, test_size=0.2, models=None, random_state=42, **kwargs):
    """
    Compare different model architectures including baseline without embeddings.

    This function handles the complete pipeline of training multiple model
    architectures and comparing their performance.

    Args:
        data_dict (dict): Dictionary containing all modality data
        val_size (float): Proportion of data to randomly hold out for validation
        test_size (float): Proportion of data for testing (default: 0.2)
        models (dict): Dictionary of model instances to evaluate (optional)
        random_state (int): Random seed for reproducibility (default: 42)
        **kwargs: Additional arguments including:
            - lr: Learning rate
            - weight_decay: L2 regularization factor
            - batch_size: Training batch size
            - epochs: Number of training epochs
            - early_stopping_patience: Patience for early stopping

    Returns:
        dict: Results for each model containing:
            - mse: Mean squared error
            - mae: Mean absolute error
            - r2: R-squared score
            - training_history: Loss history during training

    Notes:
        - If models not provided, uses default set of architectures
        - Compares performance against baseline without protein bert embeddings
        - Calculates percentage improvement over baseline
        - Handles model training, evaluation, and metric computation
    """

    trainer_init_params = {
        'lr': kwargs.get('lr', 1e-4),
        'weight_decay': kwargs.get('weight_decay', 1e-5)
    }

    training_params = {
        'batch_size': kwargs.get('batch_size', 32),
        'epochs': kwargs.get('epochs', 50),
        'early_stopping_patience': kwargs.get('early_stopping_patience', 10)
    }

    if models is None:
        # Get dimensions from data
        expression_dim = data_dict['expression_data'].shape[1]
        mutation_dim = data_dict['mutation_data'].shape[1]
        embedding_dim = data_dict['embedding_data'].shape[1]

        models = {
            'MultimodalCancerNet': MultimodalCancerNet(
                expression_dim=expression_dim,
                mutation_dim=mutation_dim,
                embedding_dim=embedding_dim
            ),
            'SimpleConcatNet': SimpleConcatNet(
                expression_dim=expression_dim,
                mutation_dim=mutation_dim,
                embedding_dim=embedding_dim
            ),
            'DeepMLP': DeepMLP(
                expression_dim=expression_dim,
                mutation_dim=mutation_dim,
                embedding_dim=embedding_dim
            ),
            'BaselineNoEmbeddings': BaselineNet(
                expression_dim=expression_dim,
                mutation_dim=mutation_dim
            )
        }

    # Split data
    train_data, test_data = train_test_split(data_dict, test_size=test_size, random_state=random_state)

    # Track comparative performance
    baseline_performance = None
    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        trainer = CancerTrainer(model=model, **trainer_init_params)

        # Train model
        trainer.train(train_data, val_size, **training_params)

        # # Load the best model checkpoint before evaluation
        # print(f"Loading best checkpoint for {name}...")
        # checkpoint = torch.load('best_model.pt', map_location=trainer.device)
        # trainer.model.load_state_dict(checkpoint['model_state_dict'])
        # trainer.normalizer = checkpoint['normalizer']

        # Evaluate on test set using best model
        test_dataset = trainer.prepare_data(test_data)
        test_loader = DataLoader(
            test_dataset,
            batch_size=training_params['batch_size'],
            drop_last=False
        )
        test_metrics = trainer.evaluate(test_loader)

        # Calculate metrics
        predictions = test_metrics['predictions']
        actuals = test_metrics['actuals']
        mse = np.mean((predictions - actuals) ** 2)
        mae = np.mean(np.abs(predictions - actuals))
        r2 = 1 - np.sum((actuals - predictions) ** 2) / np.sum((actuals - np.mean(actuals)) ** 2)

        results[name] = {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'best_model': trainer.model
        }

        print(f"{name} Results:")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R2: {r2:.4f}")

        # Store baseline performance for comparison
        if name == 'BaselineNoEmbeddings':
            baseline_performance = mse
        elif baseline_performance is not None:
            improvement = ((baseline_performance - mse) / baseline_performance) * 100
            print(f"Improvement over baseline: {improvement:.2f}%")

    return results
