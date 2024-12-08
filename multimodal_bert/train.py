"""
Training utilities for cancer dependency prediction models.

This module provides classes and functions for training and evaluating
multimodal cancer prediction models, including data normalization,
training loops, and evaluation metrics.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader


class FeatureNormalizer:
    """
    Normalizer for processing different modalities of input data.

    Handles standardization of expression and embedding features while
    leaving binary mutation data unchanged.

    Attributes:
        expression_scaler: StandardScaler for expression data
        embedding_scaler: StandardScaler for embedding data
        fitted (bool): Whether the normalizer has been fitted to data
    """
    def __init__(self):
        self.expression_scaler = StandardScaler()
        self.embedding_scaler = StandardScaler()
        self.fitted = False

    def fit(self, expression_data, embedding_data):
        """
        Fit scalers on training data (subtract mean and divide by standard deviation).

        Args:
            expression_data (np.ndarray): Gene expression features
            embedding_data (np.ndarray): Protein embedding features

        Notes:
            - Should only be called on training data
        """

        self.expression_scaler.fit(expression_data)
        self.embedding_scaler.fit(embedding_data)
        self.fitted = True

    def transform(self, expression_data, embedding_data):
        """Transform data using fitted scalers"""
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before transform")
        return (
            self.expression_scaler.transform(expression_data),
            self.embedding_scaler.transform(embedding_data)
        )


class CancerTrainer:
    """
    Trainer class for cancer dependency prediction models.

    Handles the complete training workflow including data normalization,
    training loops, evaluation, and model checkpointing.

    Args:
        model (nn.Module): PyTorch model to train
        lr (float): Learning rate for optimization (default: 1e-4)
        weight_decay (float): L2 regularization factor (default: 1e-5)

    Attributes:
        device: Computing device (CPU/GPU)
        optimizer: AdamW optimizer instance
        criterion: MSE loss function
        scheduler: Learning rate scheduler
        normalizer: Feature normalizer instance
    """
    def __init__(self, model, lr=1e-4, weight_decay=1e-5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        self.normalizer = FeatureNormalizer()

    @staticmethod
    def prepare_data(data_dict):
        """Create TensorDataset directly from dict values"""
        return TensorDataset(
            torch.tensor(data_dict['expression_data'].values, dtype=torch.float),
            torch.tensor(data_dict['mutation_data'].values, dtype=torch.float),
            torch.tensor(data_dict['embedding_data'].values, dtype=torch.float),
            torch.tensor(data_dict['effect_data']['effect_score'].values, dtype=torch.float)
        )

    def normalize_batch(self, batch):
        """Apply normalization to a batch of data with minimal CPU-GPU transfers"""
        expression_data, mutation_data, embedding_data, labels = [b.to(self.device) for b in batch]

        expr_norm, emb_norm = self.normalizer.transform(
            expression_data.cpu().numpy(),
            embedding_data.cpu().numpy()
        )

        return {
            'expression_data': torch.from_numpy(expr_norm).float().to(self.device),
            'mutation_data': mutation_data,  # Already on device
            'embedding_data': torch.from_numpy(emb_norm).float().to(self.device),
            'label': labels  # Already on device
        }

    def train_epoch(self, dataloader):
        """
        Train model for one epoch.

        Args:
            dataloader (DataLoader): Training data loader

        Returns:
            float: Average training loss for the epoch

        Notes:
            - Implements gradient clipping
            - Uses batch normalization in training mode
        """
        self.model.train()
        total_loss = 0

        for batch in dataloader:
            self.optimizer.zero_grad()
            normalized_batch = self.normalize_batch(batch)

            outputs = self.model(
                expression_data=normalized_batch['expression_data'],
                mutation_data=normalized_batch['mutation_data'],
                embedding_data=normalized_batch['embedding_data']
            )

            loss = self.criterion(outputs, normalized_batch['label'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        """
        Evaluate model on provided external data.

        Args:
            dataloader (DataLoader): Evaluation data loader

        Returns:
            dict: Evaluation metrics including:
                - loss: Average loss value
                - predictions: Model predictions
                - actuals: True target values
        """
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_actuals = []

        with torch.no_grad():
            for batch in dataloader:
                normalized_batch = self.normalize_batch(batch)
                outputs = self.model(
                    expression_data=normalized_batch['expression_data'],
                    mutation_data=normalized_batch['mutation_data'],
                    embedding_data=normalized_batch['embedding_data']
                )

                loss = self.criterion(outputs, normalized_batch['label'])
                total_loss += loss.item() * len(outputs)
                all_predictions.append(outputs.cpu().numpy())
                all_actuals.append(normalized_batch['label'].cpu().numpy())

        predictions = np.concatenate(all_predictions)
        actuals = np.concatenate(all_actuals)

        return {
            'loss': total_loss / len(dataloader.dataset),
            'predictions': predictions,
            'actuals': actuals
        }

    def train(self, train_data, val_data, batch_size=32, epochs=50, early_stopping_patience=10):
        """
        Complete training loop with validation and early stopping.

        Args:
            train_data (dict): Training data dictionary
            val_data (dict): Validation data dictionary
            batch_size (int): Batch size for training (default: 32)
            epochs (int): Maximum number of epochs (default: 50)
            early_stopping_patience (int): Patience for early stopping (default: 10)

        Returns:
            dict: Training history containing:
                - train_losses: List of training losses
                - val_losses: List of validation losses

        Notes:
            - Implements early stopping based on validation loss
            - Saves best model checkpoint
            - Uses learning rate scheduling
        """

        # Create datasets and dataloaders
        train_dataset = self.prepare_data(train_data)
        val_dataset = self.prepare_data(val_data)

        # Configure dataloaders with drop_last
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            drop_last=True
        )

        # Fit normalizer on training data
        self.normalizer.fit(
            train_data['expression_data'].values,
            train_data['embedding_data'].values
        )

        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            val_loss = val_metrics['loss']

            self.scheduler.step(val_loss)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'normalizer': self.normalizer
                }, 'best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print("Early stopping triggered!")
                    break

        return {'train_losses': train_losses, 'val_losses': val_losses}

