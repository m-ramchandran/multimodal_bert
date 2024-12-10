import torch
import torch.nn as nn


class CrossAttentionLayer(nn.Module):
    """
    Cross-attention layer for combining information from different modalities.

    Implements multi-head attention mechanism where queries come from one modality
    and keys/values come from another, allowing the model to learn which aspects
    of each modality are relevant for prediction.

    Args:
        hidden_dim (int): Dimension of hidden representations
        num_heads (int): Number of attention heads
    """
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, y):
        x_attn = x.unsqueeze(0)
        y_attn = y.unsqueeze(0)
        attn_output, _ = self.attention(x_attn, y_attn, y_attn)
        return self.norm(x + attn_output.squeeze(0))


class ModalityEncoder(nn.Module):
    """
    Encoder network for processing individual data modalities.

    Transforms raw features from each modality into a latent space using
    multiple MLP layers with residual connections and normalization.

    Args:
        input_dim (int): Dimension of input features
        hidden_size (int): Dimension of hidden layers

    Notes:
        - Uses batch normalization and dropout for regularization
        - Includes residual connections
        - Applies ReLU activation after each linear layer
        """

    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size)
        )

        self.mlp_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.2)
            ) for _ in range(4)
        ])

    def forward(self, x):
        x = self.input_proj(x)

        # Process through MLP layers with residual connections
        for layer in self.mlp_layers:
            x = x + layer(x)

        return x


class FusionNetwork(nn.Module):
    """
    Network for fusing processed features from different modalities.

    Combines representations from multiple modalities using cross-attention
    and deep fusion layers to make final predictions.

    Args:
        hidden_size (int): Dimension of hidden layers

    Notes:
        - Uses cross-attention to combine expression and mutation data
        - Applies additional cross-attention with gene embeddings
        - Includes final MLP layers for prediction
    """

    def __init__(self, hidden_size):
        super().__init__()

        # Cross-attention between modalities
        self.expr_mut_attention = CrossAttentionLayer(hidden_size)
        self.gene_attention = CrossAttentionLayer(hidden_size)

        # Final fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size * 2),
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, expr_features, mut_features, gene_features):
        # Cross attention between expression and mutation
        expr_mut = self.expr_mut_attention(expr_features, mut_features)

        # Cross attention with gene features
        expr_mut_gene = self.gene_attention(expr_mut, gene_features)

        # Concatenate all features
        combined = torch.cat([expr_mut_gene, expr_features, gene_features], dim=1)

        return self.fusion_layers(combined).squeeze()


class MultimodalCancerNet(nn.Module):
    """
    Complete architecture for multimodal deep learning architecture incorporating
    uni-modal processing, cross-attention, and a final fusion network to predict
    gene dependency scores.

    Args:
        expression_dim (int): Dimension of expression features
        mutation_dim (int): Dimension of mutation features
        embedding_dim (int): Dimension of protein embeddings
        hidden_size (int): Dimension of hidden layers

    Architecture Overview:
        1. Separate encoder networks process each modality
        2. Cross-attention layers combine modality representations
        3. Final fusion network makes dependency predictions

    Notes:
        - Designed for processing gene expression, mutation, and sequence data
        - Uses attention mechanisms for interpretable feature combination
        - Implements deep supervision through skip connections
    """

    def __init__(self, expression_dim, mutation_dim, embedding_dim, hidden_size=256):
        super().__init__()

        # Separate encoders for each modality
        self.expression_encoder = ModalityEncoder(expression_dim, hidden_size)
        self.mutation_encoder = ModalityEncoder(mutation_dim, hidden_size)
        self.gene_encoder = ModalityEncoder(embedding_dim, hidden_size)

        # Fusion network
        self.fusion_network = FusionNetwork(hidden_size)

    def forward(self, expression_data, mutation_data, embedding_data):
        # Process each modality separately
        expr_features = self.expression_encoder(expression_data)
        mut_features = self.mutation_encoder(mutation_data)
        gene_features = self.gene_encoder(embedding_data)

        # Fuse features
        return self.fusion_network(expr_features, mut_features, gene_features)


class SimpleConcatNet(nn.Module):
    """
    Baseline model that concatenates all features and applies 3 MLP layers.

    This model serves as a simple baseline that directly concatenates features
    from all modalities without sophisticated fusion mechanisms or residual connections

    Args:
        expression_dim (int): Dimension of expression features
        mutation_dim (int): Dimension of mutation features
        embedding_dim (int): Dimension of protein embeddings
        hidden_size (int): Dimension of hidden layers (default: 256)

    Notes:
        - Concatenates all features into a single vector
        - Uses batch normalization and dropout for regularization
        - Implements a 3-layer MLP architecture
    """

    def __init__(self, expression_dim, mutation_dim, embedding_dim, hidden_size=256):
        super().__init__()
        total_dim = expression_dim + mutation_dim + embedding_dim

        self.network = nn.Sequential(
            nn.Linear(total_dim, hidden_size * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size * 2),
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, expression_data, mutation_data, embedding_data):
        combined = torch.cat([expression_data, mutation_data, embedding_data], dim=1)
        return self.network(combined).squeeze()


class DeepMLP(nn.Module):
    """
    Deep MLP architecture with residual connections.

    Implements a deeper (6 layers) network architecture with skip connections
    for better gradient flow and feature processing.

    Args:
        expression_dim (int): Dimension of expression features
        mutation_dim (int): Dimension of mutation features
        embedding_dim (int): Dimension of protein embeddings
        hidden_size (int): Dimension of hidden layers (default: 256)

    Notes:
        - Uses residual connections between layers
        - Includes batch normalization and dropout in each block
    """

    def __init__(self, expression_dim, mutation_dim, embedding_dim, hidden_size=256):
        super().__init__()
        total_dim = expression_dim + mutation_dim + embedding_dim

        self.input_proj = nn.Linear(total_dim, hidden_size)

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.2)
            ) for _ in range(4)
        ])

        self.output = nn.Linear(hidden_size, 1)

    def forward(self, expression_data, mutation_data, embedding_data):
        x = torch.cat([expression_data, mutation_data, embedding_data], dim=1)
        x = self.input_proj(x)

        for layer in self.layers:
            residual = x
            x = layer(x)
            x = x + residual

        return self.output(x).squeeze()


class BaselineNet(nn.Module):
    """
    Baseline model that only uses expression and mutation data.

    This model serves as a control to evaluate the impact of adding
    protein bert embeddings to the feature set.

    Args:
        expression_dim (int): Dimension of expression features
        mutation_dim (int): Dimension of mutation features
        hidden_size (int): Dimension of hidden layers (default: 256)

    Notes:
        - Ignores protein sequence embeddings
        - Uses same architecture as SimpleConcatNet
    """

    def __init__(self, expression_dim, mutation_dim, hidden_size=256):
        super().__init__()
        total_dim = expression_dim + mutation_dim

        self.network = nn.Sequential(
            nn.Linear(total_dim, hidden_size * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size * 2),
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, expression_data, mutation_data, embedding_data=None):
        # Note: embedding_data parameter included but not used, to maintain consistent interface
        combined = torch.cat([expression_data, mutation_data], dim=1)
        return self.network(combined).squeeze()
