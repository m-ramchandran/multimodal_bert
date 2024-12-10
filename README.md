# MultimodalBERT

A deep learning package for predicting gene knockout effects in cancer cell lines using multimodal data integration. This project combines multiple data modalities to predict DepMap gene dependency scores.
Given a specific cancer cell line and target gene, the model predicts how essential that gene is for the cell line's survival (dependency score) by integrating:

- The cell line's gene expression profile
- The cell line's mutation profile
- The target gene's protein sequence (encoded via ProtBERT)

# Data Sources

- Cell line expression and mutation data from the DepMap project (CCLE)
- Gene dependency scores from the DepMap CERES dataset
- Protein sequences from UniProt
- Protein embeddings generated using the PyTorch implementation of ProtBERT

# Technical Approach

The project implements four deep learning architectures that fuse these different data modalities:

## MultimodalCancerNet Architecture

Our main architecture uses a biologically-motivated design that processes and integrates different cellular data modalities:

1. **Separate Modality Processing**:
   - Each data type (expression, mutation, protein sequence) is first processed independently through specialized encoders
   - This reflects the biological reality that these different molecular features operate through distinct mechanisms
   - Each encoder uses multiple residual MLP blocks to learn hierarchical features

2. **Cross-Attention Fusion**:
   - Expression and mutation data are first fused using cross-attention
   - This captures how mutations can affect gene expression patterns
   - The fused representation then attends to protein sequence features
   - This design reflects how protein structure/function information modulates genetic effects

3. **Final Integration**:
   - Features from all modalities are combined using a fusion network
   - Multiple fusion layers allow the model to learn complex interactions
   - Dropout and batch normalization maintain regularization


Other comparison architectures:
- SimpleConcatNet: Baseline comparison that concatenates features in a shallow MLP 
- DeepMLP: Alternative baseline comparison with residual connections and more layers
- BaselineNet: Control model without protein sequence information using shallow MLP 

## Installation

```bash
pip install git+https://github.com/yourusername/multimodal_bert.git
```

## Requirements

- Python >=3.7
- PyTorch >=1.9.0
- Transformers >=4.5.0
- scikit-learn >=0.24.0
- pandas >=1.2.0
- numpy >=1.19.0
- cancer_data >0.3.5

## Example

```python
from multimodal_bert import FullDataLoader, evaluate_models
import torch

# Set device based on GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load and process data
dl = FullDataLoader()
depmap_data = dl.load_cancer_data()

# Predict effect scores for 200 genes (total number of genes is 16,727)
sequences = dl.load_sequences(list(depmap_data['expression_data'].columns)[:201])

# Compute embeddings using specified device
bert_embed = dl.compute_protbert_embeddings(sequences, device=device)
transformed_data = dl.transform_to_long_format(depmap_data, bert_embed)

# Train and evaluate models
results = evaluate_models(
    transformed_data,
    test_size=0.2,
    val_size=0.2,
    lr=1e-4,
    weight_decay=1e-5,
    batch_size=64,
    epochs=50
)

# Print results
for name, metrics in results.items():
    print(f"\n{name}:")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"R2: {metrics['r2']:.4f}")
```

## Results

The MultimodalCancerNet architecture demonstrates superior performance compared to baseline approaches:

- Achieves >15% lower MSE compared to the next best performing model (DeepMLP), even as target gene set varies in size
- Consistent improvement across different cell lines and gene types
- Better captures complex biological relationships by explicitly modeling modality interactions
- More robust predictions, especially for genes with complex functional effects
- Protein embeddings are essential for performance; baseline with only 'omics data captures minimal signal 
This performance improvement illustrates how proper biological modeling through architectural design choices leads to better predictive power.

## Features

- Data loading and preprocessing pipeline for cancer dependency data and 'omics sets
- Protein sequence embedding using ProtBERT
- Comparison of main multi-modal fusion architecture with 3 other neural architectures
- Training pipeline with early stopping and learning rate scheduling
- Evaluation metrics
- GPU/CPU compatibility with automatic device detection

## Citation

If you use this package, please cite:

```bibtex
@software{multimodal_bert,
  author = {Maya Ramchandran},
  title = {MultimodalBERT: Cancer Dependency Prediction},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/m-ramchandran/multimodal_bert}
}
```

## License

MIT License