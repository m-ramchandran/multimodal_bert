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

- MultimodalCancerNet: Main architecture using cross-attention for modality fusion
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

# Load and process data
dl = FullDataLoader()
depmap_data = dl.load_cancer_data()

#predict effect scores for first 600 genes (total number of genes is > 16,000)
sequences = dl.load_sequences(list(depmap_data['expression_data'].columns)[:600])
bert_embed = dl.compute_protbert_embeddings(sequences)
transformed_data = dl.transform_to_long_format(depmap_data, bert_embed)

# Train and evaluate models
results = evaluate_models(
    transformed_data,
    test_size=0.2,
    val_size=0.2,
    lr=1e-4,
    weight_decay=1e-5,
    batch_size=32,
    epochs=70
)
```

## Features

- Data loading and preprocessing pipeline for cancer dependency data
- Protein sequence embedding using ProtBERT
- Comparison of main multi-modal fusion architecture with 3 other neural architectures
- Training pipeline with early stopping and learning rate scheduling
- Evaluation metrics

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