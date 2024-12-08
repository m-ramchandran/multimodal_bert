"""
multimodal_bert
==============

A package for multimodal cancer prediction using BERT embeddings and omics data.

This package combines gene expression data, mutation data, and protein sequence
embeddings from BERT to predict gene knockout effect scores.

Main Components
--------------
- FullDataLoader: Handles data loading and preprocessing
- MultimodalCancerNet: Main model architecture for multimodal prediction of knockout effect scores
- CancerTrainer: Training pipeline for cancer prediction models

Example
-------
>>> from multimodal_bert import FullDataLoader, evaluate_models
>>> dl = FullDataLoader()
>>> depmap_data = dl.load_cancer_data()
>>> sequences = dl.load_sequences(list(depmap_data['expression_data'].columns)[:601])
>>> bert_embed = dl.compute_protbert_embeddings(sequences)
>>> transformed_data = dl.transform_to_long_format(depmap_data, bert_embed)
>>> results = evaluate_models(transformed_data)
"""

from .data import FullDataLoader
from .model import (
    MultimodalCancerNet,
    SimpleConcatNet,
    DeepMLP,
    BaselineNet,
    CrossAttentionLayer,
    ModalityEncoder,
    FusionNetwork
)
from .train import CancerTrainer, FeatureNormalizer
from .train_pipelines import evaluate_models, train_test_split

__version__ = "0.1.0"
__author__ = "Maya Ramchandran"
__email__ = "ramchandran.maya@gmail.com"

__all__ = [
    'FullDataLoader',
    'MultimodalCancerNet',
    'SimpleConcatNet',
    'DeepMLP',
    'BaselineNet',
    'CrossAttentionLayer',
    'ModalityEncoder',
    'FusionNetwork',
    'CancerTrainer',
    'FeatureNormalizer',
    'evaluate_models',
    'train_test_split'
]