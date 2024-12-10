from multimodal_bert import FullDataLoader, evaluate_models

if __name__ == '__main__':
    # Load and process data
    dl = FullDataLoader()
    depmap_data = dl.load_cancer_data()
    sequences = dl.load_sequences(list(depmap_data['expression_data'].columns)[:601])
    bert_embed = dl.compute_protbert_embeddings(sequences, device="cpu", batch_size=50, max_workers=4)
    transformed_data = dl.transform_to_long_format(depmap_data, bert_embed)

    # Evaluate different models
    results = evaluate_models(
        transformed_data,
        test_size=0.2,
        val_size=0.2,
        lr=1e-4,
        weight_decay=1e-5,
        batch_size=128,
        epochs=70,
        random_state=42
    )

    # Compare results
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"Final MSE: {metrics['mse']:.4f}")
        print(f"Final MAE: {metrics['mae']:.4f}")
        print(f"R2 Score: {metrics['r2']:.4f}")

