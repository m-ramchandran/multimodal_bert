import torch
import cancer_data as cd
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import requests
from tqdm import tqdm
from typing import Dict
from transformers import BertModel, BertTokenizer
from pathlib import Path
import os


def get_project_dir() -> Path:
    """
    Get the project's root directory by looking for setup.py.

    Returns:
        Path: Project root directory path
    """
    current_dir = Path(__file__).resolve().parent
    while current_dir.parent != current_dir:
        if (current_dir / 'setup.py').exists():
            return current_dir
        current_dir = current_dir.parent
    return current_dir


# Get project directory and create data directory
project_dir = get_project_dir()
data_dir = project_dir / 'data'
data_dir.mkdir(exist_ok=True)

# Set environment variable
os.environ["CANCER_DATA_DIR"] = str(data_dir)

class FullDataLoader:
    @staticmethod
    def load_cancer_data():
        """
        Load and process cancer datasets with common cell lines and genes.
        """
        # Download datasets
        datasets = ["depmap_gene_tpm", "depmap_mutations", "depmap_sanger_ceres", "depmap_annotations"]
        for dataset in datasets:
            cd.download(dataset)

        # Load datasets from raw directory
        base_path = os.path.join(os.path.dirname(cd.__file__), 'data', 'raw')
        expression = pd.read_csv(os.path.join(base_path, "CCLE_expression_v2.csv"), index_col=0)
        mutations = pd.read_csv(os.path.join(base_path, "CCLE_mutations.csv"))
        effects = pd.read_csv(os.path.join(base_path, "gene_effect.csv"), index_col=0)
        annotations = pd.read_csv(os.path.join(base_path, "sample_info.csv"))

        expression.columns = [col.split(' (', 1)[0] for col in expression.columns]
        effects.columns = [col.split(' (', 1)[0] for col in effects.columns]

        # Get common cell lines
        common_cells = set(expression.index.unique()).intersection(
            set(effects.index.unique()),
            set(mutations['DepMap_ID'].unique()),
            set(annotations['DepMap_ID'].unique())
        )

        # Get common genes
        common_genes = set(expression.columns).intersection(set(effects.columns),
                                                            set(mutations['Hugo_Symbol']))

        # Filter all datasets to the same set of common cells and common genes (in the exact order)
        expression = expression.reindex(index=sorted(common_cells), columns=sorted(common_genes))
        effects = effects.reindex(index=sorted(common_cells), columns=sorted(common_genes))

        mutations_filtered = mutations[
            mutations['DepMap_ID'].isin(common_cells) &
            mutations['Hugo_Symbol'].isin(common_genes)
            ]

        # Then pivot to get same format as expression/effects
        # Using 1/0 to indicate presence/absence of mutation
        mutations = pd.crosstab(
            index=mutations_filtered['DepMap_ID'],
            columns=mutations_filtered['Hugo_Symbol']
        ).reindex(index=list(common_cells), columns=list(common_genes)).fillna(0)

        annotations = annotations.loc[annotations['DepMap_ID'].isin(common_cells)].set_index('DepMap_ID').reindex(
            list(common_cells))

        return {
            'expression_data': expression,
            'mutation_data': mutations,
            'effect_data': effects,
            'annotations': annotations
        }

    @staticmethod
    def _fetch_batch(batch):
        """Fetch sequences for a batch of genes"""
        try:
            gene_query = " OR ".join([f"gene_exact:{gene}" for gene in batch])
            query = f"({gene_query}) AND organism_id:9606"

            response = requests.get(
                "https://rest.uniprot.org/uniprotkb/search",
                params={
                    "query": query,
                    "format": "json",
                    "fields": "gene_names,sequence",
                    "size": len(batch)
                }
            )

            batch_sequences = {}
            if response.ok and 'results' in response.json():
                for result in response.json()['results']:
                    genes_info = result.get('genes', [])
                    if genes_info and genes_info[0].get('geneName', {}).get('value'):
                        gene = genes_info[0]['geneName']['value']
                        if gene in batch:
                            batch_sequences[gene] = result['sequence']['value']
            return batch_sequences

        except Exception as e:
            print(f"Error in batch: {str(e)}")
            return {}

    @staticmethod
    def load_sequences(genes):
        """Load protein sequences from UniProt using parallel requests"""
        batch_size = 100
        max_workers = 10  # Number of parallel threads

        # Split genes into batches
        batches = [genes[i:i + batch_size] for i in range(0, len(genes), batch_size)]

        sequences = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Use tqdm for progress bar
            futures = list(tqdm(
                executor.map(FullDataLoader._fetch_batch, batches),
                total=len(batches),
                desc="Fetching sequences"
            ))

            # Combine results
            for batch_result in futures:
                sequences.update(batch_result)

        return sequences

    @staticmethod
    def compute_protbert_embeddings(sequences: Dict[str, str], device: str = "cpu", batch_size: int = 128,
                                    max_workers: int = 4) -> pd.DataFrame:
        """
        Compute ProtBERT embeddings using concurrent processing.

        Args:
            sequences: Dictionary mapping gene names to their sequences
            device: Computing device ("cpu" or "cuda")
            batch_size: Number of sequences to process in each batch
            max_workers: Number of concurrent workers for processing

        Returns:
            DataFrame containing embeddings for all sequences
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import math

        def process_batch(batch_data):
            if not batch_data:  # Safety check for empty batches
                return {}

            batch_genes, batch_seqs = zip(*batch_data)
            batch_sequences = [" ".join(list(seq)) for seq in batch_seqs]

            with torch.no_grad():
                encoded = tokenizer(
                    batch_sequences,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(device)

                outputs = model(**encoded)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu()

                # Convert to dictionary
                return {gene: emb.numpy() for gene, emb in zip(batch_genes, batch_embeddings)}

        print("Loading ProtBERT model and tokenizer...")
        tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        model = BertModel.from_pretrained("Rostlab/prot_bert")

        # Speed optimizations
        if device == "cuda":
            model = model.half()
        model = model.to(device)
        model.eval()

        # Get ordered list of genes and create batches
        genes = sorted(sequences.keys())
        total_sequences = len(genes)

        # Create batches - ensuring all sequences are included
        batches = []
        for i in range(0, total_sequences, batch_size):
            batch_genes = genes[i:min(i + batch_size, total_sequences)]  # Use min to handle the last batch
            batch_data = [(gene, sequences[gene]) for gene in batch_genes]
            batches.append(batch_data)

        embeddings_dict = {}
        processed_batches = 0

        print(f"Processing {len(batches)} batches ({total_sequences} sequences) using {max_workers} workers...")

        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {executor.submit(process_batch, batch): i for i, batch in enumerate(batches)}

            for future in tqdm(as_completed(future_to_batch),
                               total=len(batches),
                               desc="Computing embeddings"):
                batch_idx = future_to_batch[future]
                try:
                    batch_embeddings = future.result()
                    embeddings_dict.update(batch_embeddings)
                    processed_batches += 1

                except Exception as e:
                    print(f"Batch {batch_idx} generated an exception: {e}")
                    # Log which genes were in the failed batch
                    failed_genes = [gene for gene, _ in batches[batch_idx]]
                    print(f"Failed genes: {failed_genes}")

        # Verify all sequences were processed
        processed_genes = set(embeddings_dict.keys())
        missing_genes = set(genes) - processed_genes
        if missing_genes:
            print(f"Warning: {len(missing_genes)} genes were not processed: {missing_genes}")

        # Convert to DataFrame
        embeddings_df = pd.DataFrame.from_dict(embeddings_dict, orient='index')
        embeddings_df.columns = [f'dim_{i}' for i in range(embeddings_df.shape[1])]

        print(f"\nCompleted processing {len(processed_genes)}/{total_sequences} sequences!")
        print(f"DataFrame shape: {embeddings_df.shape}")

        return embeddings_df

    @staticmethod
    def transform_to_long_format(data_dict: Dict, embeddings_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Transform data into long format with consistent (cell_line, target_gene) multi-index.
        Uses vectorized operations for efficient data alignment.
        """
        # 1. Find common genes
        depmap_genes = set(data_dict['expression_data'].columns)
        embedding_genes = set(embeddings_df.index)
        common_genes = sorted(depmap_genes.intersection(embedding_genes))

        print(f"Original number of genes in DepMap data: {len(depmap_genes)}")
        print(f"Original number of genes in embeddings: {len(embedding_genes)}")
        print(f"Number of common genes: {len(common_genes)}")

        if len(common_genes) == 0:
            raise ValueError("No common genes found between embeddings and original data!")

        # 2. Subset all dataframes to common genes
        subset_dict = {
            'expression_data': data_dict['expression_data'].reindex(columns=common_genes),
            'mutation_data': data_dict['mutation_data'].reindex(columns=common_genes),
            'effect_data': data_dict['effect_data'].reindex(columns=common_genes),
            'annotations': data_dict['annotations']
        }
        embeddings_subset = embeddings_df.reindex(common_genes)

        # 3. Create effect dataframe in long format
        effect_long = subset_dict['effect_data'].melt(
            ignore_index=False,
            var_name='target_gene',
            value_name='effect_score'
        )
        effect_long.index.name = 'cell_line'

        # Create master multi-index
        master_multi_idx = pd.MultiIndex.from_arrays(
            [effect_long.index, effect_long['target_gene']],
            names=['cell_line', 'target_gene']
        )

        # 4. Create effect dataframe
        effect_df = pd.DataFrame(
            effect_long['effect_score'].values,
            index=master_multi_idx,
            columns=['effect_score']
        )

        # 5. Get the cell line indices for vectorized operations
        cell_indices = pd.Series(range(len(subset_dict['expression_data'].index)),
                                 index=subset_dict['expression_data'].index)
        repeat_indices = cell_indices[master_multi_idx.get_level_values('cell_line')].values

        # 6. Transform expression and mutation data using vectorized indexing
        expression_df = pd.DataFrame(
            subset_dict['expression_data'].values[repeat_indices],
            index=master_multi_idx,
            columns=subset_dict['expression_data'].columns
        )

        mutation_df = pd.DataFrame(
            subset_dict['mutation_data'].values[repeat_indices],
            index=master_multi_idx,
            columns=subset_dict['mutation_data'].columns
        )

        # 7. Transform embeddings data
        # Get the gene indices for vectorized operations
        gene_indices = pd.Series(range(len(embeddings_subset)), index=embeddings_subset.index)
        gene_repeat_indices = gene_indices[master_multi_idx.get_level_values('target_gene')].values

        embedding_df = pd.DataFrame(
            embeddings_subset.values[gene_repeat_indices],
            index=master_multi_idx,
            columns=embeddings_subset.columns
        )

        # Verify all dataframes have the same index
        assert all(df.index.equals(master_multi_idx) for df in
                   [effect_df, expression_df, mutation_df, embedding_df]), "Index mismatch"

        print("\nFinal dataframe shapes:")
        print(f"Effect data: {effect_df.shape}")
        print(f"Expression data: {expression_df.shape}")
        print(f"Mutation data: {mutation_df.shape}")
        print(f"Embedding data: {embedding_df.shape}")

        return {
            'effect_data': effect_df,
            'expression_data': expression_df,
            'mutation_data': mutation_df,
            'embedding_data': embedding_df
        }

    @staticmethod
    def subset_data(data_dict, sequences):
        """Subset data from load_cancer_data and sequences from load_sequences to the same genes"""

        common_genes = set(sequences.keys()).intersection(set(list(data_dict['expression_data'].columns)))

        return {
            'expression_data': data_dict['expression_data'].reindex(columns=sorted(common_genes)),
            'mutation_data': data_dict['mutation_data'].reindex(columns=sorted(common_genes)),
            'effect_data': data_dict['effect_data'].reindex(columns=sorted(common_genes)),
            'annotations': data_dict['annotations'],
            'sequences': sequences
        }

