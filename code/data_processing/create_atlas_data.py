#!/usr/bin/env python3
"""
Create Atlas-Compatible Parquet Data

Generates parquet files from our consolidated data for Apple Embedding Atlas visualization.
Uses patents that have embeddings from multiple models for the most interesting visualization.
"""

import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.decomposition import PCA

# Add utilities import
import sys
sys.path.append(str(Path(__file__).parent.parent / "utilities"))
from file_utils import smart_jsonl_reader
from sklearn.manifold import TSNE
import umap

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_multi_model_patents(master_file: str, min_models: int = 2) -> List[Dict]:
    """
    Load patents that have embeddings from multiple models.

    Args:
        master_file: Path to master embeddings file
        min_models: Minimum number of models required per patent

    Returns:
        List of patent records with multiple models
    """
    logger.info(f"Loading patents with at least {min_models} models...")

    multi_model_patents = []

    with smart_jsonl_reader(master_file) as jsonl_data:
        for line_num, data in enumerate(jsonl_data, 1):
            embeddings = data.get('embeddings', {})

            if len(embeddings) >= min_models:
                # Add some basic text for visualization if available
                text = data.get('abstract', '')[:500] or f"Patent {data.get('patent_id', 'unknown')}"

                patent_record = {
                    'patent_id': data.get('patent_id', ''),
                    'text': text,
                    'classification': data.get('classification', ''),
                    'embeddings': embeddings,
                    'model_count': len(embeddings),
                    'has_all_three': len(embeddings) == 3
                }
                multi_model_patents.append(patent_record)

    logger.info(f"Loaded {len(multi_model_patents)} patents with multiple models")
    return multi_model_patents


def create_atlas_datasets(patents: List[Dict]) -> Dict[str, pd.DataFrame]:
    """
    Create different datasets for Atlas visualization.

    Args:
        patents: List of patent records

    Returns:
        Dictionary of DataFrames for different visualization scenarios
    """
    datasets = {}

    # Dataset 1: OpenAI embeddings for patents that have all three models
    logger.info("Creating OpenAI embeddings dataset...")
    all_three_patents = [p for p in patents if p['has_all_three']]

    if all_three_patents:
        openai_data = []
        for patent in all_three_patents:
            if 'openai_text-embedding-3-small' in patent['embeddings']:
                embedding = patent['embeddings']['openai_text-embedding-3-small']['vector']
                openai_data.append({
                    'patent_id': patent['patent_id'],
                    'text': patent['text'],
                    'classification': patent['classification'],
                    'embedding': embedding,
                    'model': 'OpenAI',
                    'model_count': patent['model_count']
                })

        if openai_data:
            df_openai = pd.DataFrame(openai_data)

            # Create UMAP projection for 2D visualization
            logger.info(f"Creating UMAP projection for {len(df_openai)} OpenAI embeddings...")
            embeddings_matrix = np.array(df_openai['embedding'].tolist())

            # Use UMAP for visualization
            umap_reducer = umap.UMAP(n_components=2, random_state=42, metric='cosine')
            umap_result = umap_reducer.fit_transform(embeddings_matrix)

            df_openai['umap_x'] = umap_result[:, 0]
            df_openai['umap_y'] = umap_result[:, 1]

            # Add PCA as backup
            if embeddings_matrix.shape[1] > 50:  # Only if high dimensional
                pca = PCA(n_components=50)
                pca_embeddings = pca.fit_transform(embeddings_matrix)
                df_openai['pca_variance_explained'] = pca.explained_variance_ratio_[:2].sum()

            datasets['openai_all_three'] = df_openai

    # Dataset 2: Nomic embeddings for broader coverage
    logger.info("Creating nomic embeddings dataset...")
    nomic_data = []

    # Take a sample of patents with nomic embeddings (limit for performance)
    nomic_patents = [p for p in patents if 'nomic-embed-text' in p['embeddings']][:5000]

    for patent in nomic_patents:
        embedding = patent['embeddings']['nomic-embed-text']['vector']
        nomic_data.append({
            'patent_id': patent['patent_id'],
            'text': patent['text'],
            'classification': patent['classification'],
            'embedding': embedding,
            'model': 'nomic',
            'model_count': patent['model_count'],
            'has_all_three': patent['has_all_three']
        })

    if nomic_data:
        df_nomic = pd.DataFrame(nomic_data)

        logger.info(f"Creating UMAP projection for {len(df_nomic)} nomic embeddings...")
        embeddings_matrix = np.array(df_nomic['embedding'].tolist())

        umap_reducer = umap.UMAP(n_components=2, random_state=42, metric='cosine')
        umap_result = umap_reducer.fit_transform(embeddings_matrix)

        df_nomic['umap_x'] = umap_result[:, 0]
        df_nomic['umap_y'] = umap_result[:, 1]

        datasets['nomic_sample'] = df_nomic

    # Dataset 3: Model comparison for patents with all three
    logger.info("Creating model comparison dataset...")
    if all_three_patents:
        comparison_data = []

        for patent in all_three_patents[:1000]:  # Limit for performance
            for model_name, model_data in patent['embeddings'].items():
                comparison_data.append({
                    'patent_id': patent['patent_id'],
                    'text': patent['text'],
                    'classification': patent['classification'],
                    'embedding': model_data['vector'],
                    'model': model_name.split('_')[0] if '_' in model_name else model_name.split('-')[0],
                    'dimension': model_data['dimension'],
                    'model_full_name': model_name
                })

        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)

            # Create UMAP for each model separately, then combine
            models = df_comparison['model'].unique()

            all_umap_results = []
            for model in models:
                model_df = df_comparison[df_comparison['model'] == model].copy()
                embeddings_matrix = np.array(model_df['embedding'].tolist())

                logger.info(f"Creating UMAP for {model} ({len(model_df)} embeddings)...")

                # Adjust UMAP parameters based on embedding dimensions
                n_neighbors = min(15, len(model_df) - 1)
                umap_reducer = umap.UMAP(
                    n_components=2,
                    random_state=42,
                    metric='cosine',
                    n_neighbors=n_neighbors
                )
                umap_result = umap_reducer.fit_transform(embeddings_matrix)

                model_df['umap_x'] = umap_result[:, 0]
                model_df['umap_y'] = umap_result[:, 1]

                all_umap_results.append(model_df)

            df_comparison = pd.concat(all_umap_results, ignore_index=True)
            datasets['model_comparison'] = df_comparison

    return datasets


def save_atlas_datasets(datasets: Dict[str, pd.DataFrame], output_dir: str) -> Dict[str, str]:
    """
    Save datasets as parquet files for Atlas.

    Args:
        datasets: Dictionary of DataFrames
        output_dir: Output directory path

    Returns:
        Dictionary mapping dataset names to file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    for dataset_name, df in datasets.items():
        # Remove the embedding column for parquet (keep UMAP coordinates)
        df_for_atlas = df.drop('embedding', axis=1)

        file_path = output_path / f"{dataset_name}_atlas.parquet"
        df_for_atlas.to_parquet(file_path, index=False)

        logger.info(f"Saved {len(df_for_atlas)} records to {file_path}")
        saved_files[dataset_name] = str(file_path)

        # Print sample
        print(f"\n📊 {dataset_name.upper()} DATASET:")
        print(f"   Records: {len(df_for_atlas)}")
        print(f"   Columns: {list(df_for_atlas.columns)}")
        if 'model' in df_for_atlas.columns:
            print(f"   Models: {df_for_atlas['model'].value_counts().to_dict()}")

    return saved_files


def update_launch_script(best_dataset: str, output_dir: str):
    """
    Update the Atlas launch script to use the best dataset.

    Args:
        best_dataset: Name of the best dataset to use as default
        output_dir: Directory containing the datasets
    """
    script_content = f'''#!/bin/bash
# Launch Apple Embedding Atlas for Patent Research
# Updated for consolidated data format

echo "Launching Apple Embedding Atlas..."
echo "Dataset: {best_dataset}_atlas.parquet"
echo "Data directory: {output_dir}"
echo ""
echo "Available datasets:"
echo "  - openai_all_three_atlas.parquet (OpenAI embeddings, patents with all 3 models)"
echo "  - nomic_sample_atlas.parquet (nomic embeddings, broad sample)"
echo "  - model_comparison_atlas.parquet (All models side by side)"
echo ""
echo "Atlas will open in your default browser"
echo "Use Ctrl+C to stop the server when done"
echo ""

cd "{output_dir}" || exit 1

# Launch with optimal parameters for patent data
embedding-atlas \\
    --data "{best_dataset}_atlas.parquet" \\
    --text "text" \\
    --x "umap_x" \\
    --y "umap_y" \\
    --color "classification" \\
    --host "localhost" \\
    --port 8000

echo ""
echo "Atlas server stopped"
'''

    script_path = Path("scripts/launch_atlas.sh")
    with open(script_path, 'w') as f:
        f.write(script_content)

    # Make executable
    script_path.chmod(0o755)

    logger.info(f"Updated launch script: {script_path}")


def main():
    """Main execution"""
    master_file = "data_v2/master_patent_embeddings.jsonl"
    output_dir = "data_v2/atlas_data"

    if not Path(master_file).exists():
        print(f"❌ Master embeddings file not found: {master_file}")
        return

    try:
        # Load patents with multiple models
        patents = load_multi_model_patents(master_file, min_models=2)

        if not patents:
            print("❌ No patents found with multiple models")
            return

        # Create Atlas datasets
        datasets = create_atlas_datasets(patents)

        if not datasets:
            print("❌ No datasets created")
            return

        # Save datasets
        saved_files = save_atlas_datasets(datasets, output_dir)

        # Determine best default dataset
        best_dataset = 'openai_all_three' if 'openai_all_three' in datasets else list(datasets.keys())[0]

        # Update launch script
        update_launch_script(best_dataset, output_dir)

        print("\n" + "="*60)
        print("🎯 ATLAS DATA CREATION COMPLETE!")
        print("="*60)
        print(f"📁 Output directory: {output_dir}")
        print(f"📊 Datasets created: {len(datasets)}")
        print(f"🎨 Default dataset: {best_dataset}")
        print("\n🚀 To launch Atlas:")
        print("   bash scripts/launch_atlas.sh")
        print("\n📋 Available datasets:")
        for name, path in saved_files.items():
            print(f"   - {name}: {path}")
        print("="*60)

    except Exception as e:
        logger.error(f"Failed to create Atlas data: {e}")
        raise


if __name__ == "__main__":
    main()