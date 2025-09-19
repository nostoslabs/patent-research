#!/usr/bin/env python3
"""
Create Better Atlas Data with Meaningful Categories

Since classifications are sparse (only 2.3% have them), create Atlas data with more meaningful groupings:
1. Model coverage (1, 2, or 3 models)
2. Text availability (has abstract, has full text, has neither)
3. Patent ID ranges (to see temporal or organizational patterns)
4. Embedding quality metrics
"""

import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import umap

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_meaningful_categories(patent_record: Dict) -> Dict:
    """
    Create meaningful categories for visualization since classifications are sparse.

    Args:
        patent_record: Patent data from master file

    Returns:
        Dictionary with various categorical assignments
    """
    patent_id = patent_record.get('patent_id', '')
    embeddings = patent_record.get('embeddings', {})
    abstract = patent_record.get('abstract', '')
    full_text = patent_record.get('full_text', '')
    classification = patent_record.get('classification', '')

    # Extract patent number if possible
    patent_num = 0
    if patent_id.startswith('patent_'):
        try:
            patent_num = int(patent_id.replace('patent_', ''))
        except:
            patent_num = 0

    categories = {
        # Model coverage
        'model_count': len(embeddings),
        'model_coverage': f"{len(embeddings)} model{'s' if len(embeddings) != 1 else ''}",

        # Text availability
        'text_quality': 'No text',

        # Patent ranges (potentially temporal/organizational)
        'patent_range': 'Unknown',

        # Original classification (if available)
        'classification': classification if classification else 'Unclassified',

        # Has all three models flag
        'complete_coverage': 'Complete' if len(embeddings) == 3 else 'Partial',

        # Text length category
        'text_length_category': 'No text'
    }

    # Determine text quality
    if abstract:
        if len(abstract) > 100:
            categories['text_quality'] = 'Rich abstract'
        else:
            categories['text_quality'] = 'Brief abstract'
    elif full_text:
        categories['text_quality'] = 'Full text only'

    # Patent number ranges (might indicate different sources/batches)
    if patent_num > 0:
        if patent_num < 1000:
            categories['patent_range'] = '0-1K'
        elif patent_num < 5000:
            categories['patent_range'] = '1K-5K'
        elif patent_num < 10000:
            categories['patent_range'] = '5K-10K'
        elif patent_num < 20000:
            categories['patent_range'] = '10K-20K'
        elif patent_num < 30000:
            categories['patent_range'] = '20K-30K'
        else:
            categories['patent_range'] = '30K+'

    # Text length categories
    total_text = (abstract + ' ' + full_text).strip()
    text_length = len(total_text)

    if text_length == 0:
        categories['text_length_category'] = 'No text'
    elif text_length < 100:
        categories['text_length_category'] = 'Very short'
    elif text_length < 500:
        categories['text_length_category'] = 'Short'
    elif text_length < 2000:
        categories['text_length_category'] = 'Medium'
    elif text_length < 5000:
        categories['text_length_category'] = 'Long'
    else:
        categories['text_length_category'] = 'Very long'

    return categories


def create_enhanced_atlas_datasets(master_file: str) -> Dict[str, pd.DataFrame]:
    """
    Create enhanced Atlas datasets with meaningful categories.

    Args:
        master_file: Path to master embeddings file

    Returns:
        Dictionary of enhanced DataFrames
    """
    logger.info("Creating enhanced Atlas datasets...")

    datasets = {}

    # Dataset 1: Patents with all three models (colored by patent range)
    logger.info("Creating complete coverage dataset...")
    all_three_data = []

    # Dataset 2: OpenAI vs nomic comparison (colored by model)
    logger.info("Creating model comparison dataset...")
    comparison_data = []

    # Dataset 3: Text quality analysis (colored by text availability)
    logger.info("Creating text quality dataset...")
    text_quality_data = []

    # Process master file
    with open(master_file, 'r') as f:
        processed_count = 0
        for line in f:
            if line.strip():
                processed_count += 1
                data = json.loads(line.strip())
                embeddings = data.get('embeddings', {})

                # Create categories
                categories = create_meaningful_categories(data)

                # Get text for display
                display_text = data.get('abstract', data.get('full_text', ''))[:500]
                if not display_text:
                    display_text = f"Patent {data.get('patent_id', 'unknown')}"

                # Dataset 1: All three models
                if len(embeddings) == 3 and 'openai_text-embedding-3-small' in embeddings:
                    all_three_data.append({
                        'patent_id': data.get('patent_id', ''),
                        'text': display_text,
                        'embedding': embeddings['openai_text-embedding-3-small']['vector'],
                        'patent_range': categories['patent_range'],
                        'text_quality': categories['text_quality'],
                        'text_length_category': categories['text_length_category'],
                        'classification': categories['classification']
                    })

                # Dataset 2: Model comparison (patents with 2+ models)
                if len(embeddings) >= 2:
                    # Add OpenAI if available
                    if 'openai_text-embedding-3-small' in embeddings:
                        comparison_data.append({
                            'patent_id': data.get('patent_id', ''),
                            'text': display_text,
                            'embedding': embeddings['openai_text-embedding-3-small']['vector'],
                            'model': 'OpenAI',
                            'patent_range': categories['patent_range'],
                            'model_coverage': categories['model_coverage'],
                            'complete_coverage': categories['complete_coverage']
                        })

                    # Add nomic if available
                    if 'nomic-embed-text' in embeddings:
                        comparison_data.append({
                            'patent_id': data.get('patent_id', ''),
                            'text': display_text,
                            'embedding': embeddings['nomic-embed-text']['vector'],
                            'model': 'Nomic',
                            'patent_range': categories['patent_range'],
                            'model_coverage': categories['model_coverage'],
                            'complete_coverage': categories['complete_coverage']
                        })

                # Dataset 3: Text quality (sample for performance)
                if len(text_quality_data) < 8000 and 'openai_text-embedding-3-small' in embeddings:
                    text_quality_data.append({
                        'patent_id': data.get('patent_id', ''),
                        'text': display_text,
                        'embedding': embeddings['openai_text-embedding-3-small']['vector'],
                        'text_quality': categories['text_quality'],
                        'text_length_category': categories['text_length_category'],
                        'patent_range': categories['patent_range']
                    })

                if processed_count % 5000 == 0:
                    logger.info(f"Processed {processed_count:,} patents...")

                # Limit for performance
                if len(all_three_data) > 5000:
                    break

    logger.info(f"Collected data: all_three={len(all_three_data)}, comparison={len(comparison_data)}, text_quality={len(text_quality_data)}")

    # Create DataFrames and UMAP projections
    if all_three_data:
        df_all_three = pd.DataFrame(all_three_data)
        embeddings_matrix = np.array(df_all_three['embedding'].tolist())

        logger.info(f"Creating UMAP for complete coverage dataset ({len(df_all_three)} patents)...")
        umap_reducer = umap.UMAP(n_components=2, random_state=42, metric='cosine')
        umap_result = umap_reducer.fit_transform(embeddings_matrix)

        df_all_three['umap_x'] = umap_result[:, 0]
        df_all_three['umap_y'] = umap_result[:, 1]
        df_all_three = df_all_three.drop('embedding', axis=1)  # Remove for parquet

        datasets['complete_coverage'] = df_all_three

    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)

        # Create UMAP for each model separately
        models = df_comparison['model'].unique()
        all_results = []

        for model in models:
            model_df = df_comparison[df_comparison['model'] == model].copy()
            if len(model_df) > 10:  # Only process if enough data
                embeddings_matrix = np.array(model_df['embedding'].tolist())

                logger.info(f"Creating UMAP for {model} comparison ({len(model_df)} patents)...")
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
                all_results.append(model_df)

        if all_results:
            df_comparison = pd.concat(all_results, ignore_index=True)
            df_comparison = df_comparison.drop('embedding', axis=1)  # Remove for parquet
            datasets['model_comparison'] = df_comparison

    if text_quality_data:
        df_text_quality = pd.DataFrame(text_quality_data)
        embeddings_matrix = np.array(df_text_quality['embedding'].tolist())

        logger.info(f"Creating UMAP for text quality dataset ({len(df_text_quality)} patents)...")
        umap_reducer = umap.UMAP(n_components=2, random_state=42, metric='cosine')
        umap_result = umap_reducer.fit_transform(embeddings_matrix)

        df_text_quality['umap_x'] = umap_result[:, 0]
        df_text_quality['umap_y'] = umap_result[:, 1]
        df_text_quality = df_text_quality.drop('embedding', axis=1)  # Remove for parquet

        datasets['text_quality'] = df_text_quality

    return datasets


def save_enhanced_datasets(datasets: Dict[str, pd.DataFrame], output_dir: str):
    """Save enhanced datasets and create launch script"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files = []

    for dataset_name, df in datasets.items():
        file_path = output_path / f"{dataset_name}_enhanced_atlas.parquet"
        df.to_parquet(file_path, index=False)
        saved_files.append(str(file_path))

        logger.info(f"Saved {len(df)} records to {file_path}")

        # Print dataset info
        print(f"\n📊 {dataset_name.upper().replace('_', ' ')} DATASET:")
        print(f"   Records: {len(df)}")
        print(f"   Columns: {list(df.columns)}")

        # Show category distributions
        for col in df.columns:
            if col not in ['patent_id', 'text', 'umap_x', 'umap_y'] and df[col].dtype == 'object':
                if df[col].nunique() <= 10:  # Only show if reasonable number of categories
                    print(f"   {col}: {dict(df[col].value_counts().head())}")

    return saved_files


def main():
    """Main execution"""
    master_file = "data_v2/master_patent_embeddings.jsonl"
    output_dir = "data_v2/atlas_data"

    if not Path(master_file).exists():
        print(f"❌ Master embeddings file not found: {master_file}")
        return

    try:
        # Create enhanced datasets
        datasets = create_enhanced_atlas_datasets(master_file)

        if not datasets:
            print("❌ No datasets created")
            return

        # Save datasets
        saved_files = save_enhanced_datasets(datasets, output_dir)

        print("\n" + "="*60)
        print("🎯 ENHANCED ATLAS DATA CREATED!")
        print("="*60)
        print("Now you have meaningful visualizations:")
        print("1. 'complete_coverage' - Patents with all 3 models, colored by patent ranges")
        print("2. 'model_comparison' - OpenAI vs Nomic side-by-side")
        print("3. 'text_quality' - Patents colored by text availability")
        print("\n🚀 Launch Atlas with:")
        print("uv run embedding-atlas complete_coverage_enhanced_atlas.parquet --text text --x umap_x --y umap_y --host localhost --port 8001")
        print("="*60)

    except Exception as e:
        logger.error(f"Failed to create enhanced Atlas data: {e}")
        raise


if __name__ == "__main__":
    main()