#!/usr/bin/env python3
"""Fair comparison benchmark using identical sample sets."""

import json
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_embeddings(file_paths, model_name):
    """Load embeddings from multiple files."""
    embeddings = {}

    for file_path in file_paths:
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue

                    entry = json.loads(line.strip())
                    patent_id = entry.get('id') or entry.get('patent_id')
                    embedding = entry.get('embedding')

                    # Handle nested structure for other models
                    if not embedding and 'models' in entry:
                        models_data = entry.get('models', {})
                        if model_name in models_data:
                            embeddings_data = models_data[model_name].get('embeddings', {})
                            if 'original' in embeddings_data:
                                embedding = embeddings_data['original'].get('embedding')

                    if patent_id and embedding:
                        embeddings[patent_id] = np.array(embedding)

        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")
            continue

    return embeddings

def fair_comparison():
    """Run fair comparison on identical samples."""

    # Load ground truth
    logger.info("Loading ground truth...")
    ground_truth = []
    with open("data/ground_truth/consolidated/ground_truth_10k.jsonl", 'r') as f:
        for line in f:
            ground_truth.append(json.loads(line.strip()))

    logger.info(f"Loaded {len(ground_truth)} ground truth pairs")

    # Load OpenAI embeddings
    logger.info("Loading OpenAI embeddings...")
    openai_embeddings = {}
    with open("results/openai_embeddings_ground_truth.jsonl", 'r') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line.strip())
            openai_embeddings[data['id']] = np.array(data['embedding'])

    logger.info(f"Loaded {len(openai_embeddings)} OpenAI embeddings")

    # Load nomic embeddings
    logger.info("Loading nomic-embed-text embeddings...")
    nomic_files = [
        "data/embeddings/by_model/nomic-embed-text/production_100k_remaining_nomic-embed-text_20250912_205647_nomic-embed-text_embeddings.jsonl",
        "data/embeddings/by_model/nomic-embed-text/production_10k_all_nomic-embed-text_20250912_150410_nomic-embed-text_embeddings.jsonl",
        "data/embeddings/by_model/nomic-embed-text/diverse_10k_full_nomic-embed-text_20250912_070401_nomic-embed-text_embeddings.jsonl",
        "data/embeddings/by_model/nomic-embed-text/production_100k_top2_nomic-embed-text_20250912_150417_nomic-embed-text_embeddings.jsonl",
        "data/embeddings/by_model/nomic-embed-text/original_500_nomic-embed-text_20250912_070406_nomic-embed-text_embeddings.jsonl",
        "data/embeddings/by_model/nomic-embed-text/production_top2_nomic-embed-text_20250912_070417_nomic-embed-text_embeddings.jsonl",
        "data/embeddings/by_model/nomic-embed-text/validation_nomic-embed-text_20250912_065708_nomic-embed-text_embeddings.jsonl"
    ]

    nomic_embeddings = load_embeddings(nomic_files, 'nomic-embed-text')
    logger.info(f"Loaded {len(nomic_embeddings)} nomic embeddings")

    # Find pairs that both models can handle
    logger.info("Finding overlapping pairs...")
    valid_pairs = []
    openai_similarities = []
    nomic_similarities = []
    llm_scores = []

    for gt_pair in ground_truth:
        patent1_id = gt_pair['patent1_id']
        patent2_id = gt_pair['patent2_id']
        llm_score = gt_pair['llm_analysis']['similarity_score']

        # Only include if both models have embeddings for both patents
        if (patent1_id in openai_embeddings and patent2_id in openai_embeddings and
            patent1_id in nomic_embeddings and patent2_id in nomic_embeddings):

            # Calculate OpenAI similarity
            openai_emb1 = openai_embeddings[patent1_id]
            openai_emb2 = openai_embeddings[patent2_id]
            openai_sim = cosine_similarity([openai_emb1], [openai_emb2])[0][0]

            # Calculate nomic similarity
            nomic_emb1 = nomic_embeddings[patent1_id]
            nomic_emb2 = nomic_embeddings[patent2_id]
            nomic_sim = cosine_similarity([nomic_emb1], [nomic_emb2])[0][0]

            valid_pairs.append((patent1_id, patent2_id))
            openai_similarities.append(openai_sim)
            nomic_similarities.append(nomic_sim)
            llm_scores.append(llm_score)

    logger.info(f"Found {len(valid_pairs)} pairs with both embeddings")

    if len(valid_pairs) < 100:
        logger.error("Insufficient overlapping pairs for fair comparison!")
        return None

    # Calculate correlations for both models on identical samples
    openai_pearson_r, openai_pearson_p = pearsonr(openai_similarities, llm_scores)
    openai_spearman_r, openai_spearman_p = spearmanr(openai_similarities, llm_scores)

    nomic_pearson_r, nomic_pearson_p = pearsonr(nomic_similarities, llm_scores)
    nomic_spearman_r, nomic_spearman_p = spearmanr(nomic_similarities, llm_scores)

    # Results
    results = {
        'sample_size': len(valid_pairs),
        'openai': {
            'pearson_r': openai_pearson_r,
            'pearson_p': openai_pearson_p,
            'spearman_r': openai_spearman_r,
            'spearman_p': openai_spearman_p,
            'mean_similarity': np.mean(openai_similarities),
            'std_similarity': np.std(openai_similarities)
        },
        'nomic': {
            'pearson_r': nomic_pearson_r,
            'pearson_p': nomic_pearson_p,
            'spearman_r': nomic_spearman_r,
            'spearman_p': nomic_spearman_p,
            'mean_similarity': np.mean(nomic_similarities),
            'std_similarity': np.std(nomic_similarities)
        },
        'llm_scores': {
            'mean': np.mean(llm_scores),
            'std': np.std(llm_scores)
        }
    }

    return results

def print_results(results):
    """Print fair comparison results."""
    if not results:
        print("❌ Fair comparison failed!")
        return

    print("=" * 80)
    print("🎯 FAIR COMPARISON RESULTS (IDENTICAL SAMPLES)")
    print("=" * 80)
    print(f"Sample size: {results['sample_size']:,} pairs")
    print()

    print("| Model | Pearson r | Spearman ρ | p-value | Mean Sim | Std Sim |")
    print("|-------|-----------|------------|---------|----------|---------|")

    openai = results['openai']
    nomic = results['nomic']

    print(f"| **text-embedding-3-small** | {openai['pearson_r']:.4f} | {openai['spearman_r']:.4f} | {openai['pearson_p']:.2e} | {openai['mean_similarity']:.3f} | {openai['std_similarity']:.3f} |")
    print(f"| **nomic-embed-text** | {nomic['pearson_r']:.4f} | {nomic['spearman_r']:.4f} | {nomic['pearson_p']:.2e} | {nomic['mean_similarity']:.3f} | {nomic['std_similarity']:.3f} |")

    print()
    print("🏆 WINNER:")
    if openai['pearson_r'] > nomic['pearson_r']:
        diff = openai['pearson_r'] - nomic['pearson_r']
        improvement = (diff / nomic['pearson_r']) * 100
        print(f"   text-embedding-3-small (r = {openai['pearson_r']:.4f})")
        print(f"   Improvement: +{diff:.4f} ({improvement:.1f}% better)")
    else:
        diff = nomic['pearson_r'] - openai['pearson_r']
        improvement = (diff / openai['pearson_r']) * 100
        print(f"   nomic-embed-text (r = {nomic['pearson_r']:.4f})")
        print(f"   Improvement: +{diff:.4f} ({improvement:.1f}% better)")

    # Save results
    with open('results/fair_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)

    print(f"\\n📊 Results saved to results/fair_comparison_results.json")

if __name__ == "__main__":
    print("🔬 Running fair comparison benchmark...")
    results = fair_comparison()
    print_results(results)