#!/usr/bin/env python3
"""Three-way fair comparison using identical sample sets for all three models."""

import json
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_all_embeddings():
    """Load embeddings for all three models."""
    # Load OpenAI embeddings
    openai_embeddings = {}
    with open("results/openai_embeddings_ground_truth.jsonl", 'r') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line.strip())
            openai_embeddings[data['id']] = np.array(data['embedding'])

    # Load nomic embeddings
    nomic_files = [
        "data/embeddings/by_model/nomic-embed-text/production_100k_remaining_nomic-embed-text_20250912_205647_nomic-embed-text_embeddings.jsonl",
        "data/embeddings/by_model/nomic-embed-text/production_10k_all_nomic-embed-text_20250912_150410_nomic-embed-text_embeddings.jsonl",
        "data/embeddings/by_model/nomic-embed-text/diverse_10k_full_nomic-embed-text_20250912_070401_nomic-embed-text_embeddings.jsonl",
        "data/embeddings/by_model/nomic-embed-text/production_100k_top2_nomic-embed-text_20250912_150417_nomic-embed-text_embeddings.jsonl",
        "data/embeddings/by_model/nomic-embed-text/original_500_nomic-embed-text_20250912_070406_nomic-embed-text_embeddings.jsonl",
        "data/embeddings/by_model/nomic-embed-text/production_top2_nomic-embed-text_20250912_070417_nomic-embed-text_embeddings.jsonl",
        "data/embeddings/by_model/nomic-embed-text/validation_nomic-embed-text_20250912_065708_nomic-embed-text_embeddings.jsonl"
    ]

    nomic_embeddings = {}
    for file_path in nomic_files:
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line.strip())
                    patent_id = entry.get('id') or entry.get('patent_id')
                    embedding = entry.get('embedding')

                    # Handle nested structure
                    if not embedding and 'models' in entry:
                        models_data = entry.get('models', {})
                        if 'nomic-embed-text' in models_data:
                            embeddings_data = models_data['nomic-embed-text'].get('embeddings', {})
                            if 'original' in embeddings_data:
                                embedding = embeddings_data['original'].get('embedding')

                    if patent_id and embedding:
                        nomic_embeddings[patent_id] = np.array(embedding)
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")
            continue

    # Load bge-m3 embeddings from bge-m3 specific files
    bge_files = [
        "data/embeddings/by_model/bge-m3/diverse_10k_full_bge-m3_20250912_070401_bge-m3_embeddings.jsonl",
        "data/embeddings/by_model/bge-m3/original_500_bge-m3_20250912_070406_bge-m3_embeddings.jsonl",
        "data/embeddings/by_model/bge-m3/production_100k_top2_bge-m3_20250912_150417_bge-m3_embeddings.jsonl",
        "data/embeddings/by_model/bge-m3/production_10k_all_bge-m3_20250912_150410_bge-m3_embeddings.jsonl",
        "data/embeddings/by_model/bge-m3/production_top2_bge-m3_20250912_070417_bge-m3_embeddings.jsonl",
        "data/embeddings/by_model/bge-m3/validation_bge-m3_20250912_065708_bge-m3_embeddings.jsonl"
    ]

    bge_embeddings = {}
    for file_path in bge_files:
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line.strip())
                    patent_id = entry.get('id') or entry.get('patent_id')
                    embedding = entry.get('embedding')

                    # Handle nested structure for bge-m3
                    if not embedding and 'models' in entry:
                        models_data = entry.get('models', {})
                        if 'bge-m3' in models_data:
                            embeddings_data = models_data['bge-m3'].get('embeddings', {})
                            if 'original' in embeddings_data:
                                embedding = embeddings_data['original'].get('embedding')

                    if patent_id and embedding:
                        bge_embeddings[patent_id] = np.array(embedding)
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")
            continue

    logger.info(f"Loaded {len(openai_embeddings)} OpenAI embeddings")
    logger.info(f"Loaded {len(nomic_embeddings)} nomic embeddings")
    logger.info(f"Loaded {len(bge_embeddings)} bge-m3 embeddings")

    return openai_embeddings, nomic_embeddings, bge_embeddings

def three_way_fair_comparison():
    """Run three-way fair comparison on identical samples."""

    # Load ground truth
    logger.info("Loading ground truth...")
    ground_truth = []
    with open("data/ground_truth/consolidated/ground_truth_10k.jsonl", 'r') as f:
        for line in f:
            ground_truth.append(json.loads(line.strip()))

    logger.info(f"Loaded {len(ground_truth)} ground truth pairs")

    # Load all embeddings
    openai_embeddings, nomic_embeddings, bge_embeddings = load_all_embeddings()

    # Find pairs that ALL THREE models can handle
    logger.info("Finding three-way overlapping pairs...")
    valid_pairs = []
    openai_similarities = []
    nomic_similarities = []
    bge_similarities = []
    llm_scores = []

    for gt_pair in ground_truth:
        patent1_id = gt_pair['patent1_id']
        patent2_id = gt_pair['patent2_id']
        llm_score = gt_pair['llm_analysis']['similarity_score']

        # Only include if ALL THREE models have embeddings for both patents
        if (patent1_id in openai_embeddings and patent2_id in openai_embeddings and
            patent1_id in nomic_embeddings and patent2_id in nomic_embeddings and
            patent1_id in bge_embeddings and patent2_id in bge_embeddings):

            # Calculate OpenAI similarity
            openai_emb1 = openai_embeddings[patent1_id]
            openai_emb2 = openai_embeddings[patent2_id]
            openai_sim = cosine_similarity([openai_emb1], [openai_emb2])[0][0]

            # Calculate nomic similarity
            nomic_emb1 = nomic_embeddings[patent1_id]
            nomic_emb2 = nomic_embeddings[patent2_id]
            nomic_sim = cosine_similarity([nomic_emb1], [nomic_emb2])[0][0]

            # Calculate bge-m3 similarity
            bge_emb1 = bge_embeddings[patent1_id]
            bge_emb2 = bge_embeddings[patent2_id]
            bge_sim = cosine_similarity([bge_emb1], [bge_emb2])[0][0]

            valid_pairs.append((patent1_id, patent2_id))
            openai_similarities.append(openai_sim)
            nomic_similarities.append(nomic_sim)
            bge_similarities.append(bge_sim)
            llm_scores.append(llm_score)

    logger.info(f"Found {len(valid_pairs)} pairs with ALL THREE model embeddings")

    if len(valid_pairs) < 100:
        logger.error("Insufficient overlapping pairs for three-way fair comparison!")
        return None

    # Calculate correlations for all three models on identical samples
    openai_pearson_r, openai_pearson_p = pearsonr(openai_similarities, llm_scores)
    openai_spearman_r, openai_spearman_p = spearmanr(openai_similarities, llm_scores)

    nomic_pearson_r, nomic_pearson_p = pearsonr(nomic_similarities, llm_scores)
    nomic_spearman_r, nomic_spearman_p = spearmanr(nomic_similarities, llm_scores)

    bge_pearson_r, bge_pearson_p = pearsonr(bge_similarities, llm_scores)
    bge_spearman_r, bge_spearman_p = spearmanr(bge_similarities, llm_scores)

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
        'bge_m3': {
            'pearson_r': bge_pearson_r,
            'pearson_p': bge_pearson_p,
            'spearman_r': bge_spearman_r,
            'spearman_p': bge_spearman_p,
            'mean_similarity': np.mean(bge_similarities),
            'std_similarity': np.std(bge_similarities)
        },
        'llm_scores': {
            'mean': np.mean(llm_scores),
            'std': np.std(llm_scores)
        },
        'valid_pairs': valid_pairs[:10]  # Store first 10 for verification
    }

    return results

def print_results(results):
    """Print three-way fair comparison results."""
    if not results:
        print("❌ Three-way fair comparison failed!")
        return

    print("=" * 80)
    print("🎯 THREE-WAY FAIR COMPARISON RESULTS (IDENTICAL SAMPLES)")
    print("=" * 80)
    print(f"Sample size: {results['sample_size']:,} pairs")
    print()

    print("| Model | Pearson r | Spearman ρ | p-value | Mean Sim | Std Sim |")
    print("|-------|-----------|------------|---------|----------|---------|")

    openai = results['openai']
    nomic = results['nomic']
    bge = results['bge_m3']

    print(f"| **text-embedding-3-small** | {openai['pearson_r']:.4f} | {openai['spearman_r']:.4f} | {openai['pearson_p']:.2e} | {openai['mean_similarity']:.3f} | {openai['std_similarity']:.3f} |")
    print(f"| **nomic-embed-text** | {nomic['pearson_r']:.4f} | {nomic['spearman_r']:.4f} | {nomic['pearson_p']:.2e} | {nomic['mean_similarity']:.3f} | {nomic['std_similarity']:.3f} |")
    print(f"| **bge-m3** | {bge['pearson_r']:.4f} | {bge['spearman_r']:.4f} | {bge['pearson_p']:.2e} | {bge['mean_similarity']:.3f} | {bge['std_similarity']:.3f} |")

    print()
    print("🏆 RANKING:")

    # Sort by pearson correlation
    models = [
        ('text-embedding-3-small', openai['pearson_r']),
        ('nomic-embed-text', nomic['pearson_r']),
        ('bge-m3', bge['pearson_r'])
    ]
    models.sort(key=lambda x: x[1], reverse=True)

    for i, (model, corr) in enumerate(models):
        if i == 0:
            print(f"   🥇 {model} (r = {corr:.4f}) - WINNER")
        elif i == 1:
            print(f"   🥈 {model} (r = {corr:.4f})")
        else:
            print(f"   🥉 {model} (r = {corr:.4f})")

    # Calculate improvements
    winner_r = models[0][1]
    second_r = models[1][1]
    third_r = models[2][1]

    improvement_over_second = (winner_r - second_r) / second_r * 100
    improvement_over_third = (winner_r - third_r) / third_r * 100

    print(f"\n📊 PERFORMANCE GAPS:")
    print(f"   1st vs 2nd: +{improvement_over_second:.1f}% improvement")
    print(f"   1st vs 3rd: +{improvement_over_third:.1f}% improvement")

    # Save results
    with open('results/three_way_fair_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)

    print(f"\n📊 Results saved to results/three_way_fair_comparison_results.json")

if __name__ == "__main__":
    print("🔬 Running three-way fair comparison benchmark...")
    results = three_way_fair_comparison()
    print_results(results)