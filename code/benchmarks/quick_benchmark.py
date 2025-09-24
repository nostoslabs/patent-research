"""Quick benchmark to get nomic-embed-text results."""

import json
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity


def quick_benchmark():
    """Quick benchmark for nomic-embed-text."""

    # Load ground truth
    ground_truth = []
    with open("data/ground_truth/consolidated/ground_truth_10k.jsonl", 'r') as f:
        for line in f:
            ground_truth.append(json.loads(line.strip()))

    print(f"Loaded {len(ground_truth)} ground truth pairs")

    # Load nomic embeddings from the main file that should have full coverage
    embeddings = {}
    embedding_file = "data/embeddings/by_model/nomic-embed-text/production_100k_remaining_nomic-embed-text_20250912_205647_nomic-embed-text_embeddings.jsonl"

    print(f"Loading embeddings from {embedding_file}")
    with open(embedding_file, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            patent_id = entry.get('id') or entry.get('patent_id')

            # Handle nested structure
            if 'models' in entry:
                models_data = entry.get('models', {})
                if 'nomic-embed-text' in models_data:
                    embeddings_data = models_data['nomic-embed-text'].get('embeddings', {})
                    if 'original' in embeddings_data:
                        embedding = embeddings_data['original'].get('embedding')
                        if patent_id and embedding:
                            embeddings[patent_id] = np.array(embedding)

    print(f"Loaded {len(embeddings)} embeddings")

    # Calculate similarities
    similarities = []
    llm_scores = []
    found_pairs = 0

    for gt_pair in ground_truth:
        patent1_id = gt_pair['patent1_id']
        patent2_id = gt_pair['patent2_id']
        llm_score = gt_pair['llm_analysis']['similarity_score']

        if patent1_id in embeddings and patent2_id in embeddings:
            emb1 = embeddings[patent1_id]
            emb2 = embeddings[patent2_id]

            # Calculate cosine similarity
            similarity = cosine_similarity([emb1], [emb2])[0][0]
            similarities.append(similarity)
            llm_scores.append(llm_score)
            found_pairs += 1

    print(f"Found {found_pairs}/{len(ground_truth)} pairs with embeddings")

    if len(similarities) > 10:
        # Calculate correlations
        pearson_r, pearson_p = pearsonr(similarities, llm_scores)
        spearman_r, spearman_p = spearmanr(similarities, llm_scores)

        print(f"\nNOMIC-EMBED-TEXT RESULTS:")
        print(f"Pairs analyzed: {len(similarities)}")
        print(f"Pearson correlation: r = {pearson_r:.4f}, p = {pearson_p:.2e}")
        print(f"Spearman correlation: ρ = {spearman_r:.4f}, p = {spearman_p:.2e}")
        print(f"Mean embedding similarity: {np.mean(similarities):.4f}")
        print(f"Mean LLM score: {np.mean(llm_scores):.4f}")

        return {
            'n_pairs': len(similarities),
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'mean_embedding_sim': np.mean(similarities),
            'mean_llm_score': np.mean(llm_scores)
        }
    else:
        print("Not enough pairs for correlation analysis")
        return None


if __name__ == "__main__":
    quick_benchmark()