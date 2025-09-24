#!/usr/bin/env python3
"""Practical search performance analysis with real-world metrics."""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, average_precision_score
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchPerformanceAnalyzer:
    """Analyze real-world search performance of embedding models."""

    def __init__(self):
        self.ground_truth = []
        self.embeddings = {}
        self.results = {}

    def load_data(self):
        """Load ground truth and embeddings."""
        logger.info("Loading ground truth data...")

        # Load ground truth
        with open("data/ground_truth/consolidated/ground_truth_10k.jsonl", 'r') as f:
            for line in f:
                if line.strip():
                    self.ground_truth.append(json.loads(line.strip()))

        logger.info(f"Loaded {len(self.ground_truth)} ground truth pairs")

        # Load three-way fair comparison results to get the exact patents we can work with
        with open("results/three_way_fair_comparison_results.json", 'r') as f:
            fair_results = json.load(f)

        # Get the patents that have all three model embeddings
        fair_patents = set()
        for pair in fair_results.get('valid_pairs', [])[:100]:  # Use first 100 for analysis
            fair_patents.add(pair[0])
            fair_patents.add(pair[1])

        logger.info(f"Working with {len(fair_patents)} patents that have all model embeddings")

        # Load embeddings for these patents
        self.load_embeddings_for_patents(list(fair_patents))

    def load_embeddings_for_patents(self, patent_list):
        """Load embeddings only for the specified patents."""

        # Load OpenAI embeddings
        self.embeddings['openai'] = {}
        with open("results/openai_embeddings_ground_truth.jsonl", 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line.strip())
                patent_id = data['id']
                if patent_id in patent_list:
                    self.embeddings['openai'][patent_id] = np.array(data['embedding'])

        # Load nomic embeddings
        self.embeddings['nomic'] = {}
        nomic_files = [
            "data/embeddings/by_model/nomic-embed-text/production_100k_remaining_nomic-embed-text_20250912_205647_nomic-embed-text_embeddings.jsonl",
            "data/embeddings/by_model/nomic-embed-text/production_10k_all_nomic-embed-text_20250912_150410_nomic-embed-text_embeddings.jsonl",
            "data/embeddings/by_model/nomic-embed-text/diverse_10k_full_nomic-embed-text_20250912_070401_nomic-embed-text_embeddings.jsonl"
        ]

        for file_path in nomic_files:
            try:
                with open(file_path, 'r') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        entry = json.loads(line.strip())
                        patent_id = entry.get('id') or entry.get('patent_id')

                        if patent_id in patent_list:
                            embedding = entry.get('embedding')
                            if not embedding and 'models' in entry:
                                models_data = entry.get('models', {})
                                if 'nomic-embed-text' in models_data:
                                    embeddings_data = models_data['nomic-embed-text'].get('embeddings', {})
                                    if 'original' in embeddings_data:
                                        embedding = embeddings_data['original'].get('embedding')

                            if embedding:
                                self.embeddings['nomic'][patent_id] = np.array(embedding)
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
                continue

        # Load bge-m3 embeddings
        self.embeddings['bge_m3'] = {}
        bge_files = [
            "data/embeddings/by_model/bge-m3/diverse_10k_full_bge-m3_20250912_070401_bge-m3_embeddings.jsonl",
            "data/embeddings/by_model/bge-m3/production_10k_all_bge-m3_20250912_150410_bge-m3_embeddings.jsonl",
            "data/embeddings/by_model/bge-m3/production_100k_top2_bge-m3_20250912_150417_bge-m3_embeddings.jsonl"
        ]

        for file_path in bge_files:
            try:
                with open(file_path, 'r') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        entry = json.loads(line.strip())
                        patent_id = entry.get('id') or entry.get('patent_id')

                        if patent_id in patent_list:
                            embedding = entry.get('embedding')
                            if not embedding and 'models' in entry:
                                models_data = entry.get('models', {})
                                if 'bge-m3' in models_data:
                                    embeddings_data = models_data['bge-m3'].get('embeddings', {})
                                    if 'original' in embeddings_data:
                                        embedding = embeddings_data['original'].get('embedding')

                            if embedding:
                                self.embeddings['bge_m3'][patent_id] = np.array(embedding)
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
                continue

        # Find patents that have all three embeddings
        common_patents = (set(self.embeddings['openai'].keys()) &
                         set(self.embeddings['nomic'].keys()) &
                         set(self.embeddings['bge_m3'].keys()))

        logger.info(f"Found {len(common_patents)} patents with all three embeddings:")
        logger.info(f"  OpenAI: {len(self.embeddings['openai'])}")
        logger.info(f"  nomic: {len(self.embeddings['nomic'])}")
        logger.info(f"  bge-m3: {len(self.embeddings['bge_m3'])}")

        # Keep only common patents
        for model in self.embeddings:
            self.embeddings[model] = {pid: emb for pid, emb in self.embeddings[model].items()
                                     if pid in common_patents}

    def create_mds_visualization(self):
        """Create MDS visualization of patent similarities."""
        logger.info("Creating MDS visualization...")

        # Get patents that have all embeddings
        patent_ids = list(self.embeddings['openai'].keys())
        n_patents = len(patent_ids)

        if n_patents < 10:
            logger.warning(f"Only {n_patents} patents available - skipping MDS")
            return

        logger.info(f"Creating MDS plot with {n_patents} patents")

        # Create LLM similarity matrix for color mapping
        llm_similarities = {}
        for gt_pair in self.ground_truth:
            p1, p2 = gt_pair['patent1_id'], gt_pair['patent2_id']
            if p1 in patent_ids and p2 in patent_ids:
                llm_score = gt_pair['llm_analysis']['similarity_score']
                llm_similarities[(p1, p2)] = llm_score
                llm_similarities[(p2, p1)] = llm_score

        # Create figure with subplots for each model
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        model_names = ['OpenAI text-embedding-3-small', 'nomic-embed-text', 'bge-m3']
        model_keys = ['openai', 'nomic', 'bge_m3']
        colors = ['steelblue', 'darkseagreen', 'coral']

        for idx, (model_key, model_name, color) in enumerate(zip(model_keys, model_names, colors)):
            ax = axes[idx]

            # Get embedding matrix for this model
            embeddings_matrix = np.array([self.embeddings[model_key][pid] for pid in patent_ids])

            # Compute pairwise distances (1 - cosine similarity)
            similarity_matrix = cosine_similarity(embeddings_matrix)
            distance_matrix = 1 - similarity_matrix

            # Apply MDS
            mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
            coords = mds.fit_transform(distance_matrix)

            # Create color map based on average LLM similarity to other patents
            patent_colors = []
            for i, pid in enumerate(patent_ids):
                # Find average LLM similarity for this patent
                similarities = []
                for j, other_pid in enumerate(patent_ids):
                    if i != j:
                        pair_key = (pid, other_pid)
                        if pair_key in llm_similarities:
                            similarities.append(llm_similarities[pair_key])

                avg_sim = np.mean(similarities) if similarities else 0
                patent_colors.append(avg_sim)

            # Create scatter plot
            scatter = ax.scatter(coords[:, 0], coords[:, 1],
                               c=patent_colors, cmap='viridis',
                               s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

            ax.set_title(f'{model_name}\nMDS Projection', fontsize=14, fontweight='bold')
            ax.set_xlabel('MDS Dimension 1')
            ax.set_ylabel('MDS Dimension 2')
            ax.grid(True, alpha=0.3)

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Avg LLM Similarity', rotation=270, labelpad=20)

            # Add some patent labels for context
            for i in range(min(5, len(patent_ids))):  # Label first 5 patents
                ax.annotate(patent_ids[i].replace('patent_', 'P'),
                           (coords[i, 0], coords[i, 1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)

        plt.tight_layout()
        plt.savefig('figures/mds_patent_clusters.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("Created MDS visualization")

    def analyze_search_accuracy(self):
        """Analyze practical search accuracy with precision@k metrics."""
        logger.info("Analyzing search accuracy...")

        patent_ids = list(self.embeddings['openai'].keys())

        if len(patent_ids) < 10:
            logger.warning("Not enough patents for search accuracy analysis")
            return {}

        # Create ground truth relevance matrix
        relevance_matrix = {}
        for gt_pair in self.ground_truth:
            p1, p2 = gt_pair['patent1_id'], gt_pair['patent2_id']
            if p1 in patent_ids and p2 in patent_ids:
                llm_score = gt_pair['llm_analysis']['similarity_score']
                # Consider patents with LLM similarity > 0.3 as "relevant"
                is_relevant = llm_score > 0.3
                relevance_matrix[(p1, p2)] = is_relevant
                relevance_matrix[(p2, p1)] = is_relevant

        results = {}
        k_values = [1, 3, 5, 10]

        for model_key, model_name in [('openai', 'OpenAI'), ('nomic', 'nomic'), ('bge_m3', 'bge-m3')]:
            logger.info(f"Analyzing {model_name}...")

            model_results = {'precision_at_k': {}, 'recall_at_k': {}, 'examples': []}

            # For each patent, find most similar patents and check relevance
            embeddings_matrix = np.array([self.embeddings[model_key][pid] for pid in patent_ids])
            similarity_matrix = cosine_similarity(embeddings_matrix)

            all_precisions_at_k = {k: [] for k in k_values}
            all_recalls_at_k = {k: [] for k in k_values}

            for i, query_patent in enumerate(patent_ids):
                # Get similarities to all other patents
                similarities = similarity_matrix[i]

                # Sort by similarity (excluding self)
                patent_similarities = [(patent_ids[j], similarities[j]) for j in range(len(patent_ids)) if i != j]
                patent_similarities.sort(key=lambda x: x[1], reverse=True)

                # Get ground truth relevant patents for this query
                true_relevant = []
                for j, other_patent in enumerate(patent_ids):
                    if i != j:
                        pair_key = (query_patent, other_patent)
                        if pair_key in relevance_matrix and relevance_matrix[pair_key]:
                            true_relevant.append(other_patent)

                if not true_relevant:  # Skip if no relevant patents
                    continue

                # Calculate precision@k and recall@k
                for k in k_values:
                    if k <= len(patent_similarities):
                        top_k = [p[0] for p in patent_similarities[:k]]
                        relevant_in_top_k = [p for p in top_k if p in true_relevant]

                        precision_at_k = len(relevant_in_top_k) / k
                        recall_at_k = len(relevant_in_top_k) / len(true_relevant) if true_relevant else 0

                        all_precisions_at_k[k].append(precision_at_k)
                        all_recalls_at_k[k].append(recall_at_k)

                # Store example for first few patents
                if len(model_results['examples']) < 3:
                    example = {
                        'query_patent': query_patent,
                        'true_relevant': true_relevant[:3],  # Top 3 for brevity
                        'top_5_results': patent_similarities[:5],
                        'precision_at_5': len([p for p in patent_similarities[:5] if p[0] in true_relevant]) / 5
                    }
                    model_results['examples'].append(example)

            # Calculate average precision@k and recall@k
            for k in k_values:
                if all_precisions_at_k[k]:
                    model_results['precision_at_k'][k] = np.mean(all_precisions_at_k[k])
                    model_results['recall_at_k'][k] = np.mean(all_recalls_at_k[k])
                else:
                    model_results['precision_at_k'][k] = 0
                    model_results['recall_at_k'][k] = 0

            results[model_key] = model_results

        self.results = results
        return results

    def create_search_performance_plots(self):
        """Create practical search performance visualizations."""
        if not self.results:
            logger.warning("No results to plot")
            return

        logger.info("Creating search performance plots...")

        # Create precision@k comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        k_values = [1, 3, 5, 10]
        models = ['openai', 'nomic', 'bge_m3']
        model_names = ['OpenAI text-embedding-3-small', 'nomic-embed-text', 'bge-m3']
        colors = ['steelblue', 'darkseagreen', 'coral']

        # Precision@K plot
        for model, name, color in zip(models, model_names, colors):
            precisions = [self.results[model]['precision_at_k'].get(k, 0) for k in k_values]
            ax1.plot(k_values, precisions, marker='o', linewidth=2.5, markersize=8,
                    label=name, color=color)

        ax1.set_xlabel('K (Number of Retrieved Results)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Precision@K', fontsize=12, fontweight='bold')
        ax1.set_title('Search Precision: How Many Retrieved Results Are Relevant?',
                     fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        # Add annotations for business interpretation
        ax1.text(5, 0.9, 'Higher is better\n(More accurate results)',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

        # Recall@K plot
        for model, name, color in zip(models, model_names, colors):
            recalls = [self.results[model]['recall_at_k'].get(k, 0) for k in k_values]
            ax2.plot(k_values, recalls, marker='s', linewidth=2.5, markersize=8,
                    label=name, color=color)

        ax2.set_xlabel('K (Number of Retrieved Results)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Recall@K', fontsize=12, fontweight='bold')
        ax2.set_title('Search Recall: How Many Relevant Results Are Found?',
                     fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

        # Add annotations
        ax2.text(5, 0.9, 'Higher is better\n(Finds more relevant patents)',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))

        plt.tight_layout()
        plt.savefig('figures/search_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("Created search performance plots")

    def create_business_summary_table(self):
        """Create business-focused summary table."""
        if not self.results:
            return

        logger.info("Creating business summary...")

        # Create summary table
        summary_data = []
        models = ['openai', 'nomic', 'bge_m3']
        model_names = ['OpenAI text-embedding-3-small', 'nomic-embed-text', 'bge-m3']

        for model, name in zip(models, model_names):
            row = {
                'Model': name,
                'Precision@5': f"{self.results[model]['precision_at_k'].get(5, 0):.1%}",
                'Precision@10': f"{self.results[model]['precision_at_k'].get(10, 0):.1%}",
                'Business Impact': ''
            }

            # Add business interpretation
            p5 = self.results[model]['precision_at_k'].get(5, 0)
            if p5 > 0.4:
                row['Business Impact'] = '✅ High accuracy - Good for production'
            elif p5 > 0.2:
                row['Business Impact'] = '⚠️ Moderate accuracy - Needs review'
            else:
                row['Business Impact'] = '❌ Low accuracy - Not recommended'

            summary_data.append(row)

        # Save as JSON for report
        with open('results/business_search_summary.json', 'w') as f:
            json.dump(summary_data, f, indent=2)

        logger.info("Business summary saved to results/business_search_summary.json")

        return summary_data

def main():
    """Run complete practical search analysis."""
    analyzer = SearchPerformanceAnalyzer()

    # Load data
    analyzer.load_data()

    # Create visualizations
    analyzer.create_mds_visualization()

    # Analyze search accuracy
    search_results = analyzer.analyze_search_accuracy()

    if search_results:
        # Create performance plots
        analyzer.create_search_performance_plots()

        # Create business summary
        business_summary = analyzer.create_business_summary_table()

        # Print results
        print("\n" + "="*80)
        print("🔍 PRACTICAL SEARCH PERFORMANCE ANALYSIS")
        print("="*80)

        for model, name in [('openai', 'OpenAI'), ('nomic', 'nomic'), ('bge_m3', 'bge-m3')]:
            if model in search_results:
                results = search_results[model]
                print(f"\n📊 {name}:")
                print(f"  Precision@5:  {results['precision_at_k'].get(5, 0):.1%}")
                print(f"  Precision@10: {results['precision_at_k'].get(10, 0):.1%}")
                print(f"  Recall@5:     {results['recall_at_k'].get(5, 0):.1%}")
                print(f"  Recall@10:    {results['recall_at_k'].get(10, 0):.1%}")

        print(f"\n💼 Business Summary:")
        if business_summary:
            for row in business_summary:
                print(f"  {row['Model']}: {row['Precision@5']} accuracy → {row['Business Impact']}")

    logger.info("Analysis complete! Check figures/ directory for visualizations.")

if __name__ == "__main__":
    main()