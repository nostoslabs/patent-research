#!/usr/bin/env python3
"""Realistic search performance analysis using the full fair comparison dataset."""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS, TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealisticSearchAnalyzer:
    """Analyze realistic search performance with business-focused metrics."""

    def __init__(self):
        self.ground_truth_pairs = []
        self.embeddings = {}
        self.patent_similarities = {}

    def load_fair_comparison_data(self):
        """Load the fair comparison results and underlying data."""
        logger.info("Loading fair comparison data...")

        # Load three-way fair comparison results
        with open("results/three_way_fair_comparison_results.json", 'r') as f:
            fair_results = json.load(f)

        # Load ground truth for the LLM similarities
        with open("data/ground_truth/consolidated/ground_truth_10k.jsonl", 'r') as f:
            all_ground_truth = []
            for line in f:
                if line.strip():
                    all_ground_truth.append(json.loads(line.strip()))

        # Get the pairs that were used in the fair comparison
        fair_patent_ids = set()
        for pair in fair_results.get('valid_pairs', []):
            fair_patent_ids.add(pair[0])
            fair_patent_ids.add(pair[1])

        # Filter ground truth to only include pairs from fair comparison
        self.ground_truth_pairs = []
        for gt in all_ground_truth:
            if (gt['patent1_id'] in fair_patent_ids and
                gt['patent2_id'] in fair_patent_ids):
                self.ground_truth_pairs.append(gt)

        logger.info(f"Loaded {len(self.ground_truth_pairs)} ground truth pairs from fair comparison")

        # Load embeddings (use first 200 patents to make analysis manageable)
        sample_patents = list(fair_patent_ids)[:200]
        self.load_embeddings_for_patents(sample_patents)

    def load_embeddings_for_patents(self, patent_list):
        """Load embeddings for specific patents."""
        self.embeddings = {'openai': {}, 'nomic': {}}

        # Load OpenAI embeddings
        with open("results/openai_embeddings_ground_truth.jsonl", 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line.strip())
                patent_id = data['id']
                if patent_id in patent_list:
                    self.embeddings['openai'][patent_id] = np.array(data['embedding'])

        # Load nomic embeddings
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
                                break  # Found embedding for this patent
            except Exception as e:
                continue

        # Keep only patents that have both embeddings
        common_patents = set(self.embeddings['openai'].keys()) & set(self.embeddings['nomic'].keys())
        for model in self.embeddings:
            self.embeddings[model] = {pid: emb for pid, emb in self.embeddings[model].items()
                                     if pid in common_patents}

        logger.info(f"Loaded embeddings for {len(common_patents)} patents")

    def create_practical_search_scenarios(self):
        """Create realistic search scenarios based on actual patent similarity data."""
        logger.info("Creating practical search scenarios...")

        patent_ids = list(self.embeddings['openai'].keys())
        if len(patent_ids) < 20:
            logger.warning("Not enough patents for realistic scenarios")
            return {}

        # Build LLM similarity lookup
        llm_similarities = {}
        for gt in self.ground_truth_pairs:
            p1, p2 = gt['patent1_id'], gt['patent2_id']
            if p1 in patent_ids and p2 in patent_ids:
                llm_score = gt['llm_analysis']['similarity_score']
                llm_similarities[(p1, p2)] = llm_score
                llm_similarities[(p2, p1)] = llm_score

        # Define relevance levels for business interpretation
        relevance_thresholds = {
            'highly_relevant': 0.4,   # Patents that solve similar problems
            'somewhat_relevant': 0.2,  # Patents with some technical overlap
            'marginally_relevant': 0.1 # Patents with minimal connection
        }

        results = {}

        for model_name, model_key in [('OpenAI text-embedding-3-small', 'openai'),
                                     ('nomic-embed-text', 'nomic')]:
            logger.info(f"Analyzing {model_name}...")

            embeddings_matrix = np.array([self.embeddings[model_key][pid] for pid in patent_ids])
            similarity_matrix = cosine_similarity(embeddings_matrix)

            model_results = {
                'precision_at_k': {},
                'recall_at_k': {},
                'relevance_distribution': {},
                'search_examples': []
            }

            k_values = [1, 3, 5, 10, 20]

            # For each relevance threshold
            for relevance_name, threshold in relevance_thresholds.items():
                precision_scores = {k: [] for k in k_values}
                recall_scores = {k: [] for k in k_values}

                search_count = 0
                for i, query_patent in enumerate(patent_ids):
                    if search_count >= 50:  # Limit for performance
                        break

                    # Find truly relevant patents (above threshold)
                    truly_relevant = []
                    for j, candidate_patent in enumerate(patent_ids):
                        if i != j:
                            pair_key = (query_patent, candidate_patent)
                            if pair_key in llm_similarities:
                                if llm_similarities[pair_key] >= threshold:
                                    truly_relevant.append(candidate_patent)

                    if len(truly_relevant) == 0:  # Skip if no relevant patents
                        continue

                    # Get embedding-based rankings
                    query_similarities = similarity_matrix[i]
                    ranked_patents = [(patent_ids[j], query_similarities[j])
                                    for j in range(len(patent_ids)) if i != j]
                    ranked_patents.sort(key=lambda x: x[1], reverse=True)

                    # Calculate precision@k and recall@k
                    for k in k_values:
                        if k <= len(ranked_patents):
                            top_k = [p[0] for p in ranked_patents[:k]]
                            relevant_in_top_k = [p for p in top_k if p in truly_relevant]

                            precision = len(relevant_in_top_k) / k
                            recall = len(relevant_in_top_k) / len(truly_relevant)

                            precision_scores[k].append(precision)
                            recall_scores[k].append(recall)

                    # Store example for the first few searches
                    if len(model_results['search_examples']) < 3 and search_count < 3:
                        example = {
                            'query_patent': query_patent,
                            'relevance_level': relevance_name,
                            'num_truly_relevant': len(truly_relevant),
                            'top_5_embedding_results': ranked_patents[:5],
                            'relevant_in_top_5': len([p for p in ranked_patents[:5] if p[0] in truly_relevant])
                        }
                        model_results['search_examples'].append(example)

                    search_count += 1

                # Calculate averages
                model_results['precision_at_k'][relevance_name] = {
                    k: np.mean(scores) if scores else 0
                    for k, scores in precision_scores.items()
                }
                model_results['recall_at_k'][relevance_name] = {
                    k: np.mean(scores) if scores else 0
                    for k, scores in recall_scores.items()
                }

            results[model_key] = model_results

        return results

    def create_business_focused_plots(self, results):
        """Create plots that managers can easily understand."""
        logger.info("Creating business-focused visualizations...")

        # 1. Precision@5 comparison for different relevance levels
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        relevance_levels = ['highly_relevant', 'somewhat_relevant', 'marginally_relevant']
        relevance_labels = ['Highly Relevant\n(Similar Solutions)',
                          'Somewhat Relevant\n(Technical Overlap)',
                          'Marginally Relevant\n(Weak Connection)']

        x = np.arange(len(relevance_levels))
        width = 0.35

        openai_p5 = [results['openai']['precision_at_k'][level].get(5, 0) for level in relevance_levels]
        nomic_p5 = [results['nomic']['precision_at_k'][level].get(5, 0) for level in relevance_levels]

        bars1 = ax1.bar(x - width/2, openai_p5, width, label='OpenAI text-embedding-3-small',
                       color='steelblue', alpha=0.8)
        bars2 = ax1.bar(x + width/2, nomic_p5, width, label='nomic-embed-text',
                       color='darkseagreen', alpha=0.8)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.1%}', ha='center', va='bottom', fontweight='bold')

        ax1.set_xlabel('Patent Relevance Level', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Precision@5 (% Accurate Results)', fontsize=12, fontweight='bold')
        ax1.set_title('Search Accuracy: Out of 5 Results, How Many Are Actually Relevant?',
                     fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(relevance_labels)
        ax1.legend()
        ax1.set_ylim(0, max(max(openai_p5), max(nomic_p5)) * 1.2)
        ax1.grid(True, alpha=0.3, axis='y')

        # Add business interpretation box
        ax1.text(0.5, max(max(openai_p5), max(nomic_p5)) * 0.9,
                'Higher bars = More accurate search results\nBetter for finding relevant patents quickly',
                ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

        # 2. Precision@K curve for highly relevant patents
        k_values = [1, 3, 5, 10, 20]
        openai_precisions = [results['openai']['precision_at_k']['highly_relevant'].get(k, 0) for k in k_values]
        nomic_precisions = [results['nomic']['precision_at_k']['highly_relevant'].get(k, 0) for k in k_values]

        ax2.plot(k_values, openai_precisions, 'o-', linewidth=3, markersize=8,
                label='OpenAI text-embedding-3-small', color='steelblue')
        ax2.plot(k_values, nomic_precisions, 's-', linewidth=3, markersize=8,
                label='nomic-embed-text', color='darkseagreen')

        ax2.set_xlabel('Number of Search Results Examined (K)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Precision@K for Highly Relevant Patents', fontsize=12, fontweight='bold')
        ax2.set_title('Precision vs Search Depth\n(Finding Patents with Similar Solutions)',
                     fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, max(max(openai_precisions), max(nomic_precisions)) * 1.1)

        # Add interpretation
        max_precision = max(max(openai_precisions), max(nomic_precisions))
        ax2.text(10, max_precision * 0.8,
                'Higher line = Better model\nSteeper drop = Need fewer results',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))

        plt.tight_layout()
        plt.savefig('figures/realistic_search_performance.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("Created business-focused search performance plots")

    def create_search_examples_visualization(self, results):
        """Create visualization showing actual search examples."""
        logger.info("Creating search examples visualization...")

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        for idx, (model_key, model_name) in enumerate([('openai', 'OpenAI text-embedding-3-small'),
                                                      ('nomic', 'nomic-embed-text')]):
            ax = axes[idx]

            # Get search examples
            examples = results[model_key]['search_examples'][:2]  # First 2 examples

            if not examples:
                ax.text(0.5, 0.5, 'No examples available', ha='center', va='center')
                continue

            # Create a simple bar chart showing search results accuracy
            example_data = []
            for i, example in enumerate(examples):
                query = example['query_patent'].replace('patent_', 'P')
                accuracy = example['relevant_in_top_5'] / 5 * 100
                example_data.append((f"Query {i+1}\n({query})", accuracy))

            queries, accuracies = zip(*example_data)
            bars = ax.bar(queries, accuracies, color='steelblue' if idx == 0 else 'darkseagreen', alpha=0.7)

            # Add value labels
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{acc:.0f}%', ha='center', va='bottom', fontweight='bold')

            ax.set_ylabel('Accuracy (% Relevant in Top 5)', fontsize=12, fontweight='bold')
            ax.set_title(f'{model_name}: Real Search Examples', fontsize=14, fontweight='bold')
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3, axis='y')

            # Add interpretation text
            avg_accuracy = np.mean(accuracies)
            interpretation = "Good accuracy" if avg_accuracy > 40 else "Moderate accuracy" if avg_accuracy > 20 else "Low accuracy"
            ax.text(0.5, 80, f'Average: {avg_accuracy:.0f}% - {interpretation}',
                   ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

        plt.tight_layout()
        plt.savefig('figures/search_examples.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("Created search examples visualization")

    def generate_manager_summary(self, results):
        """Generate executive summary for managers."""
        logger.info("Generating manager summary...")

        summary = {
            "executive_summary": {},
            "practical_impact": {},
            "recommendations": {}
        }

        # Calculate key metrics
        for model_key, model_name in [('openai', 'OpenAI'), ('nomic', 'nomic-embed-text')]:
            if model_key in results:
                # Get precision for highly relevant patents
                p5_high = results[model_key]['precision_at_k']['highly_relevant'].get(5, 0)
                p10_high = results[model_key]['precision_at_k']['highly_relevant'].get(10, 0)

                summary["executive_summary"][model_name] = {
                    "precision_at_5": f"{p5_high:.1%}",
                    "precision_at_10": f"{p10_high:.1%}",
                    "business_interpretation": self.interpret_precision(p5_high)
                }

                # Real-world impact
                patents_per_day = 100  # Assume analyst searches 100 patents per day
                accurate_results_per_day = patents_per_day * p5_high * 5  # 5 results per search

                summary["practical_impact"][model_name] = {
                    "daily_searches": patents_per_day,
                    "accurate_results_per_day": f"{accurate_results_per_day:.0f}",
                    "time_savings": self.calculate_time_savings(p5_high)
                }

        # Generate recommendations
        openai_p5 = results.get('openai', {}).get('precision_at_k', {}).get('highly_relevant', {}).get(5, 0)
        nomic_p5 = results.get('nomic', {}).get('precision_at_k', {}).get('highly_relevant', {}).get(5, 0)

        if openai_p5 > nomic_p5 * 1.1:  # At least 10% better
            recommendation = f"Use OpenAI text-embedding-3-small for production. {((openai_p5 - nomic_p5) / nomic_p5 * 100):.0f}% more accurate than open-source alternative."
        elif nomic_p5 > openai_p5 * 1.1:
            recommendation = f"Use nomic-embed-text. {((nomic_p5 - openai_p5) / openai_p5 * 100):.0f}% more accurate and cost-free."
        else:
            recommendation = "Both models show similar performance. Choose based on cost considerations."

        summary["recommendations"]["primary"] = recommendation

        # Save summary
        with open('results/manager_search_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        return summary

    def interpret_precision(self, precision):
        """Interpret precision score for business context."""
        if precision >= 0.5:
            return "Excellent - Most search results are relevant"
        elif precision >= 0.3:
            return "Good - Majority of results are useful"
        elif precision >= 0.15:
            return "Moderate - Some manual filtering needed"
        else:
            return "Poor - Requires significant manual review"

    def calculate_time_savings(self, precision):
        """Calculate time savings from better search accuracy."""
        baseline_precision = 0.1  # Assume 10% baseline without embeddings
        if precision <= baseline_precision:
            return "No time savings"

        improvement = (precision - baseline_precision) / baseline_precision
        time_saved_percent = min(improvement * 20, 80)  # Cap at 80% time savings
        return f"{time_saved_percent:.0f}% time savings in patent review"

def main():
    """Run realistic search analysis."""
    analyzer = RealisticSearchAnalyzer()

    # Load data
    analyzer.load_fair_comparison_data()

    # Create search scenarios
    results = analyzer.create_practical_search_scenarios()

    if results and any(results.values()):
        # Create visualizations
        analyzer.create_business_focused_plots(results)
        analyzer.create_search_examples_visualization(results)

        # Generate summary
        summary = analyzer.generate_manager_summary(results)

        # Print executive summary
        print("\n" + "="*80)
        print("🎯 EXECUTIVE SEARCH PERFORMANCE SUMMARY")
        print("="*80)

        for model, data in summary["executive_summary"].items():
            print(f"\n📊 {model}:")
            print(f"  ✓ Precision@5:  {data['precision_at_5']}")
            print(f"  ✓ Precision@10: {data['precision_at_10']}")
            print(f"  📈 Impact: {data['business_interpretation']}")

        print(f"\n💼 Practical Impact (per day):")
        for model, data in summary["practical_impact"].items():
            print(f"  {model}: {data['accurate_results_per_day']} accurate results → {data['time_savings']}")

        print(f"\n🎯 Recommendation:")
        print(f"  {summary['recommendations']['primary']}")

    else:
        logger.warning("No results generated - insufficient data for analysis")

if __name__ == "__main__":
    main()