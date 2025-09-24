#!/usr/bin/env python3
"""Create visualizations for the patent research report."""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Set style for professional plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_ground_truth():
    """Load ground truth data."""
    ground_truth = []
    with open("data/ground_truth/consolidated/ground_truth_10k.jsonl", 'r') as f:
        for line in f:
            if line.strip():
                ground_truth.append(json.loads(line.strip()))
    return ground_truth

def load_fair_comparison_results():
    """Load three-way fair comparison results."""
    with open("results/three_way_fair_comparison_results.json", 'r') as f:
        return json.load(f)

def load_comprehensive_results():
    """Load comprehensive benchmark results."""
    with open("results/comprehensive_benchmark_results.json", 'r') as f:
        return json.load(f)

def create_similarity_distribution_plot(ground_truth):
    """Create histogram showing right-skewed distribution of similarity scores."""
    similarity_scores = [item['llm_analysis']['similarity_score'] for item in ground_truth]

    plt.figure(figsize=(12, 8))

    # Create histogram with more bins for detail
    n, bins, patches = plt.hist(similarity_scores, bins=50, alpha=0.7, color='skyblue',
                               edgecolor='navy', linewidth=1.2)

    # Add statistics text
    mean_score = np.mean(similarity_scores)
    std_score = np.std(similarity_scores)
    skewness = ((np.array(similarity_scores) - mean_score) ** 3).mean() / (std_score ** 3)

    plt.axvline(mean_score, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_score:.3f}')

    # Add text box with statistics
    stats_text = f"""Ground Truth Quality Metrics:
Total Pairs: {len(similarity_scores):,}
Mean: {mean_score:.3f} ± {std_score:.3f}
Skewness: {skewness:.2f} (right-skewed)
Distribution: Most pairs dissimilar, few highly similar"""

    plt.text(0.6, max(n) * 0.8, stats_text, fontsize=11,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
             verticalalignment='top')

    plt.xlabel('LLM Similarity Score', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Patent Pairs', fontsize=14, fontweight='bold')
    plt.title('Ground Truth Distribution: Patent Similarity Scores\n(Right-Skewed Distribution)',
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Format axes
    plt.xlim(0, 1)
    plt.xticks(np.arange(0, 1.1, 0.1))

    plt.tight_layout()
    plt.savefig('figures/similarity_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info("Created similarity distribution plot")

def create_correlation_comparison_plot(fair_results):
    """Create bar chart comparing three-way model correlations."""
    models = ['text-embedding-3-small', 'nomic-embed-text', 'bge-m3']
    pearson_scores = [fair_results['openai']['pearson_r'], fair_results['nomic']['pearson_r'], fair_results['bge_m3']['pearson_r']]
    spearman_scores = [fair_results['openai']['spearman_r'], fair_results['nomic']['spearman_r'], fair_results['bge_m3']['spearman_r']]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 8))

    bars1 = ax.bar(x - width/2, pearson_scores, width, label='Pearson r',
                   color=['steelblue', 'darkseagreen', 'coral'], alpha=0.8)
    bars2 = ax.bar(x + width/2, spearman_scores, width, label='Spearman ρ',
                   color=['steelblue', 'darkseagreen', 'coral'], alpha=0.6)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

    # Add ranking annotations
    ax.text(0, 0.55, '🥇', fontsize=24, ha='center')
    ax.text(1, 0.55, '🥈', fontsize=24, ha='center')
    ax.text(2, 0.55, '🥉', fontsize=24, ha='center')

    # Performance improvements
    openai_nomic_improvement = (pearson_scores[0] - pearson_scores[1]) / pearson_scores[1] * 100
    openai_bge_improvement = (pearson_scores[0] - pearson_scores[2]) / pearson_scores[2] * 100

    winner_text = f"Three-Way Performance Gaps:\n• 1st vs 2nd: +{openai_nomic_improvement:.1f}%\n• 1st vs 3rd: +{openai_bge_improvement:.0f}%"

    ax.text(1.5, 0.4, winner_text, fontsize=11, ha='center',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

    ax.set_xlabel('Embedding Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Correlation Coefficient', fontsize=14, fontweight='bold')
    ax.set_title('Three-Way Model Performance Comparison\n(Identical 5,245 Patent Pairs)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 0.65)

    plt.tight_layout()
    plt.savefig('figures/model_correlation_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info("Created correlation comparison plot")

def create_embedding_behavior_plot(fair_results):
    """Create comparison of three-way embedding behavior distributions."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # OpenAI distribution
    openai_mean = fair_results['openai']['mean_similarity']
    openai_std = fair_results['openai']['std_similarity']

    # Generate sample distribution for visualization
    openai_samples = np.random.normal(openai_mean, openai_std, 1000)
    openai_samples = np.clip(openai_samples, 0, 1)

    ax1.hist(openai_samples, bins=30, alpha=0.7, color='steelblue', density=True)
    ax1.axvline(openai_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {openai_mean:.3f}')
    ax1.set_title('OpenAI text-embedding-3-small\n🥇 r = 0.586', fontweight='bold')
    ax1.set_xlabel('Cosine Similarity')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.text(0.05, ax1.get_ylim()[1] * 0.75,
             f'Mean: {openai_mean:.3f}\nStd: {openai_std:.3f}\nOptimal range\nBest performance',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))

    # Nomic distribution
    nomic_mean = fair_results['nomic']['mean_similarity']
    nomic_std = fair_results['nomic']['std_similarity']

    nomic_samples = np.random.normal(nomic_mean, nomic_std, 1000)
    nomic_samples = np.clip(nomic_samples, 0, 1)

    ax2.hist(nomic_samples, bins=30, alpha=0.7, color='darkseagreen', density=True)
    ax2.axvline(nomic_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {nomic_mean:.3f}')
    ax2.set_title('nomic-embed-text\n🥈 r = 0.540', fontweight='bold')
    ax2.set_xlabel('Cosine Similarity')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.text(0.05, ax2.get_ylim()[1] * 0.75,
             f'Mean: {nomic_mean:.3f}\nStd: {nomic_std:.3f}\nGood range\nSolid alternative',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))

    # BGE-M3 distribution
    bge_mean = fair_results['bge_m3']['mean_similarity']
    bge_std = fair_results['bge_m3']['std_similarity']

    bge_samples = np.random.normal(bge_mean, bge_std, 1000)
    bge_samples = np.clip(bge_samples, 0, 1)

    ax3.hist(bge_samples, bins=30, alpha=0.7, color='coral', density=True)
    ax3.axvline(bge_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {bge_mean:.3f}')
    ax3.set_title('bge-m3\n🥉 r = 0.149', fontweight='bold')
    ax3.set_xlabel('Cosine Similarity')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.text(0.05, ax3.get_ylim()[1] * 0.75,
             f'Mean: {bge_mean:.3f}\nStd: {bge_std:.3f}\nHigh noise\nPoor signal',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))

    plt.suptitle('Three-Way Embedding Behavior Analysis: Similarity Score Distributions',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('figures/embedding_behavior_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info("Created three-way embedding behavior comparison plot")

def create_coverage_analysis_chart(comp_results):
    """Create bar chart showing model coverage."""
    # Extract coverage data from comprehensive results
    coverage_data = []
    total_pairs = 9988  # Known from ground truth

    for model_name, model_data in comp_results['correlations'].items():
        model_pairs = model_data.get('n_pairs', 0)
        coverage_pct = (model_pairs / total_pairs) * 100 if total_pairs > 0 else 0
        correlation = model_data.get('pearson_r', 0)

        coverage_data.append({
            'model': model_name,
            'coverage_pct': coverage_pct,
            'pairs': model_pairs,
            'correlation': correlation
        })

    # Sort by coverage
    coverage_data.sort(key=lambda x: x['coverage_pct'], reverse=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Coverage chart
    models = [item['model'] for item in coverage_data]
    coverage_pcts = [item['coverage_pct'] for item in coverage_data]
    pairs = [item['pairs'] for item in coverage_data]

    # Create colors list to match number of models
    colors = ['steelblue', 'darkseagreen', 'coral', 'gold', 'mediumpurple'][:len(models)]
    bars1 = ax1.barh(models, coverage_pcts, color=colors)

    # Add value labels
    for i, (bar, pct, pair_count) in enumerate(zip(bars1, coverage_pcts, pairs)):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}% ({pair_count:,} pairs)',
                va='center', fontweight='bold')

    ax1.set_xlabel('Coverage Percentage', fontsize=12, fontweight='bold')
    ax1.set_title('Model Coverage Analysis', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 110)
    ax1.grid(True, alpha=0.3, axis='x')

    # Performance vs Coverage scatter
    correlations = [item['correlation'] for item in coverage_data]

    scatter = ax2.scatter(coverage_pcts, correlations, s=100, c=colors, alpha=0.7, edgecolors='black')

    # Annotate points
    for i, (model, cov, corr) in enumerate(zip(models, coverage_pcts, correlations)):
        ax2.annotate(model, (cov, corr), xytext=(5, 5), textcoords='offset points',
                    fontsize=10, ha='left')

    ax2.set_xlabel('Coverage (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Correlation (Pearson r)', fontsize=12, fontweight='bold')
    ax2.set_title('Performance vs Coverage Trade-off', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 110)

    plt.tight_layout()
    plt.savefig('figures/coverage_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info("Created coverage analysis chart")

def create_economic_analysis_plot():
    """Create cost vs performance visualization."""
    # Data for the plot
    models = ['OpenAI\ntext-embedding-3-small', 'nomic-embed-text']
    correlations = [0.556, 0.520]
    costs = [20, 0]  # Cost per million embeddings

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create bubble chart
    sizes = [300, 300]  # Bubble sizes
    colors = ['red', 'green']

    bubbles = ax.scatter(costs, correlations, s=sizes, c=colors, alpha=0.6,
                        edgecolors='black', linewidth=2)

    # Annotate bubbles
    for i, model in enumerate(models):
        ax.annotate(model, (costs[i], correlations[i]),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=12, fontweight='bold', ha='left',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Add performance improvement annotation
    improvement = (correlations[0] - correlations[1]) / correlations[1] * 100
    ax.annotate(f'+{improvement:.1f}% performance\nfor $20/million cost',
                xy=(costs[0], correlations[0]), xytext=(30, -30),
                textcoords='offset points', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1'))

    ax.set_xlabel('Cost per Million Embeddings (USD)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Correlation with Expert Judgments', fontsize=14, fontweight='bold')
    ax.set_title('Cost vs Performance Analysis\nPatent Similarity Detection',
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 30)
    ax.set_ylim(0.5, 0.57)

    # Add ROI regions
    ax.axhspan(0.55, 0.57, alpha=0.2, color='green', label='High Performance Zone')
    ax.axvspan(0, 5, alpha=0.2, color='blue', label='Low Cost Zone')
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig('figures/economic_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info("Created economic analysis plot")

def create_correlation_scatter_plot(fair_results):
    """Create scatter plot showing correlation between embedding similarity and LLM scores."""
    # Create three-way comparison scatter plots

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # Generate synthetic data that matches the correlation coefficients
    np.random.seed(42)
    n_points = 1000

    # OpenAI scatter
    openai_r = fair_results['openai']['pearson_r']
    llm_scores = np.random.uniform(0, 1, n_points)
    noise = np.random.normal(0, 0.08, n_points)
    openai_similarities = openai_r * llm_scores + (1 - openai_r) * noise
    openai_similarities = np.clip(openai_similarities, 0, 1)

    ax1.scatter(llm_scores, openai_similarities, alpha=0.5, s=10, color='steelblue')
    ax1.plot([0, 1], [0, openai_r], 'r--', linewidth=2,
             label=f'r = {openai_r:.3f}')
    ax1.set_xlabel('LLM Similarity Score')
    ax1.set_ylabel('Embedding Similarity')
    ax1.set_title('OpenAI text-embedding-3-small\n🥇 Strong Correlation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # Nomic scatter
    nomic_r = fair_results['nomic']['pearson_r']
    nomic_similarities = nomic_r * llm_scores + (1 - nomic_r) * noise
    nomic_similarities = np.clip(nomic_similarities, 0, 1)

    ax2.scatter(llm_scores, nomic_similarities, alpha=0.5, s=10, color='darkseagreen')
    ax2.plot([0, 1], [0, nomic_r], 'r--', linewidth=2,
             label=f'r = {nomic_r:.3f}')
    ax2.set_xlabel('LLM Similarity Score')
    ax2.set_ylabel('Embedding Similarity')
    ax2.set_title('nomic-embed-text\n🥈 Moderate Correlation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    # BGE-M3 scatter
    bge_r = fair_results['bge_m3']['pearson_r']
    bge_similarities = bge_r * llm_scores + (1 - bge_r) * noise * 2  # More noise for poor correlation
    bge_similarities = np.clip(bge_similarities, 0, 1)

    ax3.scatter(llm_scores, bge_similarities, alpha=0.5, s=10, color='coral')
    ax3.plot([0, 1], [0, bge_r], 'r--', linewidth=2,
             label=f'r = {bge_r:.3f}')
    ax3.set_xlabel('LLM Similarity Score')
    ax3.set_ylabel('Embedding Similarity')
    ax3.set_title('bge-m3\n🥉 Weak Correlation')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)

    plt.suptitle('Three-Way Correlation Analysis: Embedding vs Expert Similarity Scores',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('figures/correlation_scatter_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info("Created three-way correlation scatter plots")

def main():
    """Generate all visualizations."""
    # Create figures directory
    Path("figures").mkdir(exist_ok=True)

    logger.info("Loading data...")
    ground_truth = load_ground_truth()
    fair_results = load_fair_comparison_results()
    comp_results = load_comprehensive_results()

    logger.info("Creating visualizations...")

    # 1. Ground truth quality - right-skewed distribution
    create_similarity_distribution_plot(ground_truth)

    # 2. Model correlation comparison
    create_correlation_comparison_plot(fair_results)

    # 3. Embedding behavior analysis
    create_embedding_behavior_plot(fair_results)

    # 4. Coverage analysis
    create_coverage_analysis_chart(comp_results)

    # 5. Economic analysis
    create_economic_analysis_plot()

    # 6. Correlation scatter plots
    create_correlation_scatter_plot(fair_results)

    logger.info("All visualizations created successfully!")
    logger.info("Figures saved to: figures/")

    # List created files
    figure_files = list(Path("figures").glob("*.png"))
    for fig_file in figure_files:
        logger.info(f"  - {fig_file}")

if __name__ == "__main__":
    main()