#!/usr/bin/env python3
"""Scientific analysis of patent similarity evaluation results."""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_ground_truth_data(file_path: str) -> pd.DataFrame:
    """Load and parse ground truth evaluation data."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            record = json.loads(line.strip())
            if record.get('success', False):
                data.append({
                    'patent1_id': record['patent1_id'],
                    'patent2_id': record['patent2_id'],
                    'embedding_similarity': record['embedding_similarity'],
                    'similarity_category': record['similarity_category'],
                    'classification1': record['classification1'],
                    'classification2': record['classification2'],
                    'abstract_length1': record['abstract_length1'],
                    'abstract_length2': record['abstract_length2'],
                    'llm_similarity': record['llm_analysis']['similarity_score'],
                    'technical_field_match': record['llm_analysis']['technical_field_match'],
                    'problem_similarity': record['llm_analysis']['problem_similarity'],
                    'solution_similarity': record['llm_analysis']['solution_similarity'],
                    'llm_confidence': record['llm_analysis']['confidence']
                })
    
    return pd.DataFrame(data)


def analyze_embedding_llm_correlation(df: pd.DataFrame) -> dict:
    """Analyze correlation between embedding similarity and LLM judgments."""
    
    # Calculate correlations
    pearson_r, pearson_p = pearsonr(df['embedding_similarity'], df['llm_similarity'])
    spearman_r, spearman_p = spearmanr(df['embedding_similarity'], df['llm_similarity'])
    
    # Calculate error metrics
    mae = mean_absolute_error(df['embedding_similarity'], df['llm_similarity'])
    mse = mean_squared_error(df['embedding_similarity'], df['llm_similarity'])
    rmse = np.sqrt(mse)
    
    return {
        'pearson_correlation': pearson_r,
        'pearson_p_value': pearson_p,
        'spearman_correlation': spearman_r,
        'spearman_p_value': spearman_p,
        'mean_absolute_error': mae,
        'root_mean_square_error': rmse,
        'sample_size': len(df)
    }


def create_visualizations(df: pd.DataFrame, output_dir: str = "."):
    """Create scientific visualizations of the analysis results."""
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Scatter plot: Embedding vs LLM similarity
    axes[0, 0].scatter(df['embedding_similarity'], df['llm_similarity'], alpha=0.6, s=50)
    axes[0, 0].plot([0, 1], [0, 1], 'r--', alpha=0.8, label='Perfect Agreement')
    axes[0, 0].set_xlabel('Embedding Similarity (Cosine)')
    axes[0, 0].set_ylabel('LLM Similarity Score')
    axes[0, 0].set_title('Embedding vs LLM Similarity Correlation')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Distribution of LLM similarity scores
    axes[0, 1].hist(df['llm_similarity'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 1].axvline(df['llm_similarity'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {df["llm_similarity"].mean():.3f}')
    axes[0, 1].set_xlabel('LLM Similarity Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of LLM Similarity Scores')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Distribution of embedding similarity scores
    axes[1, 0].hist(df['embedding_similarity'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1, 0].axvline(df['embedding_similarity'].mean(), color='red', linestyle='--',
                      label=f'Mean: {df["embedding_similarity"].mean():.3f}')
    axes[1, 0].set_xlabel('Embedding Similarity (Cosine)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Embedding Similarity Scores')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Similarity by classification match
    df['same_classification'] = (df['classification1'] == df['classification2'])
    same_class = df[df['same_classification']]
    diff_class = df[~df['same_classification']]
    
    axes[1, 1].boxplot([same_class['llm_similarity'], diff_class['llm_similarity']], 
                       labels=['Same Classification', 'Different Classification'])
    axes[1, 1].set_ylabel('LLM Similarity Score')
    axes[1, 1].set_title('LLM Similarity by Classification Match')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/patent_similarity_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create correlation heatmap
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    correlation_data = df[['embedding_similarity', 'llm_similarity', 'technical_field_match', 
                          'problem_similarity', 'solution_similarity']].corr()
    
    sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.3f', ax=ax)
    ax.set_title('Correlation Matrix: Embedding vs LLM Similarity Dimensions')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_statistical_summary(df: pd.DataFrame) -> dict:
    """Generate comprehensive statistical summary."""
    
    correlation_stats = analyze_embedding_llm_correlation(df)
    
    # Descriptive statistics
    embedding_stats = {
        'mean': df['embedding_similarity'].mean(),
        'std': df['embedding_similarity'].std(),
        'min': df['embedding_similarity'].min(),
        'max': df['embedding_similarity'].max(),
        'median': df['embedding_similarity'].median()
    }
    
    llm_stats = {
        'mean': df['llm_similarity'].mean(),
        'std': df['llm_similarity'].std(),
        'min': df['llm_similarity'].min(),
        'max': df['llm_similarity'].max(),
        'median': df['llm_similarity'].median()
    }
    
    # Classification analysis
    same_class_count = (df['classification1'] == df['classification2']).sum()
    same_class_llm_mean = df[df['classification1'] == df['classification2']]['llm_similarity'].mean()
    diff_class_llm_mean = df[df['classification1'] != df['classification2']]['llm_similarity'].mean()
    
    return {
        'correlation_analysis': correlation_stats,
        'embedding_similarity_stats': embedding_stats,
        'llm_similarity_stats': llm_stats,
        'classification_analysis': {
            'same_classification_pairs': same_class_count,
            'different_classification_pairs': len(df) - same_class_count,
            'same_class_avg_llm_similarity': same_class_llm_mean,
            'diff_class_avg_llm_similarity': diff_class_llm_mean
        },
        'sample_composition': {
            'total_pairs': len(df),
            'high_embedding_similarity_pairs': (df['embedding_similarity'] > 0.7).sum(),
            'medium_embedding_similarity_pairs': ((df['embedding_similarity'] >= 0.4) & (df['embedding_similarity'] <= 0.7)).sum(),
            'low_embedding_similarity_pairs': (df['embedding_similarity'] < 0.4).sum()
        }
    }


def main():
    """Run complete scientific analysis."""
    
    # Load data
    logger.info("Loading ground truth data...")
    df = load_ground_truth_data('patent_ground_truth_100.jsonl')
    logger.info(f"Loaded {len(df)} patent pairs")
    
    # Generate statistical analysis
    logger.info("Generating statistical analysis...")
    stats = generate_statistical_summary(df)
    
    # Create visualizations
    logger.info("Creating visualizations...")
    create_visualizations(df)
    
    # Save detailed analysis
    with open('patent_similarity_statistical_analysis.json', 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        def clean_dict(d):
            if isinstance(d, dict):
                return {k: clean_dict(v) for k, v in d.items()}
            else:
                return convert_numpy(d)
        
        clean_stats = clean_dict(stats)
        json.dump(clean_stats, f, indent=2)
    
    logger.info("Analysis complete. Results saved to:")
    logger.info("- patent_similarity_analysis.png")
    logger.info("- correlation_heatmap.png") 
    logger.info("- patent_similarity_statistical_analysis.json")
    
    # Print key findings
    print("\n🔬 KEY STATISTICAL FINDINGS:")
    print(f"📊 Sample Size: {stats['correlation_analysis']['sample_size']} patent pairs")
    print(f"📈 Pearson Correlation (Embedding vs LLM): {stats['correlation_analysis']['pearson_correlation']:.3f}")
    print(f"📈 Spearman Correlation (Embedding vs LLM): {stats['correlation_analysis']['spearman_correlation']:.3f}")
    print(f"📏 Mean Absolute Error: {stats['correlation_analysis']['mean_absolute_error']:.3f}")
    print(f"🎯 RMSE: {stats['correlation_analysis']['root_mean_square_error']:.3f}")


if __name__ == "__main__":
    main()