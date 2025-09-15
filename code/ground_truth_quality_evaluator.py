"""Ground truth quality evaluation for embedding models.

This script evaluates embedding model quality using our LLM-generated ground truth dataset,
providing quantitative metrics for comparing different embedding approaches.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from scipy.stats import pearsonr, spearmanr
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Quality metrics for an embedding model."""
    model_name: str
    correlation_with_llm: float
    correlation_p_value: float
    spearman_correlation: float
    mean_similarity_score: float
    std_similarity_score: float
    total_comparisons: int
    high_agreement_rate: float  # When both embedding and LLM agree on high similarity
    low_agreement_rate: float   # When both agree on low similarity


class GroundTruthEvaluator:
    """Evaluate embedding quality using ground truth LLM comparisons."""
    
    def __init__(self, ground_truth_file: str):
        """Initialize with ground truth dataset."""
        self.ground_truth_file = ground_truth_file
        self.ground_truth_data = self.load_ground_truth()
        logger.info(f"Loaded {len(self.ground_truth_data)} ground truth comparisons")
    
    def load_ground_truth(self) -> List[Dict]:
        """Load ground truth dataset."""
        data = []
        
        with open(self.ground_truth_file, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line.strip())
                if record.get('success', False):
                    data.append(record)
        
        logger.info(f"Found {len(data)} successful ground truth evaluations")
        return data
    
    def extract_llm_similarity_scores(self) -> Dict[Tuple[str, str], Dict]:
        """Extract LLM similarity scores and metadata."""
        scores = {}
        
        for record in self.ground_truth_data:
            patent1_id = record['patent1_id']
            patent2_id = record['patent2_id']
            pair_key = (patent1_id, patent2_id)
            
            llm_analysis = record.get('llm_analysis', {})
            similarity_score = llm_analysis.get('similarity_score', 0.0)
            
            scores[pair_key] = {
                'llm_similarity': similarity_score,
                'embedding_similarity': record.get('embedding_similarity', 0.0),
                'similarity_category': record.get('similarity_category', 'unknown'),
                'technical_field_match': llm_analysis.get('technical_field_match', 0.0),
                'problem_similarity': llm_analysis.get('problem_similarity', 0.0),
                'solution_similarity': llm_analysis.get('solution_similarity', 0.0),
                'reasoning': llm_analysis.get('reasoning', ''),
                'classification1': record.get('classification1', ''),
                'classification2': record.get('classification2', '')
            }
        
        return scores
    
    def load_model_embeddings(self, embeddings_file: str) -> Dict[str, np.ndarray]:
        """Load embeddings from model results file."""
        embeddings = {}
        
        if not Path(embeddings_file).exists():
            logger.warning(f"Embeddings file not found: {embeddings_file}")
            return embeddings
        
        with open(embeddings_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                patent_id = data['id']
                
                # Extract embedding from multi-model format
                for model_name, model_data in data.get('models', {}).items():
                    if 'embeddings' in model_data and 'original' in model_data['embeddings']:
                        embedding = model_data['embeddings']['original']['embedding']
                        embeddings[patent_id] = np.array(embedding)
                        break
                
                # Fallback for simple format
                if patent_id not in embeddings and 'embedding' in data:
                    embeddings[patent_id] = np.array(data['embedding'])
        
        logger.info(f"Loaded embeddings for {len(embeddings)} patents")
        return embeddings
    
    def compute_embedding_similarities(self, embeddings: Dict[str, np.ndarray], 
                                     ground_truth_scores: Dict[Tuple[str, str], Dict]) -> Dict[Tuple[str, str], float]:
        """Compute cosine similarities for ground truth pairs."""
        embedding_similarities = {}
        
        for pair_key in ground_truth_scores.keys():
            patent1_id, patent2_id = pair_key
            
            if patent1_id in embeddings and patent2_id in embeddings:
                emb1 = embeddings[patent1_id]
                emb2 = embeddings[patent2_id]
                
                # Cosine similarity
                cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                embedding_similarities[pair_key] = float(cos_sim)
        
        logger.info(f"Computed embedding similarities for {len(embedding_similarities)} pairs")
        return embedding_similarities
    
    def evaluate_model_quality(self, embeddings_file: str, model_name: str) -> QualityMetrics:
        """Evaluate a single model's quality against ground truth."""
        # Load model embeddings
        embeddings = self.load_model_embeddings(embeddings_file)
        
        # Get ground truth scores
        ground_truth_scores = self.extract_llm_similarity_scores()
        
        # Compute embedding similarities for ground truth pairs
        embedding_similarities = self.compute_embedding_similarities(embeddings, ground_truth_scores)
        
        # Prepare data for correlation analysis
        llm_scores = []
        emb_scores = []
        
        for pair_key in ground_truth_scores.keys():
            if pair_key in embedding_similarities:
                llm_scores.append(ground_truth_scores[pair_key]['llm_similarity'])
                emb_scores.append(embedding_similarities[pair_key])
        
        if len(llm_scores) < 10:
            logger.warning(f"Only {len(llm_scores)} matching pairs found for {model_name}")
            return QualityMetrics(model_name, 0, 1, 0, 0, 0, 0, 0, 0)
        
        # Calculate correlations
        pearson_corr, pearson_p = pearsonr(llm_scores, emb_scores)
        spearman_corr, _ = spearmanr(llm_scores, emb_scores)
        
        # Agreement analysis
        high_llm_threshold = 0.7
        low_llm_threshold = 0.3
        high_emb_threshold = 0.7
        low_emb_threshold = 0.3
        
        high_agreements = sum(1 for llm, emb in zip(llm_scores, emb_scores) 
                            if llm >= high_llm_threshold and emb >= high_emb_threshold)
        low_agreements = sum(1 for llm, emb in zip(llm_scores, emb_scores)
                           if llm <= low_llm_threshold and emb <= low_emb_threshold)
        
        high_agreement_rate = high_agreements / len(llm_scores)
        low_agreement_rate = low_agreements / len(llm_scores)
        
        return QualityMetrics(
            model_name=model_name,
            correlation_with_llm=pearson_corr,
            correlation_p_value=pearson_p,
            spearman_correlation=spearman_corr,
            mean_similarity_score=np.mean(emb_scores),
            std_similarity_score=np.std(emb_scores),
            total_comparisons=len(llm_scores),
            high_agreement_rate=high_agreement_rate,
            low_agreement_rate=low_agreement_rate
        )
    
    def compare_multiple_models(self, model_files: Dict[str, str]) -> Dict[str, QualityMetrics]:
        """Compare multiple models against ground truth."""
        results = {}
        
        for model_name, embeddings_file in model_files.items():
            logger.info(f"Evaluating {model_name}...")
            results[model_name] = self.evaluate_model_quality(embeddings_file, model_name)
        
        return results
    
    def generate_comparison_report(self, results: Dict[str, QualityMetrics], 
                                 output_file: str):
        """Generate detailed comparison report."""
        # Create DataFrame for analysis
        data = []
        for model_name, metrics in results.items():
            data.append({
                'Model': model_name,
                'Pearson Correlation': metrics.correlation_with_llm,
                'P-Value': metrics.correlation_p_value,
                'Spearman Correlation': metrics.spearman_correlation,
                'Mean Similarity': metrics.mean_similarity_score,
                'Std Similarity': metrics.std_similarity_score,
                'Total Comparisons': metrics.total_comparisons,
                'High Agreement Rate': metrics.high_agreement_rate,
                'Low Agreement Rate': metrics.low_agreement_rate
            })
        
        df = pd.DataFrame(data)
        
        # Generate report
        with open(output_file, 'w') as f:
            f.write("Ground Truth Quality Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Dataset: {self.ground_truth_file}\n")
            f.write(f"Total ground truth comparisons: {len(self.ground_truth_data)}\n\n")
            
            # Model rankings
            df_sorted = df.sort_values('Pearson Correlation', ascending=False)
            
            f.write("Model Rankings (by Pearson Correlation with LLM):\n")
            f.write("-" * 50 + "\n")
            
            for i, row in df_sorted.iterrows():
                f.write(f"{row['Model']:<20} | r={row['Pearson Correlation']:.3f} ")
                f.write(f"(p={row['P-Value']:.3f}) | ")
                f.write(f"Spearman={row['Spearman Correlation']:.3f}\n")
            
            f.write("\nDetailed Metrics:\n")
            f.write("-" * 50 + "\n")
            f.write(df.to_string(index=False, float_format='%.3f'))
            
            f.write("\n\nInterpretation:\n")
            f.write("-" * 20 + "\n")
            f.write("• Pearson Correlation: Linear relationship with LLM scores (higher = better)\n")
            f.write("• Spearman Correlation: Monotonic relationship (rank-based correlation)\n")
            f.write("• High Agreement Rate: % of cases where both LLM and embedding agree on high similarity\n")
            f.write("• Low Agreement Rate: % of cases where both agree on low similarity\n")
            
            # Best model summary
            best_model = df_sorted.iloc[0]
            f.write(f"\n🏆 Best performing model: {best_model['Model']}\n")
            f.write(f"   Correlation with human-like judgments: {best_model['Pearson Correlation']:.3f}\n")
            f.write(f"   Statistical significance: {'✅ Significant' if best_model['P-Value'] < 0.05 else '❌ Not significant'}\n")
    
    def create_visualizations(self, results: Dict[str, QualityMetrics], 
                            output_dir: str = "."):
        """Create visualizations comparing models."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Data preparation
        models = list(results.keys())
        correlations = [results[m].correlation_with_llm for m in models]
        spearman_corrs = [results[m].spearman_correlation for m in models]
        high_agreements = [results[m].high_agreement_rate for m in models]
        low_agreements = [results[m].low_agreement_rate for m in models]
        
        # 1. Correlation comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pearson correlation
        bars1 = ax1.bar(models, correlations, color='skyblue', alpha=0.8)
        ax1.set_title('Pearson Correlation with LLM Judgments')
        ax1.set_ylabel('Correlation Coefficient')
        ax1.set_ylim(0, max(correlations) * 1.1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, corr in zip(bars1, correlations):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{corr:.3f}', ha='center', va='bottom')
        
        # Spearman correlation
        bars2 = ax2.bar(models, spearman_corrs, color='lightcoral', alpha=0.8)
        ax2.set_title('Spearman Correlation (Rank-based)')
        ax2.set_ylabel('Correlation Coefficient')
        ax2.set_ylim(0, max(spearman_corrs) * 1.1)
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, corr in zip(bars2, spearman_corrs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{corr:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Agreement rates
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, high_agreements, width, label='High Similarity Agreement', color='green', alpha=0.7)
        bars2 = ax.bar(x + width/2, low_agreements, width, label='Low Similarity Agreement', color='red', alpha=0.7)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Agreement Rate')
        ax.set_title('Agreement Rates with LLM Judgments')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                       f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'agreement_rates.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Overall quality score (composite metric)
        quality_scores = []
        for model in models:
            metrics = results[model]
            # Composite score: weighted average of correlation and agreement
            quality_score = (0.5 * metrics.correlation_with_llm + 
                           0.25 * metrics.high_agreement_rate + 
                           0.25 * metrics.low_agreement_rate)
            quality_scores.append(quality_score)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(models, quality_scores, color='gold', alpha=0.8)
        ax.set_title('Overall Quality Score (Composite Metric)')
        ax.set_ylabel('Quality Score')
        ax.set_ylim(0, max(quality_scores) * 1.1)
        ax.tick_params(axis='x', rotation=45)
        
        for bar, score in zip(bars, quality_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'quality_scores.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}")


def main():
    """CLI interface for ground truth evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate embedding quality using ground truth")
    parser.add_argument("ground_truth_file", help="Ground truth JSONL file")
    parser.add_argument("--model-files", nargs='+', required=True,
                       help="Model embedding files (format: model_name:file_path)")
    parser.add_argument("--output-dir", default=".", help="Output directory for reports")
    
    args = parser.parse_args()
    
    # Parse model files
    model_files = {}
    for item in args.model_files:
        if ':' not in item:
            logger.error(f"Invalid format for model file: {item}. Use format 'model_name:file_path'")
            continue
        model_name, file_path = item.split(':', 1)
        model_files[model_name] = file_path
    
    if not model_files:
        logger.error("No valid model files specified")
        return
    
    try:
        evaluator = GroundTruthEvaluator(args.ground_truth_file)
        results = evaluator.compare_multiple_models(model_files)
        
        # Generate report
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        report_file = output_dir / "ground_truth_evaluation_report.txt"
        evaluator.generate_comparison_report(results, str(report_file))
        
        # Create visualizations
        evaluator.create_visualizations(results, str(output_dir))
        
        logger.info(f"Evaluation completed!")
        logger.info(f"Report saved to: {report_file}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()