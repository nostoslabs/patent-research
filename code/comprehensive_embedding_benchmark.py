"""Comprehensive benchmark of all embedding models against 10k ground truth."""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingBenchmark:
    """Comprehensive benchmark suite for embedding models."""

    def __init__(self, ground_truth_file: str = "data/ground_truth/consolidated/ground_truth_10k.jsonl"):
        self.ground_truth_file = ground_truth_file
        self.ground_truth = []
        self.models = {}
        self.results = {}

    def load_ground_truth(self):
        """Load ground truth LLM evaluations."""
        logger.info(f"Loading ground truth from {self.ground_truth_file}")

        with open(self.ground_truth_file, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                self.ground_truth.append(entry)

        logger.info(f"Loaded {len(self.ground_truth)} ground truth pairs")

    def load_embeddings(self, embeddings_dir: str = "data/embeddings/by_model"):
        """Load embeddings for all models."""
        embeddings_path = Path(embeddings_dir)

        for model_dir in embeddings_path.iterdir():
            if model_dir.is_dir():
                model_name = model_dir.name
                logger.info(f"Loading embeddings for {model_name}")

                model_embeddings = {}
                embedding_files = list(model_dir.glob("*.jsonl"))

                for embedding_file in embedding_files:
                    logger.info(f"  Loading {embedding_file.name}")

                    with open(embedding_file, 'r') as f:
                        for line in f:
                            try:
                                entry = json.loads(line.strip())
                                patent_id = entry.get('id') or entry.get('patent_id')
                                embedding = entry.get('embedding')

                                # Handle nested structure: models.{model_name}.embeddings.original.embedding
                                if not embedding and 'models' in entry:
                                    models_data = entry.get('models', {})
                                    for model_key, model_data in models_data.items():
                                        if model_key == model_name:
                                            embeddings_data = model_data.get('embeddings', {})
                                            if 'original' in embeddings_data:
                                                embedding = embeddings_data['original'].get('embedding')
                                            break

                                if patent_id and embedding:
                                    model_embeddings[patent_id] = np.array(embedding)

                            except (json.JSONDecodeError, KeyError) as e:
                                continue

                self.models[model_name] = model_embeddings
                logger.info(f"  Loaded {len(model_embeddings)} embeddings for {model_name}")

        # Also try to load OpenAI embeddings
        openai_files = [
            "openai_embeddings_ground_truth.jsonl",
            "results/openai_embeddings_ground_truth.jsonl",
            "data/embeddings/by_model/openai/openai_embeddings_ground_truth.jsonl"
        ]

        for openai_file in openai_files:
            if Path(openai_file).exists():
                logger.info(f"Loading OpenAI embeddings from {openai_file}")
                openai_embeddings = {}

                with open(openai_file, 'r') as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            patent_id = entry.get('id') or entry.get('patent_id')
                            embedding = entry.get('embedding')

                            if patent_id and embedding:
                                openai_embeddings[patent_id] = np.array(embedding)

                        except (json.JSONDecodeError, KeyError):
                            continue

                if openai_embeddings:
                    self.models['text-embedding-3-small'] = openai_embeddings
                    logger.info(f"  Loaded {len(openai_embeddings)} OpenAI embeddings")
                break

    def calculate_embedding_similarities(self) -> Dict[str, List[float]]:
        """Calculate cosine similarities for each model."""
        model_similarities = {}

        for model_name, embeddings in self.models.items():
            logger.info(f"Calculating similarities for {model_name}")
            similarities = []
            found_pairs = 0

            for gt_pair in self.ground_truth:
                patent1_id = gt_pair['patent1_id']
                patent2_id = gt_pair['patent2_id']

                if patent1_id in embeddings and patent2_id in embeddings:
                    emb1 = embeddings[patent1_id]
                    emb2 = embeddings[patent2_id]

                    # Calculate cosine similarity
                    similarity = cosine_similarity([emb1], [emb2])[0][0]
                    similarities.append(similarity)
                    found_pairs += 1

            model_similarities[model_name] = similarities
            logger.info(f"  Found {found_pairs}/{len(self.ground_truth)} pairs for {model_name}")

        return model_similarities

    def calculate_correlations(self, model_similarities: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """Calculate correlations with LLM ground truth."""
        correlations = {}

        # Calculate correlations for each model independently
        for model_name, similarities in model_similarities.items():
            if not similarities:
                continue

            # Get corresponding LLM scores for this model's valid pairs
            model_llm_scores = []
            found_pairs = 0

            for gt_pair in self.ground_truth:
                patent1_id = gt_pair['patent1_id']
                patent2_id = gt_pair['patent2_id']

                # Only include pairs that have embeddings for this model
                if patent1_id in self.models[model_name] and patent2_id in self.models[model_name]:
                    llm_score = gt_pair['llm_analysis']['similarity_score']
                    model_llm_scores.append(llm_score)
                    found_pairs += 1

                    # Stop when we have enough scores to match similarities
                    if found_pairs >= len(similarities):
                        break

            if len(similarities) == len(model_llm_scores) and len(similarities) > 10:
                # Pearson correlation
                pearson_r, pearson_p = pearsonr(similarities, model_llm_scores)

                # Spearman correlation
                spearman_r, spearman_p = spearmanr(similarities, model_llm_scores)

                correlations[model_name] = {
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p,
                    'n_pairs': len(similarities),
                    'mean_embedding_sim': np.mean(similarities),
                    'std_embedding_sim': np.std(similarities),
                    'mean_llm_score': np.mean(model_llm_scores),
                    'std_llm_score': np.std(model_llm_scores)
                }

        return correlations

    def create_visualizations(self, model_similarities: Dict[str, List[float]], correlations: Dict[str, Dict[str, float]]):
        """Create comprehensive visualization plots."""

        # Create results directory
        results_dir = Path("results/benchmark_analysis")
        results_dir.mkdir(parents=True, exist_ok=True)

        # 1. Correlation comparison plot
        plt.figure(figsize=(12, 8))

        models = list(correlations.keys())
        pearson_values = [correlations[m]['pearson_r'] for m in models]
        spearman_values = [correlations[m]['spearman_r'] for m in models]

        x = np.arange(len(models))
        width = 0.35

        plt.bar(x - width/2, pearson_values, width, label='Pearson r', alpha=0.8)
        plt.bar(x + width/2, spearman_values, width, label='Spearman ρ', alpha=0.8)

        plt.xlabel('Embedding Model')
        plt.ylabel('Correlation with LLM Evaluation')
        plt.title('Embedding Model Performance: Correlation with LLM Ground Truth')
        plt.xticks(x, models, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(results_dir / "correlation_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Scatter plots for each model
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        # Get LLM scores
        llm_scores = [gt['llm_analysis']['similarity_score'] for gt in self.ground_truth]

        for i, (model_name, similarities) in enumerate(model_similarities.items()):
            if i >= len(axes) or len(similarities) == 0:
                if i < len(axes):
                    axes[i].set_visible(False)
                continue

            ax = axes[i]
            n_pairs = len(similarities)
            model_llm_scores = llm_scores[:n_pairs]

            ax.scatter(similarities, model_llm_scores, alpha=0.6, s=20)
            ax.set_xlabel(f'{model_name} Cosine Similarity')
            ax.set_ylabel('LLM Similarity Score')

            if model_name in correlations:
                ax.set_title(f'{model_name}\nr = {correlations[model_name]["pearson_r"]:.3f}')
            else:
                ax.set_title(f'{model_name}\n(no correlation data)')

            ax.grid(True, alpha=0.3)

            # Add trend line
            if len(similarities) > 1:
                z = np.polyfit(similarities, model_llm_scores, 1)
                p = np.poly1d(z)
                ax.plot(similarities, p(similarities), "r--", alpha=0.8)

        plt.tight_layout()
        plt.savefig(results_dir / "model_scatter_plots.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Distribution comparison
        plt.figure(figsize=(14, 8))

        for model_name, similarities in model_similarities.items():
            plt.hist(similarities, bins=50, alpha=0.6, label=model_name, density=True)

        plt.xlabel('Cosine Similarity')
        plt.ylabel('Density')
        plt.title('Distribution of Embedding Similarities Across Models')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(results_dir / "similarity_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Visualizations saved to {results_dir}")

    def generate_summary_report(self, correlations: Dict[str, Dict[str, float]]) -> str:
        """Generate comprehensive summary report."""

        report_lines = [
            "# Comprehensive Embedding Model Benchmark Report",
            "",
            f"**Ground Truth Dataset**: {len(self.ground_truth)} LLM-evaluated patent pairs",
            f"**Models Evaluated**: {len(correlations)}",
            "",
            "## Model Performance Rankings",
            "",
            "| Model | Pearson r | Spearman ρ | p-value | Sample Size | Avg Embedding Sim | Avg LLM Score |",
            "|-------|-----------|------------|---------|-------------|-------------------|---------------|"
        ]

        # Sort by Pearson correlation
        sorted_models = sorted(correlations.items(), key=lambda x: x[1]['pearson_r'], reverse=True)

        for model_name, stats in sorted_models:
            report_lines.append(
                f"| **{model_name}** | {stats['pearson_r']:.3f} | {stats['spearman_r']:.3f} | "
                f"{stats['pearson_p']:.2e} | {stats['n_pairs']:,} | {stats['mean_embedding_sim']:.3f} | "
                f"{stats['mean_llm_score']:.3f} |"
            )

        report_lines.extend([
            "",
            "## Key Findings",
            "",
            f"1. **Best Performing Model**: {sorted_models[0][0]} (r = {sorted_models[0][1]['pearson_r']:.3f})",
            f"2. **Largest Coverage**: {max(correlations.items(), key=lambda x: x[1]['n_pairs'])[0]} "
            f"({max(correlations.values(), key=lambda x: x['n_pairs'])['n_pairs']:,} pairs)",
            "",
            "## Statistical Significance",
            "",
            "All correlations with p < 0.05 are statistically significant:",
        ])

        significant_models = [name for name, stats in correlations.items() if stats['pearson_p'] < 0.05]
        for model in significant_models:
            stats = correlations[model]
            report_lines.append(f"- **{model}**: r = {stats['pearson_r']:.3f}, p = {stats['pearson_p']:.2e}")

        return "\n".join(report_lines)

    def run_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark suite."""
        logger.info("Starting comprehensive embedding benchmark")

        # Load data
        self.load_ground_truth()
        self.load_embeddings()

        if not self.models:
            logger.error("No embedding models found!")
            return {}

        # Calculate similarities
        model_similarities = self.calculate_embedding_similarities()

        if not model_similarities:
            logger.error("No similarities calculated!")
            return {}

        # Calculate correlations
        correlations = self.calculate_correlations(model_similarities)

        # Create visualizations
        self.create_visualizations(model_similarities, correlations)

        # Generate report
        report = self.generate_summary_report(correlations)

        # Save detailed results
        results = {
            'correlations': correlations,
            'model_similarities': {k: v for k, v in model_similarities.items()},  # Convert numpy arrays
            'summary_report': report,
            'ground_truth_size': len(self.ground_truth),
            'models_evaluated': list(self.models.keys())
        }

        results_file = "results/comprehensive_benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

        report_file = "results/COMPREHENSIVE_BENCHMARK_REPORT.md"
        with open(report_file, 'w') as f:
            f.write(report)

        logger.info(f"Benchmark complete! Results saved to {results_file}")
        logger.info(f"Report saved to {report_file}")

        return results


def main():
    benchmark = EmbeddingBenchmark()
    results = benchmark.run_benchmark()

    if results:
        print("\n" + "="*80)
        print("BENCHMARK COMPLETE!")
        print("="*80)
        print(results['summary_report'])


if __name__ == "__main__":
    main()