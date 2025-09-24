"""
Direct OpenAI embedding comparison using existing ground truth data.
This script evaluates OpenAI text-embedding-3-small against our validated models
using the same ground truth dataset for an apples-to-apples comparison.
"""

import json
import numpy as np
import openai
import os
from pathlib import Path
from typing import Dict, List, Tuple
import time
import logging
from scipy.stats import pearsonr, spearmanr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIDirectComparator:
    """Direct comparison of OpenAI embeddings with ground truth."""
    
    def __init__(self, api_key: str = None):
        """Initialize with OpenAI API key."""
        if api_key:
            openai.api_key = api_key
        elif os.getenv('OPENAI_API_KEY'):
            openai.api_key = os.getenv('OPENAI_API_KEY')
        else:
            raise ValueError("OpenAI API key required")
        
        self.model = "text-embedding-3-small"
        self.cost_per_token = 0.02 / 1000000  # $0.02 per 1M tokens
        self.total_tokens = 0
        self.total_cost = 0
        
    def generate_embedding(self, text: str) -> Tuple[np.ndarray, int]:
        """Generate OpenAI embedding for text."""
        try:
            response = openai.embeddings.create(
                model=self.model,
                input=text
            )
            
            embedding = response.data[0].embedding
            tokens_used = response.usage.total_tokens
            
            self.total_tokens += tokens_used
            self.total_cost += tokens_used * self.cost_per_token
            
            return np.array(embedding), tokens_used
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def load_ground_truth_with_abstracts(self, ground_truth_file: str) -> List[Dict]:
        """Load ground truth data and fetch abstracts from patent files."""
        ground_truth_data = []
        
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line.strip())
                if record.get('success', False):
                    ground_truth_data.append(record)
        
        logger.info(f"Loaded {len(ground_truth_data)} ground truth pairs")
        
        # Load patent abstracts from all available files
        patent_abstracts = {}
        patent_files = [
            'data/patent_abstracts_with_embeddings_large.jsonl',
            'data/patent_abstracts_with_embeddings.jsonl',
            'data/patent_abstracts.jsonl'
        ]
        
        for patent_file in patent_files:
            try:
                with open(patent_file, 'r', encoding='utf-8') as f:
                    count = 0
                    for line in f:
                        data = json.loads(line.strip())
                        patent_id = data.get('id')
                        abstract = data.get('abstract', '')
                        
                        if patent_id and abstract and patent_id not in patent_abstracts:
                            patent_abstracts[patent_id] = abstract
                            count += 1
                
                logger.info(f"Loaded {count} patents from {patent_file}")
            except FileNotFoundError:
                logger.warning(f"File not found: {patent_file}")
                continue
        
        logger.info(f"Total patents available: {len(patent_abstracts)}")
        
        # Filter ground truth to only include pairs where we have both abstracts
        filtered_data = []
        for record in ground_truth_data:
            patent1_id = record['patent1_id']
            patent2_id = record['patent2_id']
            
            if patent1_id in patent_abstracts and patent2_id in patent_abstracts:
                record['patent1_abstract'] = patent_abstracts[patent1_id]
                record['patent2_abstract'] = patent_abstracts[patent2_id]
                filtered_data.append(record)
        
        logger.info(f"Ground truth pairs with abstracts available: {len(filtered_data)}")
        return filtered_data
    
    def compute_openai_similarities(self, ground_truth_data: List[Dict]) -> List[Dict]:
        """Compute OpenAI embedding similarities for all ground truth pairs."""
        results = []
        processed = 0
        
        logger.info(f"Computing OpenAI similarities for {len(ground_truth_data)} pairs...")
        
        for record in ground_truth_data:
            try:
                # Generate embeddings for both patents
                abstract1 = record['patent1_abstract']
                abstract2 = record['patent2_abstract']
                
                # Truncate abstracts if needed (OpenAI limit)
                if len(abstract1) > 8000:
                    abstract1 = abstract1[:8000]
                if len(abstract2) > 8000:
                    abstract2 = abstract2[:8000]
                
                embedding1, tokens1 = self.generate_embedding(abstract1)
                embedding2, tokens2 = self.generate_embedding(abstract2)
                
                # Compute cosine similarity
                cos_sim = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
                
                # Add OpenAI results to record
                result = record.copy()
                result['openai_embedding_similarity'] = float(cos_sim)
                result['openai_tokens_used'] = tokens1 + tokens2
                results.append(result)
                
                processed += 1
                if processed % 10 == 0:
                    logger.info(f"Processed {processed}/{len(ground_truth_data)} pairs. "
                               f"Total cost: ${self.total_cost:.4f}")
                
                # Rate limiting to avoid hitting API limits
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Failed to process pair {record['patent1_id']}-{record['patent2_id']}: {e}")
                continue
        
        logger.info(f"Completed! Total tokens: {self.total_tokens}, Total cost: ${self.total_cost:.4f}")
        return results
    
    def compare_with_existing_models(self, results: List[Dict]) -> Dict:
        """Compare OpenAI results with existing model results."""
        # Extract LLM scores (ground truth)
        llm_scores = []
        openai_scores = []
        existing_embedding_scores = []
        
        for result in results:
            llm_similarity = result['llm_analysis']['similarity_score']
            openai_similarity = result['openai_embedding_similarity']
            existing_similarity = result.get('embedding_similarity', 0.0)
            
            llm_scores.append(llm_similarity)
            openai_scores.append(openai_similarity)
            existing_embedding_scores.append(existing_similarity)
        
        # Calculate correlations
        openai_pearson, openai_p = pearsonr(llm_scores, openai_scores)
        existing_pearson, existing_p = pearsonr(llm_scores, existing_embedding_scores)
        
        openai_spearman, _ = spearmanr(llm_scores, openai_scores)
        existing_spearman, _ = spearmanr(llm_scores, existing_embedding_scores)
        
        comparison = {
            'openai_model': {
                'name': self.model,
                'pearson_correlation': openai_pearson,
                'pearson_p_value': openai_p,
                'spearman_correlation': openai_spearman,
                'mean_similarity': np.mean(openai_scores),
                'std_similarity': np.std(openai_scores),
                'total_comparisons': len(openai_scores),
                'total_cost': self.total_cost,
                'total_tokens': self.total_tokens
            },
            'existing_embeddings': {
                'name': 'existing_model',
                'pearson_correlation': existing_pearson,
                'pearson_p_value': existing_p,
                'spearman_correlation': existing_spearman,
                'mean_similarity': np.mean(existing_embedding_scores),
                'std_similarity': np.std(existing_embedding_scores),
                'total_comparisons': len(existing_embedding_scores)
            }
        }
        
        return comparison
    
    def generate_comparison_report(self, comparison: Dict, output_file: str):
        """Generate detailed comparison report."""
        with open(output_file, 'w') as f:
            f.write("OpenAI vs Existing Models - Direct Comparison Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Dataset: Ground truth with {comparison['openai_model']['total_comparisons']} pairs\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # OpenAI results
            openai = comparison['openai_model']
            f.write("OpenAI text-embedding-3-small Results:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Pearson Correlation: {openai['pearson_correlation']:.3f} (p={openai['pearson_p_value']:.3f})\n")
            f.write(f"Spearman Correlation: {openai['spearman_correlation']:.3f}\n")
            f.write(f"Mean Similarity: {openai['mean_similarity']:.3f}\n")
            f.write(f"Std Similarity: {openai['std_similarity']:.3f}\n")
            f.write(f"Total Cost: ${openai['total_cost']:.4f}\n")
            f.write(f"Total Tokens: {openai['total_tokens']}\n\n")
            
            # Existing model results
            existing = comparison['existing_embeddings']
            f.write("Existing Embedding Model Results:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Pearson Correlation: {existing['pearson_correlation']:.3f} (p={existing['pearson_p_value']:.3f})\n")
            f.write(f"Spearman Correlation: {existing['spearman_correlation']:.3f}\n")
            f.write(f"Mean Similarity: {existing['mean_similarity']:.3f}\n")
            f.write(f"Std Similarity: {existing['std_similarity']:.3f}\n")
            f.write("Total Cost: $0.00\n\n")
            
            # Quality comparison
            f.write("Quality Comparison:\n")
            f.write("-" * 20 + "\n")
            
            if openai['pearson_correlation'] > existing['pearson_correlation']:
                improvement = ((openai['pearson_correlation'] - existing['pearson_correlation']) / 
                             existing['pearson_correlation']) * 100
                f.write(f"🏆 OpenAI performs better: {improvement:.1f}% improvement\n")
            else:
                improvement = ((existing['pearson_correlation'] - openai['pearson_correlation']) / 
                             openai['pearson_correlation']) * 100
                f.write(f"🏆 Existing model performs better: {improvement:.1f}% improvement\n")
            
            # Statistical significance
            f.write(f"\nStatistical Significance:\n")
            f.write(f"OpenAI: {'✅ Significant' if openai['pearson_p_value'] < 0.05 else '❌ Not significant'}\n")
            f.write(f"Existing: {'✅ Significant' if existing['pearson_p_value'] < 0.05 else '❌ Not significant'}\n")


def main():
    """Run direct OpenAI comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Direct OpenAI embedding comparison")
    parser.add_argument("ground_truth_file", help="Ground truth JSONL file")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--max-pairs", type=int, default=100, help="Maximum pairs to process")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        comparator = OpenAIDirectComparator(args.api_key)
        
        # Load ground truth data with abstracts
        ground_truth_data = comparator.load_ground_truth_with_abstracts(args.ground_truth_file)
        
        # Limit the number of pairs if specified
        if args.max_pairs and len(ground_truth_data) > args.max_pairs:
            ground_truth_data = ground_truth_data[:args.max_pairs]
            logger.info(f"Limited to {args.max_pairs} pairs for evaluation")
        
        # Compute OpenAI similarities
        results = comparator.compute_openai_similarities(ground_truth_data)
        
        # Save detailed results
        results_file = output_dir / "openai_direct_comparison_results.jsonl"
        with open(results_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        
        # Compare with existing models
        comparison = comparator.compare_with_existing_models(results)
        
        # Generate report
        report_file = output_dir / "openai_direct_comparison_report.txt"
        comparator.generate_comparison_report(comparison, str(report_file))
        
        logger.info("Direct comparison completed!")
        logger.info(f"Results: {results_file}")
        logger.info(f"Report: {report_file}")
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        raise


if __name__ == "__main__":
    main()