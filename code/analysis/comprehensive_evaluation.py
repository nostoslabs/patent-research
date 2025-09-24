"""Comprehensive evaluation pipeline for patent similarity search systems."""

import asyncio
import json
import logging
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
from collections import defaultdict

from ground_truth_generator import GroundTruthGenerator, PatentPair
from google_patents_baseline import BaselineComparator
from cross_encoder_reranker import TwoStageSearchSystem
from llm_provider_factory import LLMProvider

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingModelResult:
    """Results for a single embedding model."""
    model_name: str
    embeddings_file: str
    search_results: Dict[str, Any]  # Results from different search methods
    baseline_comparison: Dict[str, Any]
    reranking_results: Dict[str, Any]
    ground_truth_evaluation: Dict[str, Any]
    performance_metrics: Dict[str, float]


@dataclass
class ComprehensiveResults:
    """Complete evaluation results across all models and methods."""
    experiment_name: str
    timestamp: str
    models_evaluated: List[str]
    evaluation_methods: List[str]
    model_results: Dict[str, EmbeddingModelResult]
    comparative_analysis: Dict[str, Any]
    recommendations: Dict[str, Any]


class ComprehensiveEvaluator:
    """Comprehensive evaluation system for patent similarity search."""
    
    def __init__(self, 
                 embedding_files: List[str],
                 llm_provider: Optional[LLMProvider] = None):
        """Initialize with multiple embedding files and LLM provider."""
        self.embedding_files = embedding_files
        self.llm_provider = llm_provider
        self.model_names = [self._extract_model_name(f) for f in embedding_files]
        
        logger.info(f"Initialized evaluator for {len(embedding_files)} models")
        logger.info(f"Models: {', '.join(self.model_names)}")
    
    def _extract_model_name(self, filename: str) -> str:
        """Extract model name from embeddings filename."""
        # Extract model name from pattern like: diverse_10k_full_nomic-embed-text_...
        parts = Path(filename).stem.split('_')
        for i, part in enumerate(parts):
            if part in ['nomic-embed-text', 'bge-m3', 'embeddinggemma', 'mxbai-embed-large']:
                return part
            elif 'nomic' in part or 'bge' in part or 'embedding' in part or 'mxbai' in part:
                return part
        return parts[-1] if parts else 'unknown'
    
    async def run_comprehensive_evaluation(self,
                                         test_patent_ids: List[str],
                                         output_dir: str = "evaluation_results",
                                         n_ground_truth_pairs: int = 100) -> ComprehensiveResults:
        """Run complete evaluation across all models and methods."""
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        experiment_name = f"comprehensive_eval_{timestamp}"
        
        logger.info(f"Starting comprehensive evaluation: {experiment_name}")
        logger.info(f"Test patents: {len(test_patent_ids)}")
        logger.info(f"Ground truth pairs: {n_ground_truth_pairs}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        model_results = {}
        
        for i, (model_name, embeddings_file) in enumerate(zip(self.model_names, self.embedding_files)):
            logger.info(f"Evaluating model {i+1}/{len(self.model_names)}: {model_name}")
            
            try:
                # Run all evaluation methods for this model
                result = await self._evaluate_single_model(
                    model_name, 
                    embeddings_file, 
                    test_patent_ids,
                    n_ground_truth_pairs,
                    output_path
                )
                
                model_results[model_name] = result
                
                # Save intermediate results
                intermediate_file = output_path / f"{model_name}_results.json"
                with open(intermediate_file, 'w', encoding='utf-8') as f:
                    json.dump(asdict(result), f, indent=2, ensure_ascii=False)
                
            except Exception as e:
                logger.error(f"Error evaluating model {model_name}: {e}")
                # Continue with other models
                continue
        
        # Perform comparative analysis
        logger.info("Performing comparative analysis...")
        comparative_analysis = self._analyze_comparative_results(model_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(model_results, comparative_analysis)
        
        # Create comprehensive results
        results = ComprehensiveResults(
            experiment_name=experiment_name,
            timestamp=timestamp,
            models_evaluated=list(model_results.keys()),
            evaluation_methods=['embedding_search', 'baseline_comparison', 'reranking', 'ground_truth'],
            model_results=model_results,
            comparative_analysis=comparative_analysis,
            recommendations=recommendations
        )
        
        # Save comprehensive results
        results_file = output_path / f"{experiment_name}_comprehensive.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(results), f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Comprehensive evaluation completed!")
        logger.info(f"Results saved to: {results_file}")
        
        return results
    
    async def _evaluate_single_model(self,
                                   model_name: str,
                                   embeddings_file: str,
                                   test_patent_ids: List[str],
                                   n_ground_truth_pairs: int,
                                   output_path: Path) -> EmbeddingModelResult:
        """Evaluate a single embedding model across all methods."""
        
        logger.info(f"Starting evaluation for {model_name}")
        
        # Initialize components
        baseline_comparator = BaselineComparator(embeddings_file)
        two_stage_system = TwoStageSearchSystem(embeddings_file, self.llm_provider)
        ground_truth_generator = GroundTruthGenerator(embeddings_file, self.llm_provider)
        
        # 1. Baseline comparison
        logger.info(f"Running baseline comparison for {model_name}...")
        baseline_output = output_path / f"{model_name}_baseline.jsonl"
        baseline_results = await baseline_comparator.run_baseline_study(
            test_patent_ids[:5],  # Limit for efficiency
            str(baseline_output),
            delay_between_queries=2.0
        )
        
        # 2. Cross-encoder reranking evaluation
        logger.info(f"Running reranking evaluation for {model_name}...")
        reranking_output = output_path / f"{model_name}_reranking.jsonl"
        reranking_results = await two_stage_system.batch_evaluate(
            test_patent_ids[:5],  # Limit for efficiency
            str(reranking_output),
            top_k_initial=20,
            top_k_rerank=10,
            top_k_final=10
        )
        
        # 3. Ground truth evaluation
        logger.info(f"Running ground truth evaluation for {model_name}...")
        ground_truth_output = output_path / f"{model_name}_ground_truth.jsonl"
        ground_truth_file = await ground_truth_generator.generate_ground_truth_dataset(
            n_pairs=min(n_ground_truth_pairs, 50),  # Limit for efficiency
            output_file=str(ground_truth_output),
            max_concurrent=3
        )
        
        # Load ground truth results for analysis
        ground_truth_data = []
        try:
            with open(ground_truth_file, 'r', encoding='utf-8') as f:
                for line in f:
                    ground_truth_data.append(json.loads(line.strip()))
        except Exception as e:
            logger.warning(f"Could not load ground truth data: {e}")
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(
            baseline_results, reranking_results, ground_truth_data
        )
        
        # Create model result
        result = EmbeddingModelResult(
            model_name=model_name,
            embeddings_file=embeddings_file,
            search_results={'basic_embedding_search': 'completed'},
            baseline_comparison=baseline_results,
            reranking_results=reranking_results,
            ground_truth_evaluation={'file': ground_truth_file, 'pairs_generated': len(ground_truth_data)},
            performance_metrics=performance_metrics
        )
        
        logger.info(f"Completed evaluation for {model_name}")
        return result
    
    def _calculate_performance_metrics(self,
                                     baseline_results: Dict,
                                     reranking_results: Dict,
                                     ground_truth_data: List[Dict]) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        
        metrics = {}
        
        # Baseline comparison metrics
        if baseline_results:
            metrics['baseline_avg_precision_5'] = baseline_results.get('average_precision_at_5', 0)
            metrics['baseline_avg_overlap'] = baseline_results.get('average_overlap_count', 0)
            metrics['baseline_success_rate'] = baseline_results.get('success_rate', 0)
        
        # Reranking metrics
        if reranking_results:
            metrics['reranking_avg_time'] = reranking_results.get('average_reranking_time', 0)
            metrics['reranking_avg_rank_change'] = reranking_results.get('average_rank_change', 0)
            metrics['reranking_avg_llm_score'] = reranking_results.get('average_llm_score', 0)
            metrics['reranking_success_rate'] = reranking_results.get('success_rate', 0)
        
        # Ground truth metrics
        if ground_truth_data:
            llm_scores = []
            embedding_scores = []
            
            for item in ground_truth_data:
                if item.get('success') and 'llm_analysis' in item:
                    llm_analysis = item['llm_analysis']
                    if isinstance(llm_analysis, dict):
                        llm_scores.append(llm_analysis.get('similarity_score', 0))
                        embedding_scores.append(item.get('embedding_similarity', 0))
            
            if llm_scores and embedding_scores:
                # Correlation between embedding and LLM scores
                correlation = np.corrcoef(embedding_scores, llm_scores)[0, 1] if len(llm_scores) > 1 else 0
                metrics['embedding_llm_correlation'] = float(correlation) if not np.isnan(correlation) else 0
                metrics['avg_llm_similarity'] = np.mean(llm_scores)
                metrics['avg_embedding_similarity'] = np.mean(embedding_scores)
                metrics['ground_truth_pairs'] = len(llm_scores)
        
        return metrics
    
    def _analyze_comparative_results(self, model_results: Dict[str, EmbeddingModelResult]) -> Dict[str, Any]:
        """Analyze results across all models."""
        
        if not model_results:
            return {}
        
        analysis = {
            'model_rankings': {},
            'performance_summary': {},
            'strengths_weaknesses': {}
        }
        
        # Collect metrics for comparison
        metrics_data = defaultdict(list)
        model_names = []
        
        for model_name, result in model_results.items():
            model_names.append(model_name)
            for metric_name, metric_value in result.performance_metrics.items():
                metrics_data[metric_name].append(metric_value)
        
        # Rank models by key metrics
        key_metrics = [
            'baseline_avg_precision_5',
            'reranking_avg_llm_score', 
            'embedding_llm_correlation',
            'reranking_success_rate'
        ]
        
        for metric in key_metrics:
            if metric in metrics_data and len(metrics_data[metric]) > 0:
                values = metrics_data[metric]
                ranked_indices = np.argsort(values)[::-1]  # Descending order
                
                analysis['model_rankings'][metric] = [
                    {'model': model_names[i], 'score': values[i]} 
                    for i in ranked_indices
                ]
        
        # Overall performance summary
        analysis['performance_summary'] = {
            'best_baseline_precision': self._get_best_model(model_results, 'baseline_avg_precision_5'),
            'best_reranking_performance': self._get_best_model(model_results, 'reranking_avg_llm_score'),
            'best_embedding_llm_agreement': self._get_best_model(model_results, 'embedding_llm_correlation'),
            'fastest_reranking': self._get_best_model(model_results, 'reranking_avg_time', ascending=True)
        }
        
        return analysis
    
    def _get_best_model(self, model_results: Dict, metric_name: str, ascending: bool = False) -> Dict:
        """Get the best performing model for a specific metric."""
        
        best_model = None
        best_score = float('inf') if ascending else float('-inf')
        
        for model_name, result in model_results.items():
            score = result.performance_metrics.get(metric_name)
            if score is not None:
                if (ascending and score < best_score) or (not ascending and score > best_score):
                    best_score = score
                    best_model = model_name
        
        return {'model': best_model, 'score': best_score} if best_model else {}
    
    def _generate_recommendations(self, 
                                model_results: Dict[str, EmbeddingModelResult],
                                comparative_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations based on evaluation results."""
        
        recommendations = {
            'production_deployment': {},
            'use_case_specific': {},
            'optimization_suggestions': []
        }
        
        if not model_results:
            return recommendations
        
        # Production deployment recommendations
        performance_summary = comparative_analysis.get('performance_summary', {})
        
        # Overall best model based on multiple factors
        model_scores = defaultdict(float)
        
        for model_name, result in model_results.items():
            metrics = result.performance_metrics
            
            # Weight different metrics
            score = 0
            score += metrics.get('baseline_avg_precision_5', 0) * 0.3
            score += metrics.get('reranking_avg_llm_score', 0) * 0.3
            score += metrics.get('embedding_llm_correlation', 0) * 0.2
            score += metrics.get('reranking_success_rate', 0) * 0.2
            
            model_scores[model_name] = score
        
        # Best overall model
        best_overall = max(model_scores.items(), key=lambda x: x[1]) if model_scores else None
        
        recommendations['production_deployment'] = {
            'recommended_model': best_overall[0] if best_overall else 'unknown',
            'overall_score': best_overall[1] if best_overall else 0,
            'reasoning': f"Best combination of baseline precision, reranking performance, and LLM agreement"
        }
        
        # Use case specific recommendations
        recommendations['use_case_specific'] = {
            'speed_critical': performance_summary.get('fastest_reranking', {}),
            'quality_critical': performance_summary.get('best_reranking_performance', {}),
            'baseline_search': performance_summary.get('best_baseline_precision', {})
        }
        
        # Optimization suggestions
        suggestions = []
        
        # Check for low correlation models
        for model_name, result in model_results.items():
            correlation = result.performance_metrics.get('embedding_llm_correlation', 0)
            if correlation < 0.3:
                suggestions.append(f"{model_name}: Low embedding-LLM correlation ({correlation:.2f}) - consider fine-tuning")
        
        # Check for slow reranking
        for model_name, result in model_results.items():
            rerank_time = result.performance_metrics.get('reranking_avg_time', 0)
            if rerank_time > 20:
                suggestions.append(f"{model_name}: Slow reranking ({rerank_time:.1f}s) - consider reducing candidates or using faster LLM")
        
        recommendations['optimization_suggestions'] = suggestions
        
        return recommendations


async def main():
    """CLI interface for comprehensive evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive patent similarity search evaluation")
    parser.add_argument("--embedding-files", nargs='+', required=True, 
                       help="List of embedding JSONL files to evaluate")
    parser.add_argument("--test-patents", nargs='+', 
                       help="Specific patent IDs to test with")
    parser.add_argument("--sample-size", type=int, default=10,
                       help="Random sample size for testing")
    parser.add_argument("--ground-truth-pairs", type=int, default=50,
                       help="Number of ground truth pairs to generate")
    parser.add_argument("--output-dir", default="evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--provider", choices=['openai', 'google', 'anthropic', 'ollama'],
                       help="LLM provider for ground truth and reranking")
    
    args = parser.parse_args()
    
    # Validate embedding files
    for file in args.embedding_files:
        if not Path(file).exists():
            logger.error(f"Embedding file not found: {file}")
            return
    
    # Convert string to enum if provided
    provider = None
    if args.provider:
        provider = LLMProvider(args.provider)
    
    try:
        evaluator = ComprehensiveEvaluator(args.embedding_files, provider)
        
        # Determine test patents
        if args.test_patents:
            test_patent_ids = args.test_patents
        else:
            # Get random sample from first embedding file
            from cross_encoder_reranker import EmbeddingSearchEngine
            searcher = EmbeddingSearchEngine(args.embedding_files[0])
            available_patents = [p['id'] for p in searcher.embeddings]
            
            import random
            random.seed(42)
            test_patent_ids = random.sample(
                available_patents, 
                min(args.sample_size, len(available_patents))
            )
        
        logger.info(f"Running comprehensive evaluation with {len(test_patent_ids)} test patents")
        
        results = await evaluator.run_comprehensive_evaluation(
            test_patent_ids=test_patent_ids,
            output_dir=args.output_dir,
            n_ground_truth_pairs=args.ground_truth_pairs
        )
        
        # Print summary
        print("\n" + "="*80)
        print("COMPREHENSIVE EVALUATION RESULTS")
        print("="*80)
        print(f"Experiment: {results.experiment_name}")
        print(f"Models evaluated: {len(results.models_evaluated)}")
        print(f"Evaluation methods: {len(results.evaluation_methods)}")
        
        # Print recommendations
        rec = results.recommendations
        prod_rec = rec.get('production_deployment', {})
        if prod_rec.get('recommended_model'):
            print(f"\nRecommended model: {prod_rec['recommended_model']}")
            print(f"Overall score: {prod_rec['overall_score']:.3f}")
            print(f"Reasoning: {prod_rec.get('reasoning', '')}")
        
        print(f"\nResults saved to: {args.output_dir}/")
        
    except Exception as e:
        logger.error(f"Failed to run comprehensive evaluation: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())