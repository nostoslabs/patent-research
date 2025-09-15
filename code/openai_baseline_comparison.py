"""OpenAI baseline comparison for PatentXpedited.com methodology evaluation.

This script implements the exact embedding approach used by PatentXpedited.com
(OpenAI text-embedding-3-small) and compares it against our validated models.
"""

import json
import time
import asyncio
import logging
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import openai
from sklearn.metrics.pairwise import cosine_similarity

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Results from embedding generation."""
    patent_id: str
    abstract: str
    embedding: List[float]
    embedding_dim: int
    processing_time: float
    tokens_used: int
    cost_estimate: float
    needs_truncation: bool
    original_length: int
    truncated_length: int
    model: str


@dataclass 
class ComparisonMetrics:
    """Comparison metrics between models."""
    model_name: str
    total_patents: int
    avg_processing_time: float
    total_tokens: int
    total_cost: float
    embedding_dimension: int
    truncation_required: int
    avg_truncation_percentage: float
    quality_score: Optional[float] = None
    

class OpenAIEmbeddingGenerator:
    """Generate embeddings using OpenAI's text-embedding-3-small model."""
    
    def __init__(self):
        """Initialize OpenAI client."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = openai.OpenAI(api_key=api_key)
        self.model = "text-embedding-3-small"
        self.max_tokens = 8192  # OpenAI's context window
        self.cost_per_token = 0.02 / 1000000  # $0.02 per 1M tokens
        
        logger.info(f"Initialized OpenAI embedding generator with model: {self.model}")
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: 1 token ≈ 4 characters)."""
        return len(text) // 4
    
    def truncate_text(self, text: str, max_tokens: int = 8192) -> tuple[str, bool]:
        """Truncate text to fit within token limits."""
        estimated_tokens = self.estimate_tokens(text)
        
        if estimated_tokens <= max_tokens:
            return text, False
        
        # Truncate to approximately fit within token limit
        # Leave some buffer for safety
        target_chars = int(max_tokens * 4 * 0.9)  # 90% of estimated limit
        truncated = text[:target_chars]
        
        logger.warning(f"Truncated text from {len(text)} to {len(truncated)} characters")
        return truncated, True
    
    async def generate_embedding(self, patent: Dict[str, Any]) -> EmbeddingResult:
        """Generate embedding for a single patent."""
        start_time = time.time()
        
        patent_id = patent['id']
        abstract = patent['abstract']
        original_length = len(abstract)
        
        # Truncate if necessary (like PatentXpedited does)
        text_to_embed, needs_truncation = self.truncate_text(abstract, self.max_tokens)
        truncated_length = len(text_to_embed)
        
        try:
            response = await asyncio.to_thread(
                self.client.embeddings.create,
                model=self.model,
                input=text_to_embed,
                encoding_format="float"
            )
            
            embedding = response.data[0].embedding
            tokens_used = response.usage.total_tokens if response.usage else self.estimate_tokens(text_to_embed)
            cost = tokens_used * self.cost_per_token
            
            processing_time = time.time() - start_time
            
            return EmbeddingResult(
                patent_id=patent_id,
                abstract=abstract[:200] + "..." if len(abstract) > 200 else abstract,
                embedding=embedding,
                embedding_dim=len(embedding),
                processing_time=processing_time,
                tokens_used=tokens_used,
                cost_estimate=cost,
                needs_truncation=needs_truncation,
                original_length=original_length,
                truncated_length=truncated_length,
                model=self.model
            )
            
        except Exception as e:
            logger.error(f"Error generating embedding for patent {patent_id}: {e}")
            raise


class BaselineComparison:
    """Compare OpenAI embeddings against our validated models."""
    
    def __init__(self):
        """Initialize comparison framework."""
        self.openai_generator = OpenAIEmbeddingGenerator()
    
    def load_patents(self, file_path: str, max_records: int = 100) -> List[Dict]:
        """Load patents from JSONL file."""
        patents = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_records:
                    break
                    
                data = json.loads(line.strip())
                patents.append({
                    'id': data['id'],
                    'abstract': data['abstract'],
                    'classification': data.get('classification', ''),
                    'abstract_length': len(data['abstract'])
                })
        
        logger.info(f"Loaded {len(patents)} patents for comparison")
        return patents
    
    def load_existing_embeddings(self, file_path: str) -> Dict[str, Dict]:
        """Load existing embeddings from our experiments."""
        embeddings = {}
        
        if not Path(file_path).exists():
            logger.warning(f"Existing embeddings file not found: {file_path}")
            return embeddings
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                patent_id = data['id']
                embeddings[patent_id] = data
        
        logger.info(f"Loaded {len(embeddings)} existing embeddings")
        return embeddings
    
    async def generate_openai_embeddings(self, patents: List[Dict], 
                                       output_file: str,
                                       max_concurrent: int = 3) -> List[EmbeddingResult]:
        """Generate OpenAI embeddings for patents."""
        logger.info(f"Generating OpenAI embeddings for {len(patents)} patents")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        results = []
        
        async def process_patent(patent: Dict) -> EmbeddingResult:
            async with semaphore:
                # Add delay to respect rate limits
                await asyncio.sleep(0.5)
                return await self.openai_generator.generate_embedding(patent)
        
        # Process all patents
        for i in range(0, len(patents), 10):
            batch = patents[i:i + 10]
            logger.info(f"Processing batch {i//10 + 1}/{len(patents)//10 + 1}")
            
            batch_results = await asyncio.gather(
                *[process_patent(patent) for patent in batch],
                return_exceptions=True
            )
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Exception in batch processing: {result}")
                else:
                    results.append(result)
            
            # Save intermediate results
            if results:
                self._save_intermediate_results(results, f"{output_file}.partial")
            
            logger.info(f"Completed {len(results)}/{len(patents)} patents")
        
        # Save final results
        self._save_results(results, output_file)
        
        return results
    
    def _save_intermediate_results(self, results: List[EmbeddingResult], filename: str):
        """Save intermediate results to prevent data loss."""
        with open(filename, 'w', encoding='utf-8') as f:
            for result in results:
                json.dump(asdict(result), f, ensure_ascii=False)
                f.write('\n')
    
    def _save_results(self, results: List[EmbeddingResult], filename: str):
        """Save final results in our standard format."""
        with open(filename, 'w', encoding='utf-8') as f:
            for result in results:
                # Convert to our standard embedding format
                output = {
                    'id': result.patent_id,
                    'abstract': result.abstract,
                    'abstract_length': result.original_length,
                    'models': {
                        result.model: {
                            'embeddings': {
                                'original': {
                                    'embedding': result.embedding,
                                    'embedding_dim': result.embedding_dim
                                }
                            },
                            'needs_chunking': result.needs_truncation,
                            'processing_time': result.processing_time,
                            'tokens_used': result.tokens_used,
                            'cost_estimate': result.cost_estimate,
                            'truncation_info': {
                                'original_length': result.original_length,
                                'truncated_length': result.truncated_length,
                                'truncation_percentage': (result.original_length - result.truncated_length) / result.original_length if result.original_length > 0 else 0
                            }
                        }
                    }
                }
                json.dump(output, f, ensure_ascii=False)
                f.write('\n')
    
    def calculate_metrics(self, results: List[EmbeddingResult]) -> ComparisonMetrics:
        """Calculate comparison metrics."""
        if not results:
            return ComparisonMetrics("openai", 0, 0, 0, 0, 0, 0, 0)
        
        total_processing_time = sum(r.processing_time for r in results)
        total_tokens = sum(r.tokens_used for r in results)
        total_cost = sum(r.cost_estimate for r in results)
        truncation_count = sum(1 for r in results if r.needs_truncation)
        
        truncation_percentages = [
            (r.original_length - r.truncated_length) / r.original_length 
            for r in results if r.needs_truncation and r.original_length > 0
        ]
        avg_truncation = np.mean(truncation_percentages) if truncation_percentages else 0
        
        return ComparisonMetrics(
            model_name=results[0].model,
            total_patents=len(results),
            avg_processing_time=total_processing_time / len(results),
            total_tokens=total_tokens,
            total_cost=total_cost,
            embedding_dimension=results[0].embedding_dim,
            truncation_required=truncation_count,
            avg_truncation_percentage=avg_truncation * 100
        )
    
    def compare_with_existing(self, openai_results: List[EmbeddingResult],
                            existing_embeddings: Dict[str, Dict]) -> Dict[str, Any]:
        """Compare OpenAI results with existing model results."""
        comparison = {
            'openai_metrics': self.calculate_metrics(openai_results),
            'existing_models': {},
            'quality_comparison': {},
            'cost_analysis': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Extract metrics from existing models
        model_stats = {}
        for patent_id, data in existing_embeddings.items():
            for model_name, model_data in data.get('models', {}).items():
                if model_name not in model_stats:
                    model_stats[model_name] = {
                        'total_patents': 0,
                        'total_processing_time': 0,
                        'truncation_count': 0,
                        'embedding_dims': set(),
                        'truncation_percentages': []
                    }
                
                stats = model_stats[model_name]
                stats['total_patents'] += 1
                
                if 'processing_time' in model_data:
                    stats['total_processing_time'] += model_data['processing_time']
                
                if model_data.get('needs_chunking', False):
                    stats['truncation_count'] += 1
                
                original_emb = model_data.get('embeddings', {}).get('original', {})
                if 'embedding_dim' in original_emb:
                    stats['embedding_dims'].add(original_emb['embedding_dim'])
        
        # Convert to comparison metrics
        for model_name, stats in model_stats.items():
            if stats['total_patents'] > 0:
                comparison['existing_models'][model_name] = ComparisonMetrics(
                    model_name=model_name,
                    total_patents=stats['total_patents'],
                    avg_processing_time=stats['total_processing_time'] / stats['total_patents'],
                    total_tokens=0,  # Local models don't use tokens
                    total_cost=0.0,  # Local models are free
                    embedding_dimension=list(stats['embedding_dims'])[0] if stats['embedding_dims'] else 0,
                    truncation_required=stats['truncation_count'],
                    avg_truncation_percentage=(stats['truncation_count'] / stats['total_patents']) * 100
                )
        
        return comparison
    
    async def run_comparison(self, patent_file: str, 
                           existing_embeddings_file: str,
                           output_file: str,
                           max_records: int = 100) -> str:
        """Run complete comparison between OpenAI and existing models."""
        logger.info("Starting OpenAI baseline comparison")
        
        # Load data
        patents = self.load_patents(patent_file, max_records)
        existing_embeddings = self.load_existing_embeddings(existing_embeddings_file)
        
        # Generate OpenAI embeddings
        openai_results = await self.generate_openai_embeddings(patents, output_file)
        
        # Compare with existing models
        comparison_data = self.compare_with_existing(openai_results, existing_embeddings)
        
        # Save comparison results
        comparison_file = output_file.replace('.jsonl', '_comparison.json')
        with open(comparison_file, 'w', encoding='utf-8') as f:
            # Convert ComparisonMetrics objects to dictionaries for JSON serialization
            serializable_data = {}
            for key, value in comparison_data.items():
                if key == 'openai_metrics':
                    serializable_data[key] = asdict(value)
                elif key == 'existing_models':
                    serializable_data[key] = {k: asdict(v) for k, v in value.items()}
                else:
                    serializable_data[key] = value
            
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        
        # Generate summary report
        self._generate_summary_report(comparison_data, comparison_file.replace('.json', '_summary.txt'))
        
        logger.info(f"Comparison completed. Results saved to: {output_file}")
        logger.info(f"Comparison analysis saved to: {comparison_file}")
        
        return comparison_file
    
    def _generate_summary_report(self, comparison_data: Dict, output_file: str):
        """Generate human-readable summary report."""
        openai_metrics = comparison_data['openai_metrics']
        existing_models = comparison_data['existing_models']
        
        with open(output_file, 'w') as f:
            f.write("OpenAI Baseline Comparison Report\n")
            f.write("=" * 50 + "\n\n")
            
            # OpenAI Performance
            f.write("OpenAI text-embedding-3-small Performance:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total patents processed: {openai_metrics.total_patents}\n")
            f.write(f"Average processing time: {openai_metrics.avg_processing_time:.2f}s per patent\n")
            f.write(f"Total tokens used: {openai_metrics.total_tokens:,}\n")
            f.write(f"Total cost: ${openai_metrics.total_cost:.4f}\n")
            f.write(f"Embedding dimension: {openai_metrics.embedding_dimension}\n")
            f.write(f"Patents requiring truncation: {openai_metrics.truncation_required}/{openai_metrics.total_patents}")
            f.write(f" ({openai_metrics.truncation_required/openai_metrics.total_patents*100:.1f}%)\n")
            f.write(f"Average truncation: {openai_metrics.avg_truncation_percentage:.1f}%\n\n")
            
            # Comparison with existing models
            f.write("Comparison with Existing Models:\n")
            f.write("-" * 40 + "\n")
            
            for model_name, metrics in existing_models.items():
                f.write(f"\n{model_name.upper()}:\n")
                f.write(f"  Processing time: {metrics.avg_processing_time:.2f}s per patent ")
                if metrics.avg_processing_time > 0:
                    speed_ratio = openai_metrics.avg_processing_time / metrics.avg_processing_time
                    if speed_ratio > 1:
                        f.write(f"({speed_ratio:.1f}x slower than OpenAI)\n")
                    else:
                        f.write(f"({1/speed_ratio:.1f}x faster than OpenAI)\n")
                else:
                    f.write("(processing time not recorded)\n")
                f.write(f"  Cost: $0.00 (vs ${openai_metrics.total_cost:.4f} for OpenAI)\n")
                f.write(f"  Embedding dimension: {metrics.embedding_dimension}\n")
                f.write(f"  Truncation required: {metrics.truncation_required}/{metrics.total_patents}")
                f.write(f" ({metrics.avg_truncation_percentage:.1f}%)\n")
                
                # Speed comparison
                if metrics.avg_processing_time > 0:
                    speed_ratio = openai_metrics.avg_processing_time / metrics.avg_processing_time
                    if speed_ratio > 1:
                        f.write(f"  ✅ {speed_ratio:.1f}x FASTER than OpenAI\n")
                    else:
                        f.write(f"  ❌ {1/speed_ratio:.1f}x slower than OpenAI\n")
                else:
                    f.write(f"  ⚠️ Processing time comparison not available\n")
                
                # Cost comparison
                cost_savings = openai_metrics.total_cost
                f.write(f"  💰 Cost savings: ${cost_savings:.4f} (100% reduction)\n")
                
                # Truncation comparison
                truncation_improvement = openai_metrics.avg_truncation_percentage - metrics.avg_truncation_percentage
                if truncation_improvement > 0:
                    f.write(f"  📄 {truncation_improvement:.1f}% LESS truncation than OpenAI\n")
            
            f.write(f"\nReport generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


async def main():
    """CLI interface for OpenAI baseline comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare OpenAI embeddings with existing models")
    parser.add_argument("patent_file", help="Input patent JSONL file")
    parser.add_argument("--existing-embeddings", 
                       help="Existing embeddings file for comparison")
    parser.add_argument("--output", default="openai_baseline_embeddings.jsonl", 
                       help="Output file for OpenAI embeddings")
    parser.add_argument("--max-records", type=int, default=100, 
                       help="Maximum number of patents to process")
    
    args = parser.parse_args()
    
    if not Path(args.patent_file).exists():
        logger.error(f"Patent file not found: {args.patent_file}")
        return
    
    try:
        comparator = BaselineComparison()
        result_file = await comparator.run_comparison(
            patent_file=args.patent_file,
            existing_embeddings_file=args.existing_embeddings or "",
            output_file=args.output,
            max_records=args.max_records
        )
        
        logger.info(f"Comparison completed successfully!")
        logger.info(f"Results available in: {result_file}")
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())