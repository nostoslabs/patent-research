"""Ground truth dataset generator using LLMs for patent similarity evaluation."""

import json
import numpy as np
import asyncio
import logging
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity

from llm_provider_factory import LLMFactory, LLMProvider, PatentSimilarityAnalysis

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PatentPair:
    """A pair of patents with their similarity information."""
    patent1: Dict[str, Any]
    patent2: Dict[str, Any]
    embedding_similarity: float
    category: str  # 'high', 'medium', 'low'


class GroundTruthGenerator:
    """Generate ground truth dataset for patent similarity evaluation."""
    
    def __init__(self, embeddings_file: str, provider: Optional[LLMProvider] = None):
        """Initialize with embeddings data and LLM provider."""
        self.embeddings = self.load_embeddings(embeddings_file)
        self.agent = LLMFactory.create_agent(provider)
        self.provider = provider
        
        # Pre-compute similarity matrix for efficient pair selection
        self.similarity_matrix = self._compute_similarity_matrix()
        
        logger.info(f"Initialized with {len(self.embeddings)} patents")
        logger.info(f"Using LLM provider: {provider}")
    
    def load_embeddings(self, embeddings_file: str) -> List[Dict]:
        """Load embeddings from JSONL file."""
        embeddings = []
        
        with open(embeddings_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                
                # Extract embeddings based on our multi-model format
                if 'models' in data:
                    # Multi-model format - use first available model's embedding
                    for model_name, model_data in data['models'].items():
                        if 'embeddings' in model_data and 'original' in model_data['embeddings']:
                            embedding = model_data['embeddings']['original']['embedding']
                            embeddings.append({
                                'id': data['id'],
                                'abstract': data['abstract'],
                                'classification': data.get('classification', ''),
                                'embedding': embedding,
                                'abstract_length': data.get('abstract_length', len(data['abstract']))
                            })
                            break
                elif 'embedding' in data:
                    # Simple format
                    embeddings.append(data)
        
        logger.info(f"Loaded {len(embeddings)} patent embeddings")
        return embeddings
    
    def _compute_similarity_matrix(self) -> np.ndarray:
        """Pre-compute cosine similarity matrix."""
        logger.info("Computing similarity matrix...")
        
        embedding_vectors = np.array([p['embedding'] for p in self.embeddings])
        similarity_matrix = cosine_similarity(embedding_vectors)
        
        logger.info("Similarity matrix computed")
        return similarity_matrix
    
    def generate_diverse_pairs(self, n_pairs: int = 500) -> List[PatentPair]:
        """Generate diverse patent pairs across similarity ranges."""
        logger.info(f"Generating {n_pairs} diverse patent pairs...")
        
        pairs = []
        n_patents = len(self.embeddings)
        
        # Get upper triangle indices (excluding diagonal)
        triu_indices = np.triu_indices(n_patents, k=1)
        similarities = self.similarity_matrix[triu_indices]
        
        # Define similarity thresholds
        high_threshold = np.percentile(similarities, 90)  # Top 10%
        medium_low = np.percentile(similarities, 35)     # 35th percentile
        medium_high = np.percentile(similarities, 65)    # 65th percentile
        low_threshold = np.percentile(similarities, 10)   # Bottom 10%
        
        logger.info(f"Similarity thresholds: high>{high_threshold:.3f}, "
                   f"medium={medium_low:.3f}-{medium_high:.3f}, low<{low_threshold:.3f}")
        
        # Target distribution: 1/3 each of high, medium, low similarity
        target_high = n_pairs // 3
        target_medium = n_pairs // 3
        target_low = n_pairs - target_high - target_medium  # Remaining
        
        counts = {'high': 0, 'medium': 0, 'low': 0}
        
        # Create pairs based on similarity categories
        for idx in range(len(triu_indices[0])):
            if len(pairs) >= n_pairs:
                break
                
            i, j = triu_indices[0][idx], triu_indices[1][idx]
            sim = similarities[idx]
            
            category = None
            
            # Categorize similarity
            if sim >= high_threshold and counts['high'] < target_high:
                category = 'high'
            elif medium_low <= sim <= medium_high and counts['medium'] < target_medium:
                category = 'medium'
            elif sim <= low_threshold and counts['low'] < target_low:
                category = 'low'
            
            if category:
                pair = PatentPair(
                    patent1=self.embeddings[i],
                    patent2=self.embeddings[j],
                    embedding_similarity=float(sim),
                    category=category
                )
                pairs.append(pair)
                counts[category] += 1
        
        logger.info(f"Generated pairs: {counts}")
        return pairs
    
    async def evaluate_pair_with_llm(self, pair: PatentPair) -> Dict[str, Any]:
        """Evaluate a single patent pair using LLM."""
        patent1, patent2 = pair.patent1, pair.patent2
        
        # Truncate abstracts to fit context window
        abstract1 = patent1['abstract'][:2000]
        abstract2 = patent2['abstract'][:2000]
        
        prompt = f"""Compare these two patent abstracts and provide a detailed similarity assessment:

**Patent 1 (ID: {patent1['id']})**
{abstract1}

**Patent 2 (ID: {patent2['id']})**
{abstract2}

Analyze their technical similarity across all dimensions and provide specific scores and reasoning."""
        
        try:
            result = await self.agent.run(prompt)
            
            return {
                'patent1_id': patent1['id'],
                'patent2_id': patent2['id'],
                'embedding_similarity': pair.embedding_similarity,
                'similarity_category': pair.category,
                'classification1': patent1.get('classification', ''),
                'classification2': patent2.get('classification', ''),
                'abstract_length1': patent1.get('abstract_length', len(patent1['abstract'])),
                'abstract_length2': patent2.get('abstract_length', len(patent2['abstract'])),
                'llm_analysis': result.output.model_dump(),
                'success': True,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Error evaluating pair {patent1['id']}-{patent2['id']}: {e}")
            return {
                'patent1_id': patent1['id'],
                'patent2_id': patent2['id'],
                'embedding_similarity': pair.embedding_similarity,
                'similarity_category': pair.category,
                'success': False,
                'error': str(e)
            }
    
    async def process_pairs_batch(self, pairs: List[PatentPair], 
                                 max_concurrent: int = 5,
                                 save_interval: int = 10) -> List[Dict]:
        """Process pairs in batches with rate limiting and incremental saving."""
        logger.info(f"Processing {len(pairs)} pairs with max {max_concurrent} concurrent requests")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        results = []
        
        async def process_single_pair(pair: PatentPair) -> Dict:
            async with semaphore:
                # Add small delay to respect rate limits
                await asyncio.sleep(0.1)
                return await self.evaluate_pair_with_llm(pair)
        
        # Process all pairs
        for i in range(0, len(pairs), save_interval):
            batch = pairs[i:i + save_interval]
            logger.info(f"Processing batch {i//save_interval + 1}/{len(pairs)//save_interval + 1}")
            
            # Process batch
            batch_results = await asyncio.gather(
                *[process_single_pair(pair) for pair in batch],
                return_exceptions=True
            )
            
            # Handle exceptions
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Exception in batch processing: {result}")
                    # Create error record
                    pair = batch[j]
                    result = {
                        'patent1_id': pair.patent1['id'],
                        'patent2_id': pair.patent2['id'],
                        'success': False,
                        'error': str(result)
                    }
                
                results.append(result)
            
            # Save intermediate results
            self._save_intermediate_results(results, f"ground_truth_partial_{len(results)}.jsonl")
            
            logger.info(f"Completed {len(results)}/{len(pairs)} pairs")
        
        return results
    
    def _save_intermediate_results(self, results: List[Dict], filename: str):
        """Save intermediate results to prevent data loss."""
        with open(filename, 'w', encoding='utf-8') as f:
            for result in results:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')
    
    async def generate_ground_truth_dataset(self, 
                                          n_pairs: int = 500,
                                          output_file: str = "patent_ground_truth.jsonl",
                                          max_concurrent: int = 5) -> str:
        """Generate complete ground truth dataset."""
        logger.info(f"Starting ground truth generation for {n_pairs} pairs")
        
        # Estimate costs
        if self.provider:
            cost_estimate = LLMFactory.estimate_cost(self.provider, n_pairs)
            logger.info(f"Estimated cost: ${cost_estimate['estimated_cost_usd']}")
        
        start_time = time.time()
        
        # Generate diverse pairs
        pairs = self.generate_diverse_pairs(n_pairs)
        logger.info(f"Selected {len(pairs)} diverse pairs")
        
        if not pairs:
            raise ValueError("No pairs generated. Check your embeddings data.")
        
        # Process pairs with LLM
        results = await self.process_pairs_batch(pairs, max_concurrent)
        
        # Filter successful results
        successful_results = [r for r in results if r.get('success', False)]
        failed_count = len(results) - len(successful_results)
        
        # Save final results
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in successful_results:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')
        
        elapsed_time = time.time() - start_time
        
        # Generate summary
        summary = {
            'total_pairs_requested': n_pairs,
            'pairs_generated': len(pairs),
            'pairs_processed': len(results),
            'successful_evaluations': len(successful_results),
            'failed_evaluations': failed_count,
            'success_rate': len(successful_results) / len(results) if results else 0,
            'processing_time_minutes': elapsed_time / 60,
            'output_file': output_file,
            'provider_used': self.provider.value if self.provider else 'auto',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save summary
        summary_file = output_file.replace('.jsonl', '_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Ground truth generation completed!")
        logger.info(f"Successful evaluations: {len(successful_results)}/{len(results)}")
        logger.info(f"Processing time: {elapsed_time/60:.1f} minutes")
        logger.info(f"Results saved to: {output_file}")
        logger.info(f"Summary saved to: {summary_file}")
        
        return output_file


async def main():
    """CLI interface for ground truth generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate ground truth dataset for patent similarity")
    parser.add_argument("embeddings_file", help="Input embeddings JSONL file")
    parser.add_argument("--output", default="patent_ground_truth.jsonl", help="Output file")
    parser.add_argument("--pairs", type=int, default=100, help="Number of pairs to generate")
    parser.add_argument("--concurrent", type=int, default=3, help="Max concurrent requests")
    parser.add_argument("--provider", choices=['openai', 'google', 'anthropic', 'ollama'], 
                       help="LLM provider (auto-detect if not specified)")
    
    args = parser.parse_args()
    
    # Convert string to enum if provided
    provider = None
    if args.provider:
        provider = LLMProvider(args.provider)
    
    # Check if embeddings file exists
    if not Path(args.embeddings_file).exists():
        logger.error(f"Embeddings file not found: {args.embeddings_file}")
        return
    
    try:
        generator = GroundTruthGenerator(args.embeddings_file, provider)
        await generator.generate_ground_truth_dataset(
            n_pairs=args.pairs,
            output_file=args.output,
            max_concurrent=args.concurrent
        )
    except Exception as e:
        logger.error(f"Failed to generate ground truth: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())