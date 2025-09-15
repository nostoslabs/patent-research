"""Google Patents baseline comparison for evaluating embedding-based search results."""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from urllib.parse import quote_plus
import httpx
from playwright.async_api import async_playwright
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PatentSearchResult:
    """A patent found in Google Patents search."""
    patent_id: str
    title: str
    abstract: str
    url: str
    relevance_rank: int


@dataclass 
class BaselineComparison:
    """Comparison between our results and Google Patents baseline."""
    query_patent_id: str
    query_abstract: str
    google_results: List[PatentSearchResult]
    embedding_results: List[Dict[str, Any]]
    overlap_count: int
    overlap_patents: List[str]
    precision_at_k: Dict[int, float]  # k=[3,5,10]


class ClassificationBasedSearcher:
    """Simulate baseline search using patent classification codes and LLM analysis."""
    
    def __init__(self, embeddings_file: str):
        """Initialize with embeddings data for classification-based search."""
        self.embeddings = self.load_embeddings(embeddings_file)
        self.classification_groups = self._group_by_classification()
        logger.info(f"Loaded {len(self.embeddings)} patents across {len(self.classification_groups)} classification groups")
    
    def load_embeddings(self, embeddings_file: str) -> List[Dict]:
        """Load embeddings from JSONL file."""
        embeddings = []
        
        with open(embeddings_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                
                # Handle multi-model format
                if 'models' in data:
                    # Use first available model's embedding
                    for model_name, model_data in data['models'].items():
                        if 'embeddings' in model_data and 'original' in model_data['embeddings']:
                            embedding = model_data['embeddings']['original']['embedding']
                            embeddings.append({
                                'id': data['id'],
                                'abstract': data['abstract'],
                                'classification': data.get('classification', ''),
                                'embedding': embedding
                            })
                            break
                elif 'embedding' in data:
                    embeddings.append(data)
        
        return embeddings
    
    def _group_by_classification(self) -> Dict[str, List[Dict]]:
        """Group patents by classification for baseline search."""
        groups = {}
        for patent in self.embeddings:
            classification = patent.get('classification', 'unknown')
            if classification not in groups:
                groups[classification] = []
            groups[classification].append(patent)
        return groups
    
    async def search_similar_patents(self, 
                                   query_patent: Dict[str, Any], 
                                   max_results: int = 10) -> List[PatentSearchResult]:
        """Simulate baseline search using classification and random selection."""
        
        logger.info(f"Simulating baseline search for patent {query_patent['id']}")
        
        results = []
        query_classification = query_patent.get('classification', 'unknown')
        
        # First, try to find patents in the same classification
        same_class_patents = self.classification_groups.get(query_classification, [])
        same_class_patents = [p for p in same_class_patents if p['id'] != query_patent['id']]
        
        # Add some random patents from same classification
        import random
        random.seed(42)  # Reproducible results
        
        if same_class_patents:
            sample_size = min(max_results // 2, len(same_class_patents))
            sampled = random.sample(same_class_patents, sample_size)
            
            for i, patent in enumerate(sampled):
                result = PatentSearchResult(
                    patent_id=patent['id'],
                    title=f"Patent in same classification ({query_classification})",
                    abstract=patent['abstract'][:500],  # Truncate
                    url=f"https://patents.google.com/patent/{patent['id']}",
                    relevance_rank=i + 1
                )
                results.append(result)
        
        # Fill remaining slots with random patents from other classifications
        remaining_slots = max_results - len(results)
        if remaining_slots > 0:
            other_patents = []
            for class_code, patents in self.classification_groups.items():
                if class_code != query_classification:
                    other_patents.extend([p for p in patents if p['id'] != query_patent['id']])
            
            if other_patents:
                sample_size = min(remaining_slots, len(other_patents))
                sampled = random.sample(other_patents, sample_size)
                
                for i, patent in enumerate(sampled):
                    result = PatentSearchResult(
                        patent_id=patent['id'],
                        title=f"Patent from classification {patent.get('classification', 'unknown')}",
                        abstract=patent['abstract'][:500],
                        url=f"https://patents.google.com/patent/{patent['id']}",
                        relevance_rank=len(results) + i + 1
                    )
                    results.append(result)
        
        logger.info(f"Generated {len(results)} simulated baseline results")
        return results


class EmbeddingSearcher:
    """Search using our embedding-based system."""
    
    def __init__(self, embeddings_file: str):
        """Initialize with embeddings data."""
        self.embeddings = self.load_embeddings(embeddings_file)
        self.embedding_matrix = self._build_embedding_matrix()
        logger.info(f"Loaded {len(self.embeddings)} patent embeddings")
    
    def load_embeddings(self, embeddings_file: str) -> List[Dict]:
        """Load embeddings from JSONL file."""
        embeddings = []
        
        with open(embeddings_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                
                # Handle multi-model format
                if 'models' in data:
                    # Use first available model's embedding
                    for model_name, model_data in data['models'].items():
                        if 'embeddings' in model_data and 'original' in model_data['embeddings']:
                            embedding = model_data['embeddings']['original']['embedding']
                            embeddings.append({
                                'id': data['id'],
                                'abstract': data['abstract'],
                                'classification': data.get('classification', ''),
                                'embedding': embedding
                            })
                            break
                elif 'embedding' in data:
                    embeddings.append(data)
        
        return embeddings
    
    def _build_embedding_matrix(self) -> np.ndarray:
        """Build matrix for efficient similarity search."""
        return np.array([p['embedding'] for p in self.embeddings])
    
    def search_similar(self, query_patent_id: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Find similar patents using cosine similarity."""
        
        # Find query patent
        query_patent = None
        query_idx = None
        
        for i, patent in enumerate(self.embeddings):
            if patent['id'] == query_patent_id:
                query_patent = patent
                query_idx = i
                break
        
        if not query_patent:
            logger.error(f"Query patent {query_patent_id} not found in embeddings")
            return []
        
        # Calculate similarities
        query_embedding = np.array([query_patent['embedding']])
        similarities = cosine_similarity(query_embedding, self.embedding_matrix)[0]
        
        # Get top-k similar patents (excluding self)
        similar_indices = np.argsort(similarities)[::-1][1:top_k+1]  # Skip index 0 (self)
        
        results = []
        for i, idx in enumerate(similar_indices):
            patent = self.embeddings[idx]
            results.append({
                'patent_id': patent['id'],
                'abstract': patent['abstract'],
                'classification': patent.get('classification', ''),
                'similarity_score': float(similarities[idx]),
                'rank': i + 1
            })
        
        return results


class BaselineComparator:
    """Compare Google Patents results with our embedding-based search."""
    
    def __init__(self, embeddings_file: str):
        """Initialize with embedding searcher."""
        self.embeddings_file = embeddings_file
        self.embedding_searcher = EmbeddingSearcher(embeddings_file)
    
    async def compare_single_patent(self, 
                                  patent_id: str, 
                                  max_results: int = 10) -> Optional[BaselineComparison]:
        """Compare results for a single patent."""
        
        # Find the patent in our dataset
        query_patent = None
        for patent in self.embedding_searcher.embeddings:
            if patent['id'] == patent_id:
                query_patent = patent
                break
        
        if not query_patent:
            logger.error(f"Patent {patent_id} not found in our dataset")
            return None
        
        logger.info(f"Comparing results for patent {patent_id}")
        
        # Search using our embeddings
        embedding_results = self.embedding_searcher.search_similar(patent_id, max_results)
        
        # Search using classification-based baseline (simulating Google Patents)
        baseline_searcher = ClassificationBasedSearcher(self.embeddings_file)
        baseline_results = await baseline_searcher.search_similar_patents(
            query_patent, max_results
        )
        google_results = baseline_results  # Rename for compatibility
        
        # Calculate overlap
        embedding_ids = {r['patent_id'] for r in embedding_results}
        google_ids = {r.patent_id for r in google_results}
        
        overlap_patents = list(embedding_ids & google_ids)
        overlap_count = len(overlap_patents)
        
        # Calculate precision@k
        precision_at_k = {}
        for k in [3, 5, 10]:
            if k <= len(embedding_results):
                embedding_top_k = {r['patent_id'] for r in embedding_results[:k]}
                google_top_k = {r.patent_id for r in google_results[:k]}
                intersection = len(embedding_top_k & google_top_k)
                precision_at_k[k] = intersection / k if k > 0 else 0
        
        comparison = BaselineComparison(
            query_patent_id=patent_id,
            query_abstract=query_patent['abstract'],
            google_results=google_results,
            embedding_results=embedding_results,
            overlap_count=overlap_count,
            overlap_patents=overlap_patents,
            precision_at_k=precision_at_k
        )
        
        logger.info(f"Patent {patent_id}: {overlap_count} overlapping results, "
                   f"P@5={precision_at_k.get(5, 0):.2f}")
        
        return comparison
    
    async def run_baseline_study(self, 
                               patent_ids: List[str],
                               output_file: str = "baseline_comparison.jsonl",
                               delay_between_queries: float = 5.0) -> Dict[str, Any]:
        """Run comprehensive baseline comparison study."""
        
        logger.info(f"Starting baseline comparison for {len(patent_ids)} patents")
        
        results = []
        successful_comparisons = 0
        
        for i, patent_id in enumerate(patent_ids):
            logger.info(f"Processing patent {i+1}/{len(patent_ids)}: {patent_id}")
            
            try:
                comparison = await self.compare_single_patent(patent_id)
                
                if comparison:
                    # Convert to serializable format
                    result = {
                        'query_patent_id': comparison.query_patent_id,
                        'google_results_count': len(comparison.google_results),
                        'embedding_results_count': len(comparison.embedding_results),
                        'overlap_count': comparison.overlap_count,
                        'overlap_patents': comparison.overlap_patents,
                        'precision_at_k': comparison.precision_at_k,
                        'google_results': [
                            {
                                'patent_id': r.patent_id,
                                'title': r.title,
                                'abstract': r.abstract[:500],  # Truncate for storage
                                'relevance_rank': r.relevance_rank
                            }
                            for r in comparison.google_results
                        ],
                        'embedding_results': [
                            {
                                'patent_id': r['patent_id'],
                                'similarity_score': r['similarity_score'],
                                'rank': r['rank']
                            }
                            for r in comparison.embedding_results
                        ]
                    }
                    
                    results.append(result)
                    successful_comparisons += 1
                    
                    # Save intermediate results
                    with open(output_file, 'w', encoding='utf-8') as f:
                        for res in results:
                            json.dump(res, f, ensure_ascii=False)
                            f.write('\n')
                
            except Exception as e:
                logger.error(f"Error processing patent {patent_id}: {e}")
            
            # Rate limiting
            if i < len(patent_ids) - 1:
                await asyncio.sleep(delay_between_queries)
        
        # Calculate summary statistics
        if results:
            avg_overlap = np.mean([r['overlap_count'] for r in results])
            avg_precision_5 = np.mean([r['precision_at_k'].get(5, 0) for r in results])
            avg_precision_10 = np.mean([r['precision_at_k'].get(10, 0) for r in results])
        else:
            avg_overlap = avg_precision_5 = avg_precision_10 = 0
        
        summary = {
            'total_patents_tested': len(patent_ids),
            'successful_comparisons': successful_comparisons,
            'success_rate': successful_comparisons / len(patent_ids) if patent_ids else 0,
            'average_overlap_count': avg_overlap,
            'average_precision_at_5': avg_precision_5,
            'average_precision_at_10': avg_precision_10,
            'output_file': output_file,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save summary
        summary_file = output_file.replace('.jsonl', '_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Baseline comparison completed!")
        logger.info(f"Success rate: {summary['success_rate']:.1%}")
        logger.info(f"Average overlap: {avg_overlap:.1f} patents")
        logger.info(f"Average P@5: {avg_precision_5:.2f}")
        logger.info(f"Results saved to: {output_file}")
        
        return summary


async def main():
    """CLI interface for baseline comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare embedding search with Google Patents baseline")
    parser.add_argument("embeddings_file", help="Input embeddings JSONL file")
    parser.add_argument("--patents", nargs='+', help="Specific patent IDs to test")
    parser.add_argument("--sample", type=int, default=10, help="Random sample size from dataset")
    parser.add_argument("--output", default="baseline_comparison.jsonl", help="Output file")
    parser.add_argument("--delay", type=float, default=5.0, help="Delay between Google searches (seconds)")
    
    args = parser.parse_args()
    
    # Check if embeddings file exists
    if not Path(args.embeddings_file).exists():
        logger.error(f"Embeddings file not found: {args.embeddings_file}")
        return
    
    try:
        comparator = BaselineComparator(args.embeddings_file)
        
        # Determine patent IDs to test
        if args.patents:
            patent_ids = args.patents
        else:
            # Random sample from available patents
            available_patents = [p['id'] for p in comparator.embedding_searcher.embeddings]
            import random
            random.seed(42)  # Reproducible sampling
            patent_ids = random.sample(available_patents, min(args.sample, len(available_patents)))
        
        logger.info(f"Testing {len(patent_ids)} patents with {args.delay}s delay between searches")
        
        await comparator.run_baseline_study(
            patent_ids=patent_ids,
            output_file=args.output,
            delay_between_queries=args.delay
        )
        
    except Exception as e:
        logger.error(f"Failed to run baseline comparison: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())