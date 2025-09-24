"""Cross-encoder reranking system for improving patent similarity search results."""

import asyncio
import json
import logging
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity

from llm_provider_factory import LLMFactory, LLMProvider, PatentSimilarityAnalysis

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass 
class SearchResult:
    """A search result with embedding-based similarity and optional reranked score."""
    patent_id: str
    abstract: str
    embedding_similarity: float
    initial_rank: int
    reranked_score: Optional[float] = None
    reranked_rank: Optional[int] = None
    llm_analysis: Optional[Dict] = None


@dataclass
class RerankedResults:
    """Results after cross-encoder reranking."""
    query_patent_id: str
    query_abstract: str
    initial_results: List[SearchResult]
    reranked_results: List[SearchResult]
    reranking_time: float
    improvement_metrics: Dict[str, float]


class EmbeddingSearchEngine:
    """Basic embedding-based search engine for initial retrieval."""
    
    def __init__(self, embeddings_file: str):
        """Initialize with embeddings data."""
        self.embeddings_file = embeddings_file
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
    
    def search_similar(self, query_patent_id: str, top_k: int = 50) -> List[SearchResult]:
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
        similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
        
        results = []
        for i, idx in enumerate(similar_indices):
            patent = self.embeddings[idx]
            result = SearchResult(
                patent_id=patent['id'],
                abstract=patent['abstract'],
                embedding_similarity=float(similarities[idx]),
                initial_rank=i + 1
            )
            results.append(result)
        
        return results


class CrossEncoderReranker:
    """Rerank search results using cross-encoder (LLM-based) scoring."""
    
    def __init__(self, provider: Optional[LLMProvider] = None):
        """Initialize with LLM provider for reranking."""
        self.agent = LLMFactory.create_agent(provider)
        self.provider = provider
        logger.info(f"Initialized cross-encoder reranker with provider: {provider}")
    
    async def rerank_results(self, 
                           query_patent: Dict[str, Any],
                           initial_results: List[SearchResult],
                           top_k_rerank: int = 20) -> List[SearchResult]:
        """Rerank the top results using cross-encoder scoring."""
        
        if len(initial_results) == 0:
            return initial_results
        
        # Take only top K results for reranking (computational efficiency)
        candidates = initial_results[:top_k_rerank]
        logger.info(f"Reranking top {len(candidates)} results for patent {query_patent['id']}")
        
        # Score each candidate against the query
        scored_results = []
        for result in candidates:
            try:
                score = await self._score_pair(query_patent, result)
                result.reranked_score = score
                result.llm_analysis = score  # Store full analysis
                scored_results.append(result)
                
            except Exception as e:
                logger.warning(f"Failed to score pair {query_patent['id']}-{result.patent_id}: {e}")
                # Keep original result with no reranked score
                scored_results.append(result)
        
        # Sort by reranked score (higher is better)
        valid_scores = [r for r in scored_results if r.reranked_score is not None]
        no_scores = [r for r in scored_results if r.reranked_score is None]
        
        if valid_scores:
            # Sort by LLM similarity score
            valid_scores.sort(key=lambda x: x.llm_analysis.get('similarity_score', 0), reverse=True)
            
            # Update reranked ranks
            for i, result in enumerate(valid_scores):
                result.reranked_rank = i + 1
            
            # Append unscored results at the end
            for i, result in enumerate(no_scores):
                result.reranked_rank = len(valid_scores) + i + 1
            
            reranked = valid_scores + no_scores
        else:
            # No successful scores, return original order
            reranked = scored_results
            for i, result in enumerate(reranked):
                result.reranked_rank = result.initial_rank
        
        logger.info(f"Successfully reranked {len(valid_scores)}/{len(candidates)} results")
        return reranked
    
    async def _score_pair(self, query_patent: Dict, candidate: SearchResult) -> Dict[str, Any]:
        """Score a query-candidate pair using the LLM."""
        
        # Truncate abstracts to fit context window
        query_abstract = query_patent['abstract'][:2000]
        candidate_abstract = candidate.abstract[:2000]
        
        prompt = f"""Compare these two patent abstracts and provide a detailed similarity assessment:

**Query Patent (ID: {query_patent['id']})**
{query_abstract}

**Candidate Patent (ID: {candidate.patent_id})**
{candidate_abstract}

Analyze their technical similarity across all dimensions and provide specific scores and reasoning."""
        
        result = await self.agent.run(prompt)
        return result.output.model_dump()


class TwoStageSearchSystem:
    """Complete two-stage search system: embedding retrieval + cross-encoder reranking."""
    
    def __init__(self, embeddings_file: str, provider: Optional[LLMProvider] = None):
        """Initialize both search stages."""
        self.embeddings_file = embeddings_file
        self.embedding_searcher = EmbeddingSearchEngine(embeddings_file)
        self.reranker = CrossEncoderReranker(provider)
        self.provider = provider
    
    async def search(self, 
                    query_patent_id: str,
                    top_k_initial: int = 50,
                    top_k_rerank: int = 20,
                    top_k_final: int = 10) -> RerankedResults:
        """Perform complete two-stage search."""
        
        logger.info(f"Starting two-stage search for patent {query_patent_id}")
        start_time = time.time()
        
        # Find query patent
        query_patent = None
        for patent in self.embedding_searcher.embeddings:
            if patent['id'] == query_patent_id:
                query_patent = patent
                break
        
        if not query_patent:
            raise ValueError(f"Query patent {query_patent_id} not found")
        
        # Stage 1: Embedding-based retrieval
        logger.info(f"Stage 1: Retrieving top {top_k_initial} candidates")
        initial_results = self.embedding_searcher.search_similar(query_patent_id, top_k_initial)
        
        if not initial_results:
            logger.warning("No initial results found")
            return RerankedResults(
                query_patent_id=query_patent_id,
                query_abstract=query_patent['abstract'],
                initial_results=[],
                reranked_results=[],
                reranking_time=0,
                improvement_metrics={}
            )
        
        # Stage 2: Cross-encoder reranking
        logger.info(f"Stage 2: Reranking top {top_k_rerank} candidates")
        reranked_results = await self.reranker.rerank_results(
            query_patent, initial_results, top_k_rerank
        )
        
        reranking_time = time.time() - start_time
        
        # Calculate improvement metrics
        improvement_metrics = self._calculate_improvement_metrics(
            initial_results[:top_k_final], 
            reranked_results[:top_k_final]
        )
        
        results = RerankedResults(
            query_patent_id=query_patent_id,
            query_abstract=query_patent['abstract'],
            initial_results=initial_results,
            reranked_results=reranked_results,
            reranking_time=reranking_time,
            improvement_metrics=improvement_metrics
        )
        
        logger.info(f"Two-stage search completed in {reranking_time:.1f}s")
        return results
    
    def _calculate_improvement_metrics(self, 
                                     initial: List[SearchResult], 
                                     reranked: List[SearchResult]) -> Dict[str, float]:
        """Calculate metrics comparing initial vs reranked results."""
        
        if not initial or not reranked:
            return {}
        
        # Calculate rank changes
        rank_changes = []
        initial_ids = {r.patent_id: r.initial_rank for r in initial}
        
        for result in reranked:
            if result.patent_id in initial_ids and result.reranked_rank:
                initial_rank = initial_ids[result.patent_id]
                rank_change = initial_rank - result.reranked_rank  # Positive = moved up
                rank_changes.append(rank_change)
        
        # Calculate average LLM scores
        llm_scores = []
        for result in reranked:
            if result.llm_analysis and isinstance(result.llm_analysis, dict):
                score = result.llm_analysis.get('similarity_score', 0)
                llm_scores.append(score)
        
        metrics = {
            'average_rank_change': np.mean(rank_changes) if rank_changes else 0,
            'rank_improvements': sum(1 for c in rank_changes if c > 0),
            'rank_degradations': sum(1 for c in rank_changes if c < 0),
            'average_llm_score': np.mean(llm_scores) if llm_scores else 0,
            'total_reranked': len([r for r in reranked if r.reranked_score is not None])
        }
        
        return metrics
    
    async def batch_evaluate(self, 
                           patent_ids: List[str],
                           output_file: str = "reranking_evaluation.jsonl",
                           **search_kwargs) -> Dict[str, Any]:
        """Evaluate reranking on multiple patents."""
        
        logger.info(f"Starting batch evaluation for {len(patent_ids)} patents")
        results = []
        
        for i, patent_id in enumerate(patent_ids):
            logger.info(f"Processing patent {i+1}/{len(patent_ids)}: {patent_id}")
            
            try:
                result = await self.search(patent_id, **search_kwargs)
                
                # Convert to serializable format
                serializable_result = {
                    'query_patent_id': result.query_patent_id,
                    'reranking_time': result.reranking_time,
                    'improvement_metrics': result.improvement_metrics,
                    'initial_top10': [
                        {
                            'patent_id': r.patent_id,
                            'embedding_similarity': r.embedding_similarity,
                            'initial_rank': r.initial_rank
                        }
                        for r in result.initial_results[:10]
                    ],
                    'reranked_top10': [
                        {
                            'patent_id': r.patent_id,
                            'embedding_similarity': r.embedding_similarity,
                            'initial_rank': r.initial_rank,
                            'reranked_rank': r.reranked_rank,
                            'llm_score': r.llm_analysis.get('similarity_score', 0) if r.llm_analysis else None,
                            'llm_confidence': r.llm_analysis.get('confidence', 0) if r.llm_analysis else None
                        }
                        for r in result.reranked_results[:10]
                    ]
                }
                
                results.append(serializable_result)
                
                # Save intermediate results
                with open(output_file, 'w', encoding='utf-8') as f:
                    for res in results:
                        json.dump(res, f, ensure_ascii=False)
                        f.write('\n')
                
            except Exception as e:
                logger.error(f"Error processing patent {patent_id}: {e}")
        
        # Calculate summary statistics
        if results:
            avg_time = np.mean([r['reranking_time'] for r in results])
            avg_rank_change = np.mean([
                r['improvement_metrics'].get('average_rank_change', 0) 
                for r in results
            ])
            avg_llm_score = np.mean([
                r['improvement_metrics'].get('average_llm_score', 0) 
                for r in results
            ])
            total_improvements = sum([
                r['improvement_metrics'].get('rank_improvements', 0) 
                for r in results
            ])
        else:
            avg_time = avg_rank_change = avg_llm_score = total_improvements = 0
        
        summary = {
            'total_patents_evaluated': len(patent_ids),
            'successful_evaluations': len(results),
            'success_rate': len(results) / len(patent_ids) if patent_ids else 0,
            'average_reranking_time': avg_time,
            'average_rank_change': avg_rank_change,
            'average_llm_score': avg_llm_score,
            'total_rank_improvements': total_improvements,
            'output_file': output_file,
            'provider_used': self.provider.value if self.provider else 'auto',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save summary
        summary_file = output_file.replace('.jsonl', '_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Batch evaluation completed!")
        logger.info(f"Success rate: {summary['success_rate']:.1%}")
        logger.info(f"Average reranking time: {avg_time:.1f}s")
        logger.info(f"Average rank change: {avg_rank_change:+.1f}")
        logger.info(f"Results saved to: {output_file}")
        
        return summary


async def main():
    """CLI interface for cross-encoder reranking."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cross-encoder reranking for patent search")
    parser.add_argument("embeddings_file", help="Input embeddings JSONL file")
    parser.add_argument("--query", help="Specific patent ID to search for")
    parser.add_argument("--batch", nargs='+', help="Multiple patent IDs to evaluate")
    parser.add_argument("--sample", type=int, default=5, help="Random sample size for evaluation")
    parser.add_argument("--output", default="reranking_evaluation.jsonl", help="Output file")
    parser.add_argument("--provider", choices=['openai', 'google', 'anthropic', 'ollama'], 
                       help="LLM provider for reranking")
    parser.add_argument("--top-k-initial", type=int, default=50, help="Initial retrieval candidates")
    parser.add_argument("--top-k-rerank", type=int, default=20, help="Number to rerank")
    parser.add_argument("--top-k-final", type=int, default=10, help="Final results to return")
    
    args = parser.parse_args()
    
    # Check if embeddings file exists
    if not Path(args.embeddings_file).exists():
        logger.error(f"Embeddings file not found: {args.embeddings_file}")
        return
    
    # Convert string to enum if provided
    provider = None
    if args.provider:
        provider = LLMProvider(args.provider)
    
    try:
        system = TwoStageSearchSystem(args.embeddings_file, provider)
        
        if args.query:
            # Single query
            logger.info(f"Performing two-stage search for patent {args.query}")
            result = await system.search(
                args.query,
                top_k_initial=args.top_k_initial,
                top_k_rerank=args.top_k_rerank,
                top_k_final=args.top_k_final
            )
            
            print(f"Query: {args.query}")
            print(f"Reranking time: {result.reranking_time:.1f}s")
            print(f"Improvement metrics: {result.improvement_metrics}")
            print("\nTop 10 reranked results:")
            for r in result.reranked_results[:10]:
                llm_score = r.llm_analysis.get('similarity_score', 0) if r.llm_analysis else 0
                print(f"  {r.reranked_rank:2d}. {r.patent_id} "
                      f"(was #{r.initial_rank}, LLM: {llm_score:.2f})")
        
        else:
            # Batch evaluation
            if args.batch:
                patent_ids = args.batch
            else:
                # Random sample
                available_patents = [p['id'] for p in system.embedding_searcher.embeddings]
                import random
                random.seed(42)
                patent_ids = random.sample(available_patents, min(args.sample, len(available_patents)))
            
            logger.info(f"Evaluating {len(patent_ids)} patents")
            
            await system.batch_evaluate(
                patent_ids=patent_ids,
                output_file=args.output,
                top_k_initial=args.top_k_initial,
                top_k_rerank=args.top_k_rerank,
                top_k_final=args.top_k_final
            )
        
    except Exception as e:
        logger.error(f"Failed to run cross-encoder reranking: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())