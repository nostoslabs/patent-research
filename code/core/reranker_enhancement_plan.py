"""Enhanced reranking system incorporating dedicated reranker models."""

import asyncio
import logging
import os
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import time

from cross_encoder_reranker import SearchResult, EmbeddingSearchEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RerankerModel:
    """Configuration for different reranker models."""
    name: str
    model_path: str
    framework: str  # 'sentence-transformers', 'cohere', 'bge'
    batch_size: int = 32
    max_length: int = 512


class AdvancedRerankerSystem:
    """Enhanced reranking using multiple reranker model options."""
    
    RERANKER_CONFIGS = {
        'bge-reranker-large': RerankerModel(
            name='bge-reranker-large',
            model_path='BAAI/bge-reranker-large',
            framework='sentence-transformers',
            batch_size=16,  # Larger model, smaller batches
            max_length=512
        ),
        'bge-reranker-base': RerankerModel(
            name='bge-reranker-base', 
            model_path='BAAI/bge-reranker-base',
            framework='sentence-transformers',
            batch_size=32,
            max_length=512
        ),
        'cohere-rerank': RerankerModel(
            name='cohere-rerank',
            model_path='cohere',  # API-based
            framework='cohere',
            batch_size=1000,  # API can handle larger batches
            max_length=4096   # Cohere supports longer contexts
        ),
        'llm-cross-encoder': RerankerModel(
            name='llm-cross-encoder',
            model_path='gemini-1.5-flash',  # Our current approach
            framework='llm',
            batch_size=10,   # LLM concurrent requests
            max_length=2048
        )
    }
    
    def __init__(self, 
                 embeddings_file: str,
                 reranker_type: str = 'bge-reranker-base',
                 fallback_to_llm: bool = True):
        """Initialize with embedding searcher and reranker model."""
        self.embeddings_file = embeddings_file
        self.embedding_searcher = EmbeddingSearchEngine(embeddings_file)
        self.reranker_type = reranker_type
        self.fallback_to_llm = fallback_to_llm
        
        # Load reranker model
        self.reranker_config = self.RERANKER_CONFIGS[reranker_type]
        self.reranker_model = self._load_reranker_model()
        
        logger.info(f"Initialized advanced reranker with {reranker_type}")
    
    def _load_reranker_model(self):
        """Load the specified reranker model."""
        config = self.reranker_config
        
        if config.framework == 'sentence-transformers':
            try:
                from sentence_transformers import CrossEncoder
                model = CrossEncoder(config.model_path)
                logger.info(f"Loaded sentence-transformers reranker: {config.name}")
                return model
            except ImportError:
                logger.error("sentence-transformers not installed. Install with: uv add sentence-transformers")
                if self.fallback_to_llm:
                    logger.info("Falling back to LLM-based reranking")
                    return None
                raise
        
        elif config.framework == 'cohere':
            try:
                import cohere
                api_key = os.getenv('COHERE_API_KEY')
                if not api_key:
                    logger.error("COHERE_API_KEY not found in environment")
                    if self.fallback_to_llm:
                        return None
                    raise ValueError("Cohere API key required")
                
                client = cohere.Client(api_key)
                logger.info("Initialized Cohere reranker")
                return client
            except ImportError:
                logger.error("cohere package not installed. Install with: uv add cohere")
                if self.fallback_to_llm:
                    return None
                raise
        
        elif config.framework == 'llm':
            # Use our existing PydanticAI-based LLM approach
            from llm_provider_factory import LLMFactory
            agent = LLMFactory.create_agent()
            logger.info("Using PydanticAI-based cross-encoder reranking")
            return agent
        
        else:
            raise ValueError(f"Unknown reranker framework: {config.framework}")
    
    async def rerank_results(self, 
                           query_patent: Dict[str, Any],
                           initial_results: List[SearchResult],
                           top_k: int = 10) -> List[SearchResult]:
        """Rerank results using the configured reranker model."""
        
        if not initial_results:
            return []
        
        logger.info(f"Reranking {len(initial_results)} results using {self.reranker_type}")
        
        if self.reranker_model is None:
            logger.warning("No reranker model available, returning original results")
            return initial_results[:top_k]
        
        config = self.reranker_config
        
        if config.framework == 'sentence-transformers':
            return await self._rerank_with_sentence_transformers(
                query_patent, initial_results, top_k
            )
        elif config.framework == 'cohere':
            return await self._rerank_with_cohere(
                query_patent, initial_results, top_k
            )
        elif config.framework == 'llm':
            # Use existing PydanticAI-based reranking
            from cross_encoder_reranker import CrossEncoderReranker
            reranker = CrossEncoderReranker()
            return await reranker.rerank_results(query_patent, initial_results, len(initial_results))
        
        return initial_results[:top_k]
    
    async def _rerank_with_sentence_transformers(self,
                                               query_patent: Dict,
                                               candidates: List[SearchResult],
                                               top_k: int) -> List[SearchResult]:
        """Rerank using sentence-transformers CrossEncoder."""
        
        # Prepare query-candidate pairs
        query_text = query_patent['abstract'][:self.reranker_config.max_length]
        
        pairs = []
        for candidate in candidates:
            candidate_text = candidate.abstract[:self.reranker_config.max_length]
            pairs.append([query_text, candidate_text])
        
        # Get reranking scores
        scores = self.reranker_model.predict(pairs)
        
        # Create reranked results
        scored_results = list(zip(candidates, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        reranked = []
        for i, (result, score) in enumerate(scored_results[:top_k]):
            result.reranked_score = float(score)
            result.reranked_rank = i + 1
            reranked.append(result)
        
        logger.info(f"Reranked {len(candidates)} candidates to top {len(reranked)}")
        return reranked
    
    async def _rerank_with_cohere(self,
                                query_patent: Dict,
                                candidates: List[SearchResult], 
                                top_k: int) -> List[SearchResult]:
        """Rerank using Cohere Rerank API."""
        
        query_text = query_patent['abstract']
        documents = [candidate.abstract for candidate in candidates]
        
        try:
            response = self.reranker_model.rerank(
                model='rerank-english-v2.0',
                query=query_text,
                documents=documents,
                top_k=top_k
            )
            
            # Create reranked results
            reranked = []
            for i, result in enumerate(response.results):
                original_candidate = candidates[result.index]
                original_candidate.reranked_score = result.relevance_score
                original_candidate.reranked_rank = i + 1
                reranked.append(original_candidate)
            
            logger.info(f"Cohere reranked {len(candidates)} to top {len(reranked)}")
            return reranked
            
        except Exception as e:
            logger.error(f"Cohere reranking failed: {e}")
            if self.fallback_to_llm:
                logger.info("Falling back to PydanticAI-based LLM reranking")
                # Fallback to PydanticAI-based LLM approach
                from cross_encoder_reranker import CrossEncoderReranker
                reranker = CrossEncoderReranker()
                return await reranker.rerank_results(query_patent, candidates, top_k)
            raise
    
    async def benchmark_rerankers(self,
                                test_patent_ids: List[str],
                                output_file: str = "reranker_benchmark.json") -> Dict[str, Any]:
        """Benchmark different reranker models."""
        
        logger.info(f"Benchmarking rerankers on {len(test_patent_ids)} patents")
        
        results = {}
        
        for reranker_type in ['bge-reranker-base', 'llm-cross-encoder']:
            logger.info(f"Testing {reranker_type}")
            
            try:
                # Create reranker system
                system = AdvancedRerankerSystem(
                    self.embeddings_file, 
                    reranker_type,
                    fallback_to_llm=True
                )
                
                # Test on sample patents
                start_time = time.time()
                successful_tests = 0
                
                for patent_id in test_patent_ids[:3]:  # Limit for benchmarking
                    try:
                        # Get initial candidates
                        initial_results = self.embedding_searcher.search_similar(patent_id, 20)
                        
                        if initial_results:
                            # Find query patent
                            query_patent = None
                            for p in self.embedding_searcher.embeddings:
                                if p['id'] == patent_id:
                                    query_patent = p
                                    break
                            
                            if query_patent:
                                # Rerank
                                reranked = await system.rerank_results(
                                    query_patent, initial_results, 10
                                )
                                successful_tests += 1
                    
                    except Exception as e:
                        logger.warning(f"Test failed for {patent_id}: {e}")
                
                processing_time = time.time() - start_time
                
                results[reranker_type] = {
                    'successful_tests': successful_tests,
                    'total_tests': len(test_patent_ids[:3]),
                    'success_rate': successful_tests / len(test_patent_ids[:3]),
                    'average_time': processing_time / max(successful_tests, 1),
                    'framework': system.reranker_config.framework
                }
                
            except Exception as e:
                logger.error(f"Failed to test {reranker_type}: {e}")
                results[reranker_type] = {
                    'error': str(e),
                    'successful_tests': 0
                }
        
        # Save benchmark results
        import json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Benchmark results saved to {output_file}")
        return results


async def main():
    """Test and benchmark reranker systems."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced reranker system testing")
    parser.add_argument("embeddings_file", help="Embeddings file")
    parser.add_argument("--reranker", default="bge-reranker-base", 
                       choices=['bge-reranker-base', 'bge-reranker-large', 'cohere-rerank', 'llm-cross-encoder'],
                       help="Reranker model to use")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark comparison")
    parser.add_argument("--query", help="Single patent ID to test")
    
    args = parser.parse_args()
    
    try:
        system = AdvancedRerankerSystem(
            args.embeddings_file,
            args.reranker,
            fallback_to_llm=True
        )
        
        if args.benchmark:
            # Run benchmark
            test_patents = [p['id'] for p in system.embedding_searcher.embeddings[:10]]
            results = await system.benchmark_rerankers(test_patents)
            
            print("\n🔬 RERANKER BENCHMARK RESULTS:")
            print("="*50)
            for reranker, data in results.items():
                if 'error' not in data:
                    print(f"{reranker}:")
                    print(f"  Success Rate: {data['success_rate']:.1%}")
                    print(f"  Avg Time: {data['average_time']:.1f}s")
                    print(f"  Framework: {data['framework']}")
                else:
                    print(f"{reranker}: FAILED - {data['error']}")
        
        elif args.query:
            # Single query test
            query_patent = None
            for p in system.embedding_searcher.embeddings:
                if p['id'] == args.query:
                    query_patent = p
                    break
            
            if query_patent:
                initial_results = system.embedding_searcher.search_similar(args.query, 20)
                reranked = await system.rerank_results(query_patent, initial_results, 10)
                
                print(f"\n🎯 RERANKING RESULTS for {args.query}:")
                print("="*50)
                for i, result in enumerate(reranked[:5]):
                    score = getattr(result, 'reranked_score', 'N/A')
                    print(f"{i+1}. {result.patent_id} (Score: {score})")
            else:
                print(f"Patent {args.query} not found")
        
    except Exception as e:
        logger.error(f"Error running advanced reranker: {e}")


if __name__ == "__main__":
    asyncio.run(main())