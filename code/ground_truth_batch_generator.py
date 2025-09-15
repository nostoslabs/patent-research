"""Ground truth batch request generator for Gemini Batch API."""

import json
import numpy as np
import logging
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity

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


class BatchRequestGenerator:
    """Generate batch requests for patent similarity evaluation."""
    
    def __init__(self, embeddings_file: str):
        """Initialize with embeddings data."""
        self.embeddings = self.load_embeddings(embeddings_file)
        
        # Pre-compute similarity matrix for efficient pair selection
        self.similarity_matrix = self._compute_similarity_matrix()
        
        logger.info(f"Initialized with {len(self.embeddings)} patents")
    
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
    
    def generate_diverse_pairs(self, n_pairs: int = 10000) -> List[PatentPair]:
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
    
    def create_batch_request(self, pair: PatentPair, pair_id: str) -> Dict[str, Any]:
        """Create a single batch request for patent pair comparison."""
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
        
        # Response schema for structured output
        response_schema = {
            "type": "object",
            "properties": {
                "similarity_score": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Overall similarity score between 0 and 1"
                },
                "technical_field_match": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "How similar are the technical fields (0=different, 1=identical)"
                },
                "problem_similarity": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "How similar are the problems being solved"
                },
                "solution_similarity": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "How similar are the proposed solutions"
                },
                "explanation": {
                    "type": "string",
                    "description": "Detailed explanation of similarity assessment"
                },
                "key_concepts_1": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key technical concepts from patent 1"
                },
                "key_concepts_2": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key technical concepts from patent 2"
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Confidence in this assessment"
                }
            },
            "required": [
                "similarity_score", "technical_field_match", "problem_similarity",
                "solution_similarity", "explanation", "key_concepts_1", 
                "key_concepts_2", "confidence"
            ]
        }
        
        # Create batch request
        batch_request = {
            "key": pair_id,
            "request": {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "response_mime_type": "application/json",
                    "response_schema": response_schema
                },
                "systemInstruction": {
                    "parts": [{
                        "text": """You are an expert patent examiner with deep technical knowledge across multiple fields. Your task is to evaluate the similarity between two patent abstracts.

Consider these aspects:
1. **Technical Field**: Are they in the same or related technical domains?
2. **Problem Solved**: Do they address similar problems or challenges?
3. **Solution Approach**: Are the proposed solutions similar in method or outcome?
4. **Technical Depth**: Consider the technical sophistication and novelty

Provide a thorough analysis with specific evidence from the abstracts. Be precise in your similarity scoring:
- 0.9-1.0: Nearly identical inventions (potential interference)
- 0.7-0.9: Very similar inventions in same field with similar solutions
- 0.5-0.7: Related inventions with some common elements
- 0.3-0.5: Loosely related inventions in similar fields
- 0.0-0.3: Different inventions with minimal similarity

Extract key technical concepts and provide detailed reasoning for your assessment."""
                    }]
                }
            },
            "metadata": {
                "patent1_id": patent1['id'],
                "patent2_id": patent2['id'],
                "embedding_similarity": pair.embedding_similarity,
                "similarity_category": pair.category,
                "classification1": patent1.get('classification', ''),
                "classification2": patent2.get('classification', ''),
                "abstract_length1": patent1.get('abstract_length', len(patent1['abstract'])),
                "abstract_length2": patent2.get('abstract_length', len(patent2['abstract']))
            }
        }
        
        return batch_request
    
    def generate_batch_file(self, 
                           output_file: str,
                           n_pairs: int = 10000) -> str:
        """Generate complete batch request file."""
        logger.info(f"Starting batch file generation for {n_pairs} pairs")
        
        start_time = time.time()
        
        # Generate diverse pairs
        pairs = self.generate_diverse_pairs(n_pairs)
        logger.info(f"Selected {len(pairs)} diverse pairs")
        
        if not pairs:
            raise ValueError("No pairs generated. Check your embeddings data.")
        
        # Write batch requests to JSONL file
        requests_written = 0
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, pair in enumerate(pairs):
                pair_id = f"pair_{pair.patent1['id']}_{pair.patent2['id']}"
                batch_request = self.create_batch_request(pair, pair_id)
                
                json.dump(batch_request, f, ensure_ascii=False)
                f.write('\n')
                requests_written += 1
                
                if (i + 1) % 1000 == 0:
                    logger.info(f"Generated {i + 1}/{len(pairs)} requests")
        
        elapsed_time = time.time() - start_time
        
        # Generate summary
        summary = {
            'total_pairs_requested': n_pairs,
            'requests_generated': requests_written,
            'generation_time_seconds': elapsed_time,
            'output_file': output_file,
            'estimated_cost_usd': self._estimate_batch_cost(requests_written),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save summary
        summary_file = output_file.replace('.jsonl', '_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Batch file generation completed!")
        logger.info(f"Requests generated: {requests_written}")
        logger.info(f"Generation time: {elapsed_time:.1f} seconds")
        logger.info(f"Estimated cost: ${summary['estimated_cost_usd']:.2f}")
        logger.info(f"Batch file saved to: {output_file}")
        logger.info(f"Summary saved to: {summary_file}")
        
        return output_file
    
    def _estimate_batch_cost(self, num_requests: int) -> float:
        """Estimate cost for batch processing (50% discount applied)."""
        # Gemini 1.5 Flash batch pricing (50% of regular)
        input_cost_per_1m = 0.075 / 2  # $0.0375 per 1M tokens
        output_cost_per_1m = 0.30 / 2  # $0.15 per 1M tokens
        
        # Estimate tokens (rough: 1 token ≈ 4 characters)
        chars_per_request = 6000  # Two patents + prompt
        tokens_per_request = chars_per_request / 4
        
        total_input_tokens = (tokens_per_request * num_requests) / 1_000_000
        total_output_tokens = (200 * num_requests) / 1_000_000  # ~200 tokens output
        
        total_cost = (total_input_tokens * input_cost_per_1m + 
                     total_output_tokens * output_cost_per_1m)
        
        return total_cost


def main():
    """CLI interface for batch request generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate batch requests for patent similarity evaluation")
    parser.add_argument("embeddings_file", help="Input embeddings JSONL file")
    parser.add_argument("--output", default="ground_truth_batch_requests.jsonl", help="Output batch file")
    parser.add_argument("--pairs", type=int, default=10000, help="Number of pairs to generate")
    
    args = parser.parse_args()
    
    # Check if embeddings file exists
    if not Path(args.embeddings_file).exists():
        logger.error(f"Embeddings file not found: {args.embeddings_file}")
        return
    
    try:
        generator = BatchRequestGenerator(args.embeddings_file)
        generator.generate_batch_file(
            output_file=args.output,
            n_pairs=args.pairs
        )
    except Exception as e:
        logger.error(f"Failed to generate batch file: {e}")
        raise


if __name__ == "__main__":
    main()