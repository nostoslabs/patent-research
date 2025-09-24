"""
Generate OpenAI embeddings for ground truth evaluation.
This creates embeddings for the same patent pairs used in our ground truth evaluation
to enable direct comparison with our validated models.
"""

import json
import numpy as np
import openai
import os
from pathlib import Path
from typing import Dict, List
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIGroundTruthEvaluator:
    """Generate OpenAI embeddings for ground truth comparison."""
    
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
        
    def load_ground_truth_patents(self, ground_truth_file: str) -> Dict[str, str]:
        """Extract unique patent IDs from ground truth dataset and load abstracts from original data."""
        patent_ids = set()
        
        # First, collect all patent IDs from ground truth
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line.strip())
                if record.get('success', False):
                    patent_ids.add(record['patent1_id'])
                    patent_ids.add(record['patent2_id'])
        
        logger.info(f"Found {len(patent_ids)} unique patent IDs in ground truth dataset")
        
        # Load abstracts from the original patent data
        patents = {}
        patent_files = [
            'data/patent_abstracts.jsonl',
            'patent_abstracts.jsonl'
        ]
        
        for patent_file in patent_files:
            try:
                with open(patent_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line.strip())
                        patent_id = data.get('id')
                        abstract = data.get('abstract', '')
                        
                        if patent_id in patent_ids and abstract:
                            patents[patent_id] = abstract
                            
                logger.info(f"Loaded {len(patents)} patent abstracts from {patent_file}")
                break
            except FileNotFoundError:
                continue
        
        logger.info(f"Final dataset: {len(patents)} patents with abstracts")
        return patents
        
    def generate_embedding(self, text: str) -> tuple:
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
    
    def generate_all_embeddings(self, patents: Dict[str, str], output_file: str, delay: float = 0.1):
        """Generate embeddings for all patents with rate limiting."""
        embeddings = {}
        processed = 0
        
        logger.info(f"Generating OpenAI embeddings for {len(patents)} patents...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for patent_id, abstract in patents.items():
                try:
                    # Generate embedding
                    embedding, tokens = self.generate_embedding(abstract)
                    
                    # Store result
                    result = {
                        'id': patent_id,
                        'abstract': abstract,
                        'embedding': embedding.tolist(),
                        'tokens_used': tokens,
                        'model': self.model
                    }
                    
                    f.write(json.dumps(result) + '\n')
                    f.flush()
                    
                    processed += 1
                    if processed % 10 == 0:
                        logger.info(f"Processed {processed}/{len(patents)} patents. "
                                   f"Total cost: ${self.total_cost:.4f}")
                    
                    # Rate limiting
                    time.sleep(delay)
                    
                except Exception as e:
                    logger.error(f"Failed to process patent {patent_id}: {e}")
                    continue
        
        logger.info(f"Completed! Total tokens: {self.total_tokens}, Total cost: ${self.total_cost:.4f}")
        return embeddings
    
    def compute_similarities_for_ground_truth(self, embeddings_file: str, ground_truth_file: str, output_file: str):
        """Compute cosine similarities for ground truth pairs using OpenAI embeddings."""
        # Load OpenAI embeddings
        embeddings = {}
        with open(embeddings_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                embeddings[data['id']] = np.array(data['embedding'])
        
        logger.info(f"Loaded {len(embeddings)} OpenAI embeddings")
        
        # Process ground truth pairs
        results = []
        
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line.strip())
                if not record.get('success', False):
                    continue
                
                patent1_id = record['patent1_id']
                patent2_id = record['patent2_id']
                
                if patent1_id in embeddings and patent2_id in embeddings:
                    # Compute cosine similarity
                    emb1 = embeddings[patent1_id]
                    emb2 = embeddings[patent2_id]
                    
                    cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                    
                    # Store result with original ground truth data
                    result = record.copy()
                    result['openai_embedding_similarity'] = float(cos_sim)
                    results.append(result)
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        
        logger.info(f"Computed similarities for {len(results)} pairs, saved to {output_file}")
        return results


def main():
    """Run OpenAI ground truth evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate OpenAI embeddings for ground truth evaluation")
    parser.add_argument("ground_truth_file", help="Ground truth JSONL file")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        evaluator = OpenAIGroundTruthEvaluator(args.api_key)
        
        # Step 1: Extract patents from ground truth
        patents = evaluator.load_ground_truth_patents(args.ground_truth_file)
        
        # Step 2: Generate embeddings
        embeddings_file = output_dir / "openai_embeddings_ground_truth.jsonl"
        evaluator.generate_all_embeddings(patents, str(embeddings_file))
        
        # Step 3: Compute similarities for ground truth pairs
        similarities_file = output_dir / "openai_ground_truth_similarities.jsonl"
        evaluator.compute_similarities_for_ground_truth(
            str(embeddings_file), 
            args.ground_truth_file, 
            str(similarities_file)
        )
        
        logger.info("OpenAI ground truth evaluation completed!")
        logger.info(f"Embeddings: {embeddings_file}")
        logger.info(f"Similarities: {similarities_file}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()