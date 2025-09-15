"""Experimental embedding generation with comprehensive chunking strategies."""

import json
import time
import re
from pathlib import Path
from typing import Any, List, Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import ollama
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ChunkResult:
    """Result of chunking a text."""
    text: str
    start_pos: int
    end_pos: int
    token_count: int
    

@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    embedding: List[float]
    processing_time: float
    was_truncated: bool


class TextChunker:
    """Handles various text chunking strategies."""
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimation (4 chars per token average)."""
        return len(text) // 4
    
    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """Split text into sentences using basic regex."""
        # Handle common sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk_fixed_size(self, text: str, chunk_size: int = 512) -> List[ChunkResult]:
        """Fixed-size chunking in characters (approximating tokens)."""
        char_size = chunk_size * 4  # ~4 chars per token
        chunks = []
        
        for i in range(0, len(text), char_size):
            chunk_text = text[i:i + char_size]
            chunks.append(ChunkResult(
                text=chunk_text,
                start_pos=i,
                end_pos=min(i + char_size, len(text)),
                token_count=self.estimate_tokens(chunk_text)
            ))
        
        return chunks
    
    def chunk_with_overlap(self, text: str, chunk_size: int = 550, overlap: int = 50) -> List[ChunkResult]:
        """Chunking with overlap between chunks."""
        char_chunk_size = chunk_size * 4
        char_overlap = overlap * 4
        step_size = char_chunk_size - char_overlap
        
        chunks = []
        for i in range(0, len(text), step_size):
            end_pos = min(i + char_chunk_size, len(text))
            chunk_text = text[i:end_pos]
            
            # Don't create tiny chunks at the end
            if len(chunk_text) < char_overlap:
                break
                
            chunks.append(ChunkResult(
                text=chunk_text,
                start_pos=i,
                end_pos=end_pos,
                token_count=self.estimate_tokens(chunk_text)
            ))
            
            # Break if we've reached the end
            if end_pos >= len(text):
                break
        
        return chunks
    
    def chunk_sentence_boundary(self, text: str, target_tokens: int = 512) -> List[ChunkResult]:
        """Chunk text at sentence boundaries, targeting token count."""
        sentences = self.split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        start_pos = 0
        
        for sentence in sentences:
            sentence_tokens = self.estimate_tokens(sentence)
            
            # If adding this sentence would exceed target, finalize current chunk
            if current_tokens + sentence_tokens > target_tokens and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(ChunkResult(
                    text=chunk_text,
                    start_pos=start_pos,
                    end_pos=start_pos + len(chunk_text),
                    token_count=current_tokens
                ))
                
                # Start new chunk
                current_chunk = [sentence]
                current_tokens = sentence_tokens
                start_pos = start_pos + len(chunk_text) + 1
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add final chunk if exists
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(ChunkResult(
                text=chunk_text,
                start_pos=start_pos,
                end_pos=start_pos + len(chunk_text),
                token_count=current_tokens
            ))
        
        return chunks
    
    def chunk_semantic(self, text: str, model: str = "embeddinggemma", 
                      similarity_threshold: float = 0.7) -> List[ChunkResult]:
        """Semantic chunking based on sentence similarity."""
        sentences = self.split_into_sentences(text)
        
        if len(sentences) <= 2:
            return [ChunkResult(text=text, start_pos=0, end_pos=len(text), 
                              token_count=self.estimate_tokens(text))]
        
        # Get embeddings for each sentence
        sentence_embeddings = []
        for sentence in sentences:
            try:
                response = ollama.embeddings(model=model, prompt=sentence)
                sentence_embeddings.append(np.array(response['embedding']))
            except Exception as e:
                print(f"Warning: Could not embed sentence for semantic chunking: {e}")
                # Fallback to sentence boundary chunking
                return self.chunk_sentence_boundary(text)
        
        # Find semantic boundaries
        boundaries = [0]  # Always start at 0
        
        for i in range(1, len(sentence_embeddings)):
            similarity = cosine_similarity(
                sentence_embeddings[i-1].reshape(1, -1),
                sentence_embeddings[i].reshape(1, -1)
            )[0][0]
            
            if similarity < similarity_threshold:
                boundaries.append(i)
        
        boundaries.append(len(sentences))  # Always end at last sentence
        
        # Create chunks from boundaries
        chunks = []
        text_pos = 0
        
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            
            chunk_sentences = sentences[start_idx:end_idx]
            chunk_text = ' '.join(chunk_sentences)
            
            chunks.append(ChunkResult(
                text=chunk_text,
                start_pos=text_pos,
                end_pos=text_pos + len(chunk_text),
                token_count=self.estimate_tokens(chunk_text)
            ))
            
            text_pos += len(chunk_text) + 1
        
        return chunks


class EmbeddingAggregator:
    """Handles various embedding aggregation strategies."""
    
    @staticmethod
    def mean_pooling(embeddings: List[np.ndarray]) -> np.ndarray:
        """Simple mean pooling of embeddings."""
        return np.mean(embeddings, axis=0)
    
    @staticmethod
    def max_pooling(embeddings: List[np.ndarray]) -> np.ndarray:
        """Element-wise maximum pooling."""
        return np.max(embeddings, axis=0)
    
    @staticmethod
    def weighted_average(embeddings: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
        """Weighted average of embeddings."""
        if weights is None:
            # Default: weight by position (first and last chunks more important)
            n = len(embeddings)
            if n == 1:
                weights = [1.0]
            elif n == 2:
                weights = [0.6, 0.4]
            else:
                weights = [0.3] + [0.4 / (n-2)] * (n-2) + [0.3]
        
        weighted_embeddings = [w * emb for w, emb in zip(weights, embeddings)]
        return np.sum(weighted_embeddings, axis=0)
    
    @staticmethod
    def attention_weighted(embeddings: List[np.ndarray], query_embedding: np.ndarray) -> np.ndarray:
        """Attention-based weighted average using query similarity."""
        if len(embeddings) == 1:
            return embeddings[0]
        
        # Calculate attention weights based on similarity to mean
        similarities = [cosine_similarity(emb.reshape(1, -1), query_embedding.reshape(1, -1))[0][0] 
                       for emb in embeddings]
        
        # Softmax to normalize weights
        exp_similarities = np.exp(similarities)
        weights = exp_similarities / np.sum(exp_similarities)
        
        weighted_embeddings = [w * emb for w, emb in zip(weights, embeddings)]
        return np.sum(weighted_embeddings, axis=0)


class ExperimentalEmbeddingGenerator:
    """Generate embeddings using multiple chunking and aggregation strategies."""
    
    def __init__(self, model: str = "embeddinggemma", token_limit: int = 2000):
        self.model = model
        self.token_limit = token_limit
        self.char_limit = token_limit * 4  # Rough approximation
        self.chunker = TextChunker()
        self.aggregator = EmbeddingAggregator()
        
    def generate_embedding(self, text: str) -> EmbeddingResult:
        """Generate embedding for text, handling truncation."""
        start_time = time.time()
        was_truncated = False
        
        # Check if text needs truncation
        if len(text) > self.char_limit:
            text = text[:self.char_limit]
            was_truncated = True
        
        try:
            response = ollama.embeddings(model=self.model, prompt=text)
            embedding = response['embedding']
        except Exception as e:
            print(f"Error generating embedding: {e}")
            embedding = [0.0] * 768  # Default embedding size
        
        processing_time = time.time() - start_time
        
        return EmbeddingResult(
            embedding=embedding,
            processing_time=processing_time,
            was_truncated=was_truncated
        )
    
    def generate_chunk_embeddings(self, chunks: List[ChunkResult]) -> List[EmbeddingResult]:
        """Generate embeddings for all chunks."""
        chunk_embeddings = []
        
        for chunk in chunks:
            result = self.generate_embedding(chunk.text)
            chunk_embeddings.append(result)
        
        return chunk_embeddings
    
    def process_text_comprehensive(self, text: str, patent_id: str) -> Dict[str, Any]:
        """Process text with all chunking and aggregation strategies."""
        print(f"Processing {patent_id} ({len(text)} chars)...")
        
        results = {
            "id": patent_id,
            "text_length": len(text),
            "estimated_tokens": self.chunker.estimate_tokens(text),
            "embeddings": {}
        }
        
        # 1. Original embedding (potentially truncated)
        print("  Generating original embedding...")
        original_result = self.generate_embedding(text)
        results["embeddings"]["original"] = {
            "embedding": original_result.embedding,
            "processing_time": original_result.processing_time,
            "was_truncated": original_result.was_truncated
        }
        
        # Skip chunking if text is short enough
        if len(text) <= self.char_limit:
            results["embeddings"]["chunking_needed"] = False
            return results
        
        results["embeddings"]["chunking_needed"] = True
        results["embeddings"]["chunking_strategies"] = {}
        
        # 2. Chunking strategies
        chunking_strategies = {
            "fixed_512": lambda: self.chunker.chunk_fixed_size(text, 512),
            "fixed_768": lambda: self.chunker.chunk_fixed_size(text, 768),
            "overlapping_550": lambda: self.chunker.chunk_with_overlap(text, 550, 50),
            "sentence_boundary_512": lambda: self.chunker.chunk_sentence_boundary(text, 512),
            "sentence_boundary_768": lambda: self.chunker.chunk_sentence_boundary(text, 768),
            "semantic": lambda: self.chunker.chunk_semantic(text, self.model)
        }
        
        for strategy_name, strategy_func in chunking_strategies.items():
            print(f"  Processing {strategy_name}...")
            start_time = time.time()
            
            try:
                # Generate chunks
                chunks = strategy_func()
                
                # Generate embeddings for chunks
                chunk_embeddings = self.generate_chunk_embeddings(chunks)
                
                # Extract numpy arrays for aggregation
                embedding_arrays = [np.array(result.embedding) for result in chunk_embeddings]
                
                # Generate all aggregation methods
                aggregations = {}
                
                if embedding_arrays:
                    aggregations["mean"] = self.aggregator.mean_pooling(embedding_arrays).tolist()
                    aggregations["max"] = self.aggregator.max_pooling(embedding_arrays).tolist()
                    aggregations["weighted"] = self.aggregator.weighted_average(embedding_arrays).tolist()
                    
                    # Attention-weighted using mean as query
                    mean_embedding = np.array(aggregations["mean"])
                    aggregations["attention"] = self.aggregator.attention_weighted(
                        embedding_arrays, mean_embedding
                    ).tolist()
                
                # Store results
                results["embeddings"]["chunking_strategies"][strategy_name] = {
                    "num_chunks": len(chunks),
                    "chunks": [
                        {
                            "text": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                            "start_pos": chunk.start_pos,
                            "end_pos": chunk.end_pos,
                            "token_count": chunk.token_count,
                            "embedding": chunk_result.embedding,
                            "processing_time": chunk_result.processing_time
                        }
                        for chunk, chunk_result in zip(chunks, chunk_embeddings)
                    ],
                    "aggregations": aggregations,
                    "total_processing_time": time.time() - start_time
                }
                
            except Exception as e:
                print(f"    Error with {strategy_name}: {e}")
                results["embeddings"]["chunking_strategies"][strategy_name] = {
                    "error": str(e)
                }
        
        return results


def process_patents_experimental(
    input_file: str,
    output_file: str,
    max_records: Optional[int] = None,
    model: str = "embeddinggemma"
) -> None:
    """Process patents with experimental chunking strategies."""
    
    print(f"Experimental Patent Embedding Generation")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Model: {model}")
    print(f"Max records: {max_records or 'All'}")
    print("=" * 60)
    
    # Initialize generator
    generator = ExperimentalEmbeddingGenerator(model=model)
    
    # Test model connection
    print("Testing model connection...")
    test_result = generator.generate_embedding("test")
    print(f"Model ready. Embedding dimension: {len(test_result.embedding)}")
    print()
    
    # Load patents
    patents = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_records and i >= max_records:
                break
            patents.append(json.loads(line.strip()))
    
    print(f"Loaded {len(patents)} patents for processing")
    
    # Process patents
    results = []
    total_start_time = time.time()
    
    for i, patent in enumerate(patents, 1):
        print(f"Processing {i}/{len(patents)}: {patent['id']}")
        
        # Get abstract text
        abstract = patent.get('abstract', '')
        
        if not abstract.strip():
            print(f"  Skipping {patent['id']} - no abstract")
            continue
        
        # Process with comprehensive strategies
        result = generator.process_text_comprehensive(abstract, patent['id'])
        
        # Add original patent metadata
        result.update({
            "classification": patent.get('classification', ''),
            "abstract": abstract,
            "abstract_length": len(abstract)
        })
        
        results.append(result)
        
        # Save progress every 10 records
        if i % 10 == 0:
            print(f"  Saving progress... ({i}/{len(patents)} records)")
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    json.dump(result, f, ensure_ascii=False)
                    f.write('\n')
    
    # Save final results
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')
    
    total_time = time.time() - total_start_time
    print(f"\n" + "=" * 60)
    print(f"EXPERIMENTAL PROCESSING COMPLETE")
    print(f"Processed: {len(results)} patents")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average time per patent: {total_time/len(results):.1f} seconds")
    print(f"Output saved to: {output_file}")


def main() -> None:
    """Main function for experimental processing."""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python generate_embeddings_experimental.py <input_file> <output_file> [max_records]")
        print("Example: python generate_embeddings_experimental.py patent_abstracts.jsonl patent_experimental_embeddings.jsonl 100")
        return
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    max_records = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    if not Path(input_file).exists():
        print(f"Error: Input file {input_file} not found")
        return
    
    process_patents_experimental(input_file, output_file, max_records)


if __name__ == "__main__":
    main()