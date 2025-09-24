"""Multi-model embedding generation with comprehensive chunking strategies."""

import json
import time
import re
from pathlib import Path
from typing import Any, List, Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import ollama
from sklearn.metrics.pairwise import cosine_similarity

from experiment_tracker import ExperimentTracker, ModelConfig


@dataclass
class ModelSpecs:
    """Runtime specifications for embedding models."""
    name: str
    context_limit: int
    embedding_dim: int
    char_limit: int
    test_prompt: str = "test"
    
    @classmethod
    def from_model_config(cls, config: ModelConfig) -> 'ModelSpecs':
        return cls(
            name=config.name,
            context_limit=config.context_limit,
            embedding_dim=config.embedding_dim,
            char_limit=config.char_limit
        )


class MultiModelEmbeddingGenerator:
    """Generate embeddings using multiple models with comprehensive chunking."""
    
    def __init__(self, experiment_tracker: ExperimentTracker):
        self.tracker = experiment_tracker
        self.available_models: Dict[str, ModelSpecs] = {}
        self._discover_available_models()
    
    def _discover_available_models(self) -> None:
        """Discover and validate available embedding models."""
        print("Discovering available embedding models...")
        
        # Get model configs from tracker
        for model_name, model_config in self.tracker.model_configs.items():
            try:
                # Test model availability
                print(f"  Testing {model_name}...")
                response = ollama.embeddings(model=model_name, prompt="test")
                
                # Validate embedding dimension
                actual_dim = len(response['embedding'])
                expected_dim = model_config.embedding_dim
                
                if actual_dim != expected_dim:
                    print(f"    Warning: Expected {expected_dim}D, got {actual_dim}D")
                    # Update the config with actual dimension
                    model_config.embedding_dim = actual_dim
                    self.tracker.add_model_config(model_config)
                
                # Create runtime specs
                specs = ModelSpecs.from_model_config(model_config)
                specs.embedding_dim = actual_dim  # Use actual dimension
                
                self.available_models[model_name] = specs
                print(f"    ✅ {model_name}: {actual_dim}D embeddings, {specs.context_limit} token limit")
                
            except Exception as e:
                print(f"    ❌ {model_name}: {e}")
                continue
        
        print(f"\nFound {len(self.available_models)} available models:")
        for model_name in self.available_models:
            print(f"  - {model_name}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        return list(self.available_models.keys())
    
    def estimate_tokens(self, text: str, model_specs: ModelSpecs) -> int:
        """Estimate token count for given text and model."""
        return len(text) // int(model_specs.char_limit / model_specs.context_limit)
    
    def needs_chunking(self, text: str, model_specs: ModelSpecs) -> bool:
        """Check if text needs chunking for given model."""
        return len(text) > model_specs.char_limit
    
    def generate_embedding(self, text: str, model_name: str) -> Dict[str, Any]:
        """Generate embedding with model-specific handling."""
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not available")
        
        model_specs = self.available_models[model_name]
        start_time = time.time()
        was_truncated = False
        
        # Check if text needs truncation
        if len(text) > model_specs.char_limit:
            original_length = len(text)
            text = text[:model_specs.char_limit]
            was_truncated = True
            print(f"    Truncated from {original_length} to {len(text)} chars for {model_name}")
        
        try:
            response = ollama.embeddings(model=model_name, prompt=text)
            embedding = response['embedding']
        except Exception as e:
            print(f"    Error generating embedding with {model_name}: {e}")
            # Return zero embedding as fallback
            embedding = [0.0] * model_specs.embedding_dim
        
        processing_time = time.time() - start_time
        
        return {
            "embedding": embedding,
            "processing_time": processing_time,
            "was_truncated": was_truncated,
            "model": model_name,
            "embedding_dim": len(embedding),
            "original_length": len(text) + (model_specs.char_limit - len(text) if was_truncated else 0)
        }
    
    def chunk_text_for_model(self, text: str, model_name: str, strategy: str = "fixed") -> List[str]:
        """Chunk text appropriately for specific model."""
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not available")
        
        model_specs = self.available_models[model_name]
        
        if not self.needs_chunking(text, model_specs):
            return [text]
        
        # Use model-specific chunk size (80% of limit for safety)
        target_chunk_size = int(model_specs.char_limit * 0.8)
        
        if strategy == "fixed":
            chunks = []
            for i in range(0, len(text), target_chunk_size):
                chunk = text[i:i + target_chunk_size]
                chunks.append(chunk)
            return chunks
        
        elif strategy == "overlapping":
            overlap_size = int(target_chunk_size * 0.1)  # 10% overlap
            chunks = []
            i = 0
            while i < len(text):
                end_pos = min(i + target_chunk_size, len(text))
                chunk = text[i:end_pos]
                chunks.append(chunk)
                
                if end_pos >= len(text):
                    break
                    
                i += (target_chunk_size - overlap_size)
            
            return chunks
        
        elif strategy == "sentence":
            # Split by sentences and group to fit in chunks
            sentences = re.split(r'(?<=[.!?])\s+', text)
            chunks = []
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                sentence_length = len(sentence)
                
                if current_length + sentence_length > target_chunk_size and current_chunk:
                    # Finalize current chunk
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_length
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_length
            
            # Add final chunk
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            return chunks
        
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
    
    def process_patent_multimodel(
        self,
        patent_data: Dict[str, Any],
        models: List[str],
        chunking_strategies: List[str] = None,
        aggregation_methods: List[str] = None
    ) -> Dict[str, Any]:
        """Process single patent with multiple models and strategies."""
        
        if chunking_strategies is None:
            chunking_strategies = ["fixed", "overlapping", "sentence"]
        
        if aggregation_methods is None:
            aggregation_methods = ["mean", "max", "weighted"]
        
        patent_id = patent_data.get("id", "unknown")
        abstract = patent_data.get("abstract", "")
        
        print(f"  Processing {patent_id} with {len(models)} models...")
        
        result = {
            "id": patent_id,
            "abstract": abstract,
            "abstract_length": len(abstract),
            "classification": patent_data.get("classification", ""),
            "models": {}
        }
        
        for model_name in models:
            if model_name not in self.available_models:
                print(f"    Skipping unavailable model: {model_name}")
                continue
            
            print(f"    Processing with {model_name}...")
            model_specs = self.available_models[model_name]
            
            model_result = {
                "model_specs": {
                    "name": model_name,
                    "context_limit": model_specs.context_limit,
                    "embedding_dim": model_specs.embedding_dim,
                    "char_limit": model_specs.char_limit
                },
                "needs_chunking": self.needs_chunking(abstract, model_specs),
                "embeddings": {}
            }
            
            # Original embedding (potentially truncated)
            original_embedding = self.generate_embedding(abstract, model_name)
            model_result["embeddings"]["original"] = original_embedding
            
            # Skip chunking if text fits in context window
            if not model_result["needs_chunking"]:
                result["models"][model_name] = model_result
                continue
            
            # Process with chunking strategies
            model_result["embeddings"]["chunked"] = {}
            
            for strategy in chunking_strategies:
                print(f"      Chunking strategy: {strategy}")
                strategy_start = time.time()
                
                try:
                    # Chunk text
                    chunks = self.chunk_text_for_model(abstract, model_name, strategy)
                    
                    # Generate embeddings for chunks
                    chunk_embeddings = []
                    chunk_results = []
                    
                    for i, chunk in enumerate(chunks):
                        chunk_embedding = self.generate_embedding(chunk, model_name)
                        chunk_embeddings.append(np.array(chunk_embedding["embedding"]))
                        
                        chunk_results.append({
                            "chunk_index": i,
                            "text": chunk[:200] + "..." if len(chunk) > 200 else chunk,
                            "text_length": len(chunk),
                            "embedding": chunk_embedding["embedding"],
                            "processing_time": chunk_embedding["processing_time"]
                        })
                    
                    # Aggregate embeddings
                    aggregations = {}
                    
                    if chunk_embeddings:
                        chunk_array = np.array(chunk_embeddings)
                        
                        if "mean" in aggregation_methods:
                            aggregations["mean"] = np.mean(chunk_array, axis=0).tolist()
                        
                        if "max" in aggregation_methods:
                            aggregations["max"] = np.max(chunk_array, axis=0).tolist()
                        
                        if "weighted" in aggregation_methods:
                            # Position-based weighting (first and last chunks more important)
                            n_chunks = len(chunk_embeddings)
                            if n_chunks == 1:
                                weights = [1.0]
                            elif n_chunks == 2:
                                weights = [0.6, 0.4]
                            else:
                                weights = [0.3] + [0.4 / (n_chunks - 2)] * (n_chunks - 2) + [0.3]
                            
                            weighted_sum = sum(w * emb for w, emb in zip(weights, chunk_embeddings))
                            aggregations["weighted"] = weighted_sum.tolist()
                    
                    strategy_time = time.time() - strategy_start
                    
                    model_result["embeddings"]["chunked"][strategy] = {
                        "num_chunks": len(chunks),
                        "chunks": chunk_results,
                        "aggregations": aggregations,
                        "processing_time": strategy_time
                    }
                
                except Exception as e:
                    print(f"        Error with {strategy}: {e}")
                    model_result["embeddings"]["chunked"][strategy] = {
                        "error": str(e)
                    }
            
            result["models"][model_name] = model_result
        
        return result


def process_patents_multimodel(
    input_file: str,
    output_file: str,
    models: List[str],
    max_records: Optional[int] = None,
    chunking_strategies: List[str] = None,
    aggregation_methods: List[str] = None,
    experiment_id: str = None
) -> None:
    """Process patents with multiple models and comprehensive analysis."""
    
    print(f"Multi-Model Patent Embedding Generation")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Models: {', '.join(models)}")
    print(f"Max records: {max_records or 'All'}")
    print("=" * 60)
    
    # Initialize tracker and generator
    tracker = ExperimentTracker()
    generator = MultiModelEmbeddingGenerator(tracker)
    
    # Validate models
    available_models = generator.get_available_models()
    valid_models = [m for m in models if m in available_models]
    invalid_models = [m for m in models if m not in available_models]
    
    if invalid_models:
        print(f"Warning: Unavailable models: {', '.join(invalid_models)}")
    
    if not valid_models:
        print("Error: No valid models available")
        return
    
    print(f"Using models: {', '.join(valid_models)}")
    print()
    
    # Load patents
    patents = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_records and i >= max_records:
                break
            patents.append(json.loads(line.strip()))
    
    print(f"Loaded {len(patents)} patents for processing")
    
    # Start experiment tracking
    if experiment_id:
        try:
            experiment = tracker.get_experiment(experiment_id)
            if experiment:
                tracker.start_experiment(experiment_id, len(patents))
            else:
                print(f"Warning: Experiment {experiment_id} not found in tracker")
        except Exception as e:
            print(f"Warning: Could not start experiment tracking: {e}")
    
    # Process patents
    results = []
    total_start_time = time.time()
    processed_count = 0
    success_count = 0
    
    for i, patent in enumerate(patents, 1):
        print(f"Processing {i}/{len(patents)}: {patent.get('id', 'unknown')}")
        
        try:
            result = generator.process_patent_multimodel(
                patent, valid_models, chunking_strategies, aggregation_methods
            )
            results.append(result)
            success_count += 1
            
        except Exception as e:
            print(f"  Error processing patent: {e}")
            # Add error record
            results.append({
                "id": patent.get("id", "unknown"),
                "error": str(e),
                "abstract_length": len(patent.get("abstract", "")),
                "classification": patent.get("classification", "")
            })
        
        processed_count += 1
        
        # Update progress tracking
        if experiment_id:
            try:
                tracker.update_progress(experiment_id, processed_count, success_count, processed_count - success_count)
            except Exception as e:
                pass  # Continue even if tracking fails
        
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
    
    # Complete experiment tracking
    if experiment_id:
        try:
            results_summary = {
                "total_patents": len(patents),
                "successful": success_count,
                "failed": processed_count - success_count,
                "models_used": valid_models,
                "processing_time_minutes": total_time / 60,
                "avg_time_per_patent": total_time / len(patents)
            }
            
            output_files = {
                "embeddings": output_file
            }
            
            tracker.complete_experiment(experiment_id, output_files, results_summary)
        except Exception as e:
            print(f"Warning: Could not complete experiment tracking: {e}")
    
    print(f"\n" + "=" * 60)
    print(f"MULTI-MODEL PROCESSING COMPLETE")
    print(f"Processed: {processed_count}/{len(patents)} patents")
    print(f"Successful: {success_count}")
    print(f"Failed: {processed_count - success_count}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average time per patent: {total_time/len(patents):.1f} seconds")
    print(f"Models used: {', '.join(valid_models)}")
    print(f"Output saved to: {output_file}")


def main() -> None:
    """Main function for multi-model processing."""
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python generate_embeddings_multimodel.py <input_file> <output_file> <models> [max_records] [experiment_id]")
        print("Example: python generate_embeddings_multimodel.py patent_abstracts.jsonl multi_model_embeddings.jsonl embeddinggemma,bge-m3 100")
        return
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    models = sys.argv[3].split(',')
    max_records = int(sys.argv[4]) if len(sys.argv) > 4 else None
    experiment_id = sys.argv[5] if len(sys.argv) > 5 else None
    
    if not Path(input_file).exists():
        print(f"Error: Input file {input_file} not found")
        return
    
    process_patents_multimodel(
        input_file=input_file,
        output_file=output_file,
        models=models,
        max_records=max_records,
        experiment_id=experiment_id
    )


if __name__ == "__main__":
    main()