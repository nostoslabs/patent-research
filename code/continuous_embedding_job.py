#!/usr/bin/env python3
"""Continuous embedding generation for large patent dataset."""

import json
import time
import signal
import sys
from pathlib import Path

from generate_embeddings import generate_embedding
import ollama


def signal_handler(sig, frame):
    """Handle graceful shutdown on Ctrl+C."""
    print("\nReceived interrupt signal. Gracefully shutting down...")
    print("Current progress has been saved.")
    sys.exit(0)


def continuous_embedding_generation(
    input_file: str = "patent_abstracts_100k_diverse.jsonl",
    output_file: str = "patent_abstracts_with_embeddings_large.jsonl",
    model: str = "embeddinggemma",
    batch_size: int = 50,
    save_frequency: int = 10,
    resume: bool = True
) -> None:
    """Generate embeddings continuously with progress saving."""
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("Starting continuous embedding generation...")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Model: {model}")
    print(f"Batch size: {batch_size}")
    print(f"Save frequency: every {save_frequency} records")
    print("-" * 60)
    
    # Test connection
    try:
        response = ollama.embeddings(model=model, prompt="test")
        print(f"Model {model} ready. Embedding dimension: {len(response['embedding'])}")
    except Exception as e:
        print(f"Error: Cannot connect to {model} model: {e}")
        return
    
    # Load input patents
    print("Loading patents...")
    patents = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            patents.append(json.loads(line.strip()))
    
    print(f"Loaded {len(patents)} patents")
    
    # Check for existing progress
    processed_ids = set()
    output_path = Path(output_file)
    
    if resume and output_path.exists():
        print("Checking existing progress...")
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    processed_ids.add(record['id'])
                except:
                    continue
        print(f"Found {len(processed_ids)} already processed patents")
    
    # Filter unprocessed patents
    unprocessed = [p for p in patents if p['id'] not in processed_ids]
    print(f"{len(unprocessed)} patents remaining to process")
    
    if not unprocessed:
        print("All patents already processed!")
        return
    
    # Process in batches
    start_time = time.time()
    processed_count = len(processed_ids)
    
    try:
        with open(output_path, 'a', encoding='utf-8') as f:
            for i, patent in enumerate(unprocessed, 1):
                print(f"Processing record {processed_count + i}/{len(patents)}: {patent['id']}")
                
                # Generate embedding
                embedding = generate_embedding(patent['abstract'], model=model)
                
                if embedding is not None:
                    # Create record with embedding
                    record = patent.copy()
                    record['embeddings'] = {
                        'embeddinggemma': {
                            'model': model,
                            'embedding': embedding.tolist(),
                            'text_source': 'abstract',
                            'embedding_dim': len(embedding),
                            'generated_at': time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                    }
                    
                    # Write record
                    json.dump(record, f, ensure_ascii=False)
                    f.write('\n')
                    
                    # Save progress periodically
                    if i % save_frequency == 0:
                        f.flush()  # Ensure data is written to disk
                        
                        elapsed = time.time() - start_time
                        rate = i / elapsed
                        remaining = len(unprocessed) - i
                        eta = remaining / rate if rate > 0 else 0
                        
                        print(f"Progress saved ({processed_count + i}/{len(patents)}) - "
                              f"{rate:.2f} patents/sec - ETA: {eta/60:.1f} minutes")
                else:
                    print(f"Failed to generate embedding for {patent['id']}")
                
                # Small delay to prevent overwhelming the server
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError during processing: {e}")
    finally:
        elapsed = time.time() - start_time
        final_count = processed_count + len(unprocessed)
        print(f"\nProcessing session complete!")
        print(f"Total time: {elapsed/60:.1f} minutes")
        print(f"Final progress: {final_count}/{len(patents)} patents")


if __name__ == "__main__":
    continuous_embedding_generation()
