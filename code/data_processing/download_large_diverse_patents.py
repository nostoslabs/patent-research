"""Download and sample patent abstracts with diverse categories from BigPatent dataset."""

import json
import random
import time
from pathlib import Path
from typing import Any
from collections import defaultdict, Counter

from datasets import load_dataset


def explore_dataset_categories(max_explore: int = 50000) -> dict[str, int]:
    """Explore available categories in the BigPatent dataset."""
    print("Exploring BigPatent dataset categories...")
    
    dataset = load_dataset("mteb/big-patent", split="test", streaming=True)
    category_counts = Counter()
    
    for i, patent in enumerate(dataset):
        if i >= max_explore:
            break
        
        if i % 10000 == 0:
            print(f"Explored {i} patents so far...")
        
        classification = str(patent.get("labels", "unknown"))
        category_counts[classification] += 1
    
    print(f"\nFound {len(category_counts)} categories in {i+1} patents:")
    for category, count in category_counts.most_common():
        percentage = (count / (i+1)) * 100
        print(f"  Category {category}: {count} patents ({percentage:.1f}%)")
    
    return dict(category_counts)


def download_diverse_patents(
    target_size: int = 100000,
    categories_per_batch: int = 8,
    output_file: str = "patent_abstracts_diverse.jsonl",
    seed: int = 42,
    min_abstract_length: int = 100,
    max_patents_to_scan: int = 200000
) -> list[dict[str, Any]]:
    """
    Download patents with diverse category representation.
    
    Args:
        target_size: Target number of patents to collect
        categories_per_batch: Number of different categories to aim for
        output_file: Output JSONL file name  
        seed: Random seed for reproducibility
        min_abstract_length: Minimum abstract length to include
        max_patents_to_scan: Maximum patents to scan before stopping
        
    Returns:
        List of diverse patent data
    """
    print(f"Collecting {target_size} diverse patents from BigPatent dataset...")
    
    # Load dataset
    dataset = load_dataset("mteb/big-patent", split="test", streaming=True)
    
    # Track patents by category for balanced sampling
    patents_by_category = defaultdict(list)
    category_counts = Counter()
    total_processed = 0
    patents_with_abstracts = 0
    
    random.seed(seed)
    
    print("Scanning patents and extracting abstracts...")
    
    for i, patent in enumerate(dataset):
        if total_processed >= max_patents_to_scan:
            break
            
        if i % 5000 == 0:
            print(f"Processed {i} patents, found {patents_with_abstracts} with valid abstracts")
            print(f"Categories found: {len(patents_by_category)}")
            
        total_processed += 1
        
        # Extract full text and classification
        full_text = patent.get("sentences", "")
        classification = str(patent.get("labels", "unknown"))
        
        # Extract abstract from patent text
        abstract = extract_abstract_from_text(full_text)
        
        if abstract and len(abstract) >= min_abstract_length:
            patent_data = {
                "id": f"patent_{i}",
                "abstract": abstract,
                "full_text": full_text[:2000] + "..." if len(full_text) > 2000 else full_text,
                "classification": classification,
                "abstract_length": len(abstract),
                "source_index": i
            }
            
            patents_by_category[classification].append(patent_data)
            category_counts[classification] += 1
            patents_with_abstracts += 1
            
        # Early stopping if we have enough diverse patents
        if (patents_with_abstracts >= target_size * 2 and 
            len(patents_by_category) >= categories_per_batch):
            print(f"Collected sufficient diversity, stopping early at {i} patents")
            break
    
    print(f"\nCollection complete!")
    print(f"Total patents processed: {total_processed}")
    print(f"Patents with valid abstracts: {patents_with_abstracts}")
    print(f"Categories found: {len(patents_by_category)}")
    
    # Show category distribution
    print("\nCategory distribution:")
    for category, count in category_counts.most_common():
        percentage = (count / patents_with_abstracts) * 100
        print(f"  Category {category}: {count} patents ({percentage:.1f}%)")
    
    # Sample balanced representation from each category
    print(f"\nSampling {target_size} patents with balanced category representation...")
    
    sampled_patents = []
    categories = list(patents_by_category.keys())
    patents_per_category = target_size // len(categories)
    remainder = target_size % len(categories)
    
    print(f"Target: ~{patents_per_category} patents per category")
    
    for i, category in enumerate(categories):
        available_patents = patents_by_category[category]
        
        # Add extra patent to some categories to handle remainder
        category_target = patents_per_category + (1 if i < remainder else 0)
        category_sample_size = min(category_target, len(available_patents))
        
        category_sample = random.sample(available_patents, category_sample_size)
        sampled_patents.extend(category_sample)
        
        print(f"  Category {category}: sampled {len(category_sample)}/{len(available_patents)} patents")
    
    # Shuffle final list
    random.shuffle(sampled_patents)
    
    print(f"\nFinal sample: {len(sampled_patents)} patents")
    
    # Save to JSONL file
    output_path = Path(output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        for patent in sampled_patents:
            json.dump(patent, f, ensure_ascii=False)
            f.write("\n")
    
    print(f"Saved diverse patent dataset to {output_path}")
    
    # Print final statistics
    final_category_counts = Counter(p["classification"] for p in sampled_patents)
    print(f"\nFinal category distribution:")
    for category, count in final_category_counts.most_common():
        percentage = (count / len(sampled_patents)) * 100
        print(f"  Category {category}: {count} patents ({percentage:.1f}%)")
    
    return sampled_patents


def extract_abstract_from_text(full_text: str) -> str:
    """Extract abstract from patent full text."""
    if not full_text:
        return ""
    
    lines = full_text.split('\n')
    abstract = ""
    
    # Look for abstract section
    for j, line in enumerate(lines[:100]):  # Check first 100 lines
        line = line.strip()
        if any(keyword in line.upper() for keyword in [
            'ABSTRACT', 'SUMMARY', 'FIELD OF THE INVENTION', 'BACKGROUND',
            'TECHNICAL FIELD', 'BRIEF SUMMARY'
        ]):
            # Found abstract section, get next meaningful lines
            abstract_lines = []
            for k in range(j + 1, min(j + 30, len(lines))):
                next_line = lines[k].strip()
                if (next_line and 
                    not next_line.startswith('[') and 
                    len(next_line) > 30 and
                    not next_line.upper().startswith('BRIEF DESCRIPTION') and
                    not next_line.upper().startswith('DETAILED DESCRIPTION')):
                    abstract_lines.append(next_line)
                elif len(abstract_lines) >= 3:  # Stop if we have enough
                    break
            
            abstract = ' '.join(abstract_lines)
            if len(abstract) > 100:  # Only use if substantial
                break
    
    # If no abstract section found, look for first substantial paragraphs
    if not abstract or len(abstract) < 100:
        substantial_lines = []
        for line in lines[:50]:
            line = line.strip()
            if (len(line) > 80 and 
                not line.startswith('[') and 
                '.' in line and
                not line.upper().startswith('BRIEF DESCRIPTION') and
                not line.upper().startswith('DETAILED DESCRIPTION')):
                substantial_lines.append(line)
                if len(' '.join(substantial_lines)) > 500:  # Got enough
                    break
        
        if substantial_lines:
            abstract = ' '.join(substantial_lines)
    
    return abstract


def create_continuous_embedding_job(
    patent_file: str,
    batch_size: int = 50,
    save_frequency: int = 10
) -> None:
    """Create a background job script for continuous embedding generation."""
    
    script_content = f'''#!/usr/bin/env python3
"""Continuous embedding generation for large patent dataset."""

import json
import time
import signal
import sys
from pathlib import Path

from generate_embeddings import generate_embedding, test_ollama_connection


def signal_handler(sig, frame):
    """Handle graceful shutdown on Ctrl+C."""
    print("\\nReceived interrupt signal. Gracefully shutting down...")
    print("Current progress has been saved.")
    sys.exit(0)


def continuous_embedding_generation(
    input_file: str = "{patent_file}",
    output_file: str = "patent_abstracts_with_embeddings_large.jsonl",
    model: str = "embeddinggemma",
    batch_size: int = {batch_size},
    save_frequency: int = {save_frequency},
    resume: bool = True
) -> None:
    """Generate embeddings continuously with progress saving."""
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("Starting continuous embedding generation...")
    print(f"Input file: {{input_file}}")
    print(f"Output file: {{output_file}}")
    print(f"Model: {{model}}")
    print(f"Batch size: {{batch_size}}")
    print(f"Save frequency: every {{save_frequency}} records")
    print("-" * 60)
    
    # Test connection
    if not test_ollama_connection(model):
        print(f"Error: Cannot connect to {{model}} model")
        return
    
    # Load input patents
    print("Loading patents...")
    patents = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            patents.append(json.loads(line.strip()))
    
    print(f"Loaded {{len(patents)}} patents")
    
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
        print(f"Found {{len(processed_ids)}} already processed patents")
    
    # Filter unprocessed patents
    unprocessed = [p for p in patents if p['id'] not in processed_ids]
    print(f"{{len(unprocessed)}} patents remaining to process")
    
    if not unprocessed:
        print("All patents already processed!")
        return
    
    # Process in batches
    start_time = time.time()
    processed_count = len(processed_ids)
    
    try:
        with open(output_path, 'a', encoding='utf-8') as f:
            for i, patent in enumerate(unprocessed, 1):
                print(f"Processing record {{processed_count + i}}/{{len(patents)}}: {{patent['id']}}")
                
                # Generate embedding
                embedding = generate_embedding(patent['abstract'], model=model)
                
                if embedding is not None:
                    # Create record with embedding
                    record = patent.copy()
                    record['embeddings'] = {{
                        'embeddinggemma': {{
                            'model': model,
                            'embedding': embedding.tolist(),
                            'text_source': 'abstract',
                            'embedding_dim': len(embedding),
                            'generated_at': time.strftime("%Y-%m-%d %H:%M:%S")
                        }}
                    }}
                    
                    # Write record
                    json.dump(record, f, ensure_ascii=False)
                    f.write('\\n')
                    
                    # Save progress periodically
                    if i % save_frequency == 0:
                        f.flush()  # Ensure data is written to disk
                        
                        elapsed = time.time() - start_time
                        rate = i / elapsed
                        remaining = len(unprocessed) - i
                        eta = remaining / rate if rate > 0 else 0
                        
                        print(f"Progress saved ({{processed_count + i}}/{{len(patents)}}) - "
                              f"{{rate:.2f}} patents/sec - ETA: {{eta/60:.1f}} minutes")
                else:
                    print(f"Failed to generate embedding for {{patent['id']}}")
                
                # Small delay to prevent overwhelming the server
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("\\nInterrupted by user")
    except Exception as e:
        print(f"\\nError during processing: {{e}}")
    finally:
        elapsed = time.time() - start_time
        final_count = processed_count + len(unprocessed)
        print(f"\\nProcessing session complete!")
        print(f"Total time: {{elapsed/60:.1f}} minutes")
        print(f"Final progress: {{final_count}}/{{len(patents)}} patents")


if __name__ == "__main__":
    continuous_embedding_generation()
'''
    
    # Write the script
    script_path = Path("continuous_embedding_job.py")
    with open(script_path, "w") as f:
        f.write(script_content)
    
    # Make executable
    import stat
    script_path.chmod(stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
    
    print(f"Created continuous embedding job script: {script_path}")
    print("To run in background: nohup python continuous_embedding_job.py > embedding_log.txt 2>&1 &")


def main() -> None:
    """Main function with options for different operations."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "explore":
        # Explore categories only
        explore_dataset_categories()
    elif len(sys.argv) > 1 and sys.argv[1] == "100k":
        # Download 100k diverse patents
        print("Downloading 100,000 diverse patents...")
        patents = download_diverse_patents(
            target_size=100000,
            output_file="patent_abstracts_100k_diverse.jsonl"
        )
        
        # Create continuous embedding job
        create_continuous_embedding_job("patent_abstracts_100k_diverse.jsonl")
        
    elif len(sys.argv) > 1 and sys.argv[1] == "10k":
        # Download 10k diverse patents for testing
        print("Downloading 10,000 diverse patents for testing...")
        patents = download_diverse_patents(
            target_size=10000,
            output_file="patent_abstracts_10k_diverse.jsonl"
        )
        
        # Create continuous embedding job
        create_continuous_embedding_job("patent_abstracts_10k_diverse.jsonl")
        
    else:
        print("Usage:")
        print("  python download_large_diverse_patents.py explore    # Explore available categories")
        print("  python download_large_diverse_patents.py 10k       # Download 10k diverse patents")
        print("  python download_large_diverse_patents.py 100k      # Download 100k diverse patents")


if __name__ == "__main__":
    main()