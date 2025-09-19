#!/usr/bin/env python3
"""Manage OpenAI batch processing and consolidate results."""

import json
import glob
from pathlib import Path

def consolidate_all_results():
    """Consolidate all batch results into one file."""
    result_files = glob.glob("openai_batch_results_*.jsonl")

    if not result_files:
        print("No result files found")
        return 0

    output_file = "results/openai_embeddings_consolidated.jsonl"
    Path("results").mkdir(exist_ok=True)

    total_embeddings = 0
    all_patent_ids = set()

    print(f"Found {len(result_files)} result files to consolidate")

    with open(output_file, 'w') as outfile:
        for result_file in sorted(result_files):
            print(f"Processing {result_file}...")

            with open(result_file, 'r') as infile:
                for line in infile:
                    if not line.strip():
                        continue

                    try:
                        result = json.loads(line)

                        if result.get('response', {}).get('status_code') == 200:
                            custom_id = result['custom_id']
                            patent_id = custom_id.replace('embedding-', '')

                            # Avoid duplicates
                            if patent_id in all_patent_ids:
                                continue

                            all_patent_ids.add(patent_id)

                            embedding_data = result['response']['body']['data'][0]
                            embedding = embedding_data['embedding']

                            embedding_entry = {
                                'id': patent_id,
                                'embedding': embedding,
                                'model': 'text-embedding-3-small'
                            }

                            outfile.write(json.dumps(embedding_entry) + '\\n')
                            total_embeddings += 1

                    except (json.JSONDecodeError, KeyError) as e:
                        continue

    print(f"Consolidated {total_embeddings} unique embeddings to {output_file}")
    return total_embeddings

def submit_remaining_chunks():
    """Submit all remaining chunks."""
    import subprocess

    chunk_files = sorted(glob.glob("openai_batch_chunk_*.jsonl"))

    # Skip chunks 0 and 1 (already processed)
    remaining_chunks = chunk_files[2:]  # Start from chunk_02

    print(f"Submitting {len(remaining_chunks)} remaining chunks...")

    for chunk_file in remaining_chunks:
        print(f"Submitting {chunk_file}...")
        try:
            result = subprocess.run(
                ["uv", "run", "python", "code/submit_openai_batch.py", "submit", chunk_file],
                capture_output=True, text=True, timeout=60
            )

            if result.returncode == 0:
                print(f"✅ {chunk_file} submitted successfully")
            else:
                print(f"❌ {chunk_file} failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            print(f"⏰ {chunk_file} submission timed out")
        except Exception as e:
            print(f"❌ {chunk_file} error: {e}")

if __name__ == "__main__":
    print("=== CONSOLIDATING EXISTING RESULTS ===")
    total = consolidate_all_results()

    print("\\n=== SUBMITTING REMAINING CHUNKS ===")
    submit_remaining_chunks()

    print(f"\\n=== SUMMARY ===")
    print(f"Embeddings consolidated: {total}")
    print("Remaining chunks submitted for processing")