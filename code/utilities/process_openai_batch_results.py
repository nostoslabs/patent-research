"""Process OpenAI batch embedding results and create embeddings file."""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any


def process_batch_results(results_file: str, output_file: str = "openai_embeddings_ground_truth.jsonl"):
    """Process OpenAI batch results into our embedding format."""

    embeddings = []
    errors = []

    print(f"Processing results from {results_file}")

    with open(results_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                result = json.loads(line.strip())

                if result.get('response', {}).get('status_code') == 200:
                    # Successful embedding
                    custom_id = result['custom_id']
                    patent_id = custom_id.replace('embedding-', '')

                    response_body = result['response']['body']
                    embedding = response_body['data'][0]['embedding']

                    # Create our format
                    embedding_record = {
                        "id": patent_id,
                        "patent_id": patent_id,
                        "model": "text-embedding-3-small",
                        "embedding": embedding,
                        "embedding_dim": len(embedding),
                        "tokens_used": response_body['usage']['total_tokens'],
                        "success": True
                    }

                    embeddings.append(embedding_record)

                else:
                    # Error case
                    custom_id = result['custom_id']
                    patent_id = custom_id.replace('embedding-', '')
                    error_info = result.get('error', {})

                    errors.append({
                        "patent_id": patent_id,
                        "error": error_info
                    })

                if line_num % 1000 == 0:
                    print(f"Processed {line_num} results...")

            except json.JSONDecodeError:
                print(f"Error parsing line {line_num}")
                continue

    # Save embeddings
    with open(output_file, 'w') as f:
        for embedding in embeddings:
            f.write(json.dumps(embedding) + '\n')

    print(f"Processed {len(embeddings)} successful embeddings")
    print(f"Found {len(errors)} errors")
    print(f"Embeddings saved to {output_file}")

    if errors:
        error_file = output_file.replace('.jsonl', '_errors.json')
        with open(error_file, 'w') as f:
            json.dump(errors, f, indent=2)
        print(f"Errors saved to {error_file}")

    # Calculate stats
    if embeddings:
        total_tokens = sum(e['tokens_used'] for e in embeddings)
        avg_tokens = total_tokens / len(embeddings)
        print(f"Total tokens used: {total_tokens:,}")
        print(f"Average tokens per patent: {avg_tokens:.1f}")

    return output_file


def main():
    if len(sys.argv) < 2:
        print("Usage: python process_openai_batch_results.py <results_file> [output_file]")
        sys.exit(1)

    results_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "openai_embeddings_ground_truth.jsonl"

    if not Path(results_file).exists():
        print(f"Results file not found: {results_file}")
        sys.exit(1)

    process_batch_results(results_file, output_file)


if __name__ == "__main__":
    main()