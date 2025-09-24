"""Generate OpenAI embeddings using Batch API for ground truth patents."""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import uuid


def load_patent_abstracts(raw_data_path: str) -> Dict[str, str]:
    """Load patent abstracts from raw data files."""
    abstracts = {}

    # Load from all raw patent files
    raw_dir = Path(raw_data_path)
    for jsonl_file in raw_dir.glob("*.jsonl"):
        print(f"Loading abstracts from {jsonl_file.name}...")
        with open(jsonl_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    patent = json.loads(line.strip())
                    patent_id = patent.get('id') or patent.get('patent_id')
                    abstract = patent.get('abstract', '')

                    if patent_id and abstract:
                        abstracts[patent_id] = abstract

                except json.JSONDecodeError:
                    print(f"Error parsing line {line_num} in {jsonl_file.name}")
                    continue

    print(f"Loaded {len(abstracts)} patent abstracts")
    return abstracts


def create_batch_requests(patent_ids: List[str], abstracts: Dict[str, str], output_file: str):
    """Create OpenAI Batch API requests for embeddings."""

    batch_requests = []
    missing_patents = []

    for i, patent_id in enumerate(patent_ids):
        if patent_id not in abstracts:
            missing_patents.append(patent_id)
            continue

        # Create batch request
        request = {
            "custom_id": f"embedding-{patent_id}",
            "method": "POST",
            "url": "/v1/embeddings",
            "body": {
                "model": "text-embedding-3-small",
                "input": abstracts[patent_id],
                "encoding_format": "float"
            }
        }
        batch_requests.append(request)

        if (i + 1) % 1000 == 0:
            print(f"Created {i + 1} batch requests...")

    # Save batch requests
    with open(output_file, 'w') as f:
        for request in batch_requests:
            f.write(json.dumps(request) + '\n')

    print(f"Created {len(batch_requests)} batch requests")
    print(f"Saved to {output_file}")

    if missing_patents:
        print(f"Warning: {len(missing_patents)} patents not found in abstracts:")
        for patent_id in missing_patents[:10]:  # Show first 10
            print(f"  - {patent_id}")
        if len(missing_patents) > 10:
            print(f"  ... and {len(missing_patents) - 10} more")

    return len(batch_requests), missing_patents


def estimate_costs(num_requests: int, avg_tokens: int = 500):
    """Estimate OpenAI Batch API costs."""
    # text-embedding-3-small pricing: $0.02 / 1M tokens
    # Batch API discount: 50% off
    total_tokens = num_requests * avg_tokens
    cost_regular = (total_tokens / 1_000_000) * 0.02
    cost_batch = cost_regular * 0.5  # 50% discount

    print(f"\nCost Estimation:")
    print(f"  Requests: {num_requests:,}")
    print(f"  Estimated tokens: {total_tokens:,} ({avg_tokens} avg per patent)")
    print(f"  Regular API cost: ${cost_regular:.2f}")
    print(f"  Batch API cost (50% off): ${cost_batch:.2f}")

    return cost_batch


def main():
    if len(sys.argv) != 2:
        print("Usage: python openai_batch_embeddings.py <patent_ids_file>")
        sys.exit(1)

    patent_ids_file = sys.argv[1]

    # Load patent IDs that need embeddings
    with open(patent_ids_file, 'r') as f:
        patent_ids = [line.strip() for line in f if line.strip()]

    print(f"Need embeddings for {len(patent_ids)} patents")

    # Load patent abstracts
    abstracts = load_patent_abstracts("data/raw")

    # Create batch requests
    output_file = "openai_batch_requests.jsonl"
    num_requests, missing = create_batch_requests(patent_ids, abstracts, output_file)

    # Estimate costs
    estimated_cost = estimate_costs(num_requests)

    # Instructions for submitting batch
    print(f"\nNext steps:")
    print(f"1. Upload batch file: openai api batches create --input-file {output_file}")
    print(f"2. Monitor batch: openai api batches list")
    print(f"3. Download results when complete")
    print(f"4. Process results with: python process_openai_batch_results.py")

    return output_file


if __name__ == "__main__":
    main()