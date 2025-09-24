#!/usr/bin/env python3
"""Generate bge-m3 embeddings for the 8,494 fair comparison pairs."""

import json
import numpy as np
from pathlib import Path
import logging
from ollama import Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_ground_truth():
    """Load ground truth data."""
    ground_truth = []
    with open("data/ground_truth/consolidated/ground_truth_10k.jsonl", 'r') as f:
        for line in f:
            if line.strip():
                ground_truth.append(json.loads(line.strip()))
    return ground_truth

def load_existing_embeddings():
    """Load existing embeddings for OpenAI and nomic."""
    # Load OpenAI embeddings
    openai_embeddings = {}
    with open("results/openai_embeddings_ground_truth.jsonl", 'r') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line.strip())
            openai_embeddings[data['id']] = np.array(data['embedding'])

    # Load nomic embeddings
    nomic_files = [
        "data/embeddings/by_model/nomic-embed-text/production_100k_remaining_nomic-embed-text_20250912_205647_nomic-embed-text_embeddings.jsonl",
        "data/embeddings/by_model/nomic-embed-text/production_10k_all_nomic-embed-text_20250912_150410_nomic-embed-text_embeddings.jsonl",
        "data/embeddings/by_model/nomic-embed-text/diverse_10k_full_nomic-embed-text_20250912_070401_nomic-embed-text_embeddings.jsonl",
        "data/embeddings/by_model/nomic-embed-text/production_100k_top2_nomic-embed-text_20250912_150417_nomic-embed-text_embeddings.jsonl",
        "data/embeddings/by_model/nomic-embed-text/original_500_nomic-embed-text_20250912_070406_nomic-embed-text_embeddings.jsonl",
        "data/embeddings/by_model/nomic-embed-text/production_top2_nomic-embed-text_20250912_070417_nomic-embed-text_embeddings.jsonl",
        "data/embeddings/by_model/nomic-embed-text/validation_nomic-embed-text_20250912_065708_nomic-embed-text_embeddings.jsonl"
    ]

    nomic_embeddings = {}
    for file_path in nomic_files:
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line.strip())
                    patent_id = entry.get('id') or entry.get('patent_id')
                    embedding = entry.get('embedding')

                    # Handle nested structure
                    if not embedding and 'models' in entry:
                        models_data = entry.get('models', {})
                        if 'nomic-embed-text' in models_data:
                            embeddings_data = models_data['nomic-embed-text'].get('embeddings', {})
                            if 'original' in embeddings_data:
                                embedding = embeddings_data['original'].get('embedding')

                    if patent_id and embedding:
                        nomic_embeddings[patent_id] = np.array(embedding)
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")
            continue

    # Load existing bge-m3 embeddings
    bge_files = [
        "data/embeddings/by_model/bge-m3/production_100k_remaining_bge-m3_20250912_195813_bge-m3_embeddings.jsonl",
        "data/embeddings/by_model/bge-m3/production_10k_all_bge-m3_20250912_142636_bge-m3_embeddings.jsonl",
        "data/embeddings/by_model/bge-m3/diverse_10k_full_bge-m3_20250912_055529_bge-m3_embeddings.jsonl",
        "data/embeddings/by_model/bge-m3/production_100k_top2_bge-m3_20250912_142643_bge-m3_embeddings.jsonl",
        "data/embeddings/by_model/bge-m3/original_500_bge-m3_20250912_055534_bge-m3_embeddings.jsonl",
        "data/embeddings/by_model/bge-m3/production_top2_bge-m3_20250912_055549_bge-m3_embeddings.jsonl",
        "data/embeddings/by_model/bge-m3/validation_bge-m3_20250912_055016_bge-m3_embeddings.jsonl"
    ]

    bge_embeddings = {}
    for file_path in bge_files:
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line.strip())
                    patent_id = entry.get('id') or entry.get('patent_id')
                    embedding = entry.get('embedding')

                    # Handle nested structure
                    if not embedding and 'models' in entry:
                        models_data = entry.get('models', {})
                        if 'bge-m3' in models_data:
                            embeddings_data = models_data['bge-m3'].get('embeddings', {})
                            if 'original' in embeddings_data:
                                embedding = embeddings_data['original'].get('embedding')

                    if patent_id and embedding:
                        bge_embeddings[patent_id] = np.array(embedding)
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")
            continue

    return openai_embeddings, nomic_embeddings, bge_embeddings

def load_patent_texts():
    """Load patent texts for embedding generation."""
    patents = {}

    # Try multiple possible locations for patent data
    patent_files = [
        "data/raw/production_100k_remaining.jsonl",
        "data/raw/production_10k_all.jsonl",
        "data/raw/diverse_10k_full.jsonl",
        "data/raw/production_100k_top2.jsonl",
        "data/raw/original_500.jsonl",
        "data/raw/production_top2.jsonl",
        "data/raw/validation.jsonl"
    ]

    for file_path in patent_files:
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    patent = json.loads(line.strip())
                    patent_id = patent.get('id') or patent.get('patent_id')
                    if patent_id:
                        patents[patent_id] = patent
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")
            continue

    return patents

def find_missing_patents():
    """Find patents that need bge-m3 embeddings for fair comparison."""
    logger.info("Loading data...")
    ground_truth = load_ground_truth()
    openai_embeddings, nomic_embeddings, bge_embeddings = load_existing_embeddings()

    logger.info(f"Loaded {len(openai_embeddings)} OpenAI embeddings")
    logger.info(f"Loaded {len(nomic_embeddings)} nomic embeddings")
    logger.info(f"Loaded {len(bge_embeddings)} bge-m3 embeddings")

    # Find the 8,494 pairs that have both OpenAI and nomic embeddings
    fair_comparison_patents = set()

    for gt_pair in ground_truth:
        patent1_id = gt_pair['patent1_id']
        patent2_id = gt_pair['patent2_id']

        # Only include if both OpenAI and nomic have embeddings for both patents
        if (patent1_id in openai_embeddings and patent2_id in openai_embeddings and
            patent1_id in nomic_embeddings and patent2_id in nomic_embeddings):
            fair_comparison_patents.add(patent1_id)
            fair_comparison_patents.add(patent2_id)

    logger.info(f"Found {len(fair_comparison_patents)} unique patents in fair comparison set")

    # Find which ones are missing bge-m3 embeddings
    missing_bge = fair_comparison_patents - set(bge_embeddings.keys())
    logger.info(f"Missing bge-m3 embeddings for {len(missing_bge)} patents")

    return list(missing_bge), fair_comparison_patents

def generate_bge_embeddings(patent_ids, patent_texts):
    """Generate bge-m3 embeddings for missing patents."""
    if not patent_ids:
        logger.info("No missing patents to process")
        return {}

    logger.info(f"Generating bge-m3 embeddings for {len(patent_ids)} patents...")

    # Initialize Ollama client
    client = Client()

    # Create output file
    output_file = "results/bge_m3_fair_comparison_embeddings.jsonl"
    Path("results").mkdir(exist_ok=True)

    embeddings = {}

    with open(output_file, 'w') as f:
        for i, patent_id in enumerate(patent_ids):
            if patent_id not in patent_texts:
                logger.warning(f"Patent {patent_id} not found in text data")
                continue

            patent = patent_texts[patent_id]
            text = patent.get('abstract', '')

            if not text:
                logger.warning(f"No abstract found for patent {patent_id}")
                continue

            try:
                # Generate embedding using Ollama
                response = client.embeddings(
                    model='bge-m3',
                    prompt=text
                )

                embedding = response['embedding']
                embeddings[patent_id] = embedding

                # Save to file
                result = {
                    'id': patent_id,
                    'embedding': embedding
                }
                f.write(json.dumps(result) + '\n')
                f.flush()

                if (i + 1) % 100 == 0:
                    logger.info(f"Generated {i + 1}/{len(patent_ids)} embeddings")

            except Exception as e:
                logger.error(f"Error generating embedding for {patent_id}: {e}")
                continue

    logger.info(f"Generated {len(embeddings)} new bge-m3 embeddings")
    logger.info(f"Saved to: {output_file}")

    return embeddings

def main():
    """Main execution function."""
    # Find missing patents
    missing_patents, fair_comparison_patents = find_missing_patents()

    if not missing_patents:
        logger.info("All patents in fair comparison set already have bge-m3 embeddings!")
        return

    # Load patent texts
    logger.info("Loading patent texts...")
    patent_texts = load_patent_texts()
    logger.info(f"Loaded {len(patent_texts)} patent texts")

    # Generate missing embeddings
    new_embeddings = generate_bge_embeddings(missing_patents, patent_texts)

    logger.info(f"Successfully generated {len(new_embeddings)} new bge-m3 embeddings")
    logger.info("Ready for three-way fair comparison!")

if __name__ == "__main__":
    main()