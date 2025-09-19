#!/usr/bin/env python3
"""
Find Patents Missing bge-m3 Embeddings

Identifies patents that have both OpenAI and nomic embeddings but are missing bge-m3 embeddings.
Prepares the text data for bge-m3 embedding generation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_missing_bge_patents(master_file: str) -> List[Dict]:
    """
    Find patents that have OpenAI + nomic but missing bge-m3 embeddings.

    Args:
        master_file: Path to master embeddings file

    Returns:
        List of patent records missing bge-m3 embeddings
    """
    logger.info("Searching for patents missing bge-m3 embeddings...")

    missing_bge_patents = []
    stats = {
        'total_patents': 0,
        'has_openai_and_nomic': 0,
        'has_all_three': 0,
        'missing_bge': 0,
        'no_text_available': 0
    }

    with open(master_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                stats['total_patents'] += 1

                try:
                    data = json.loads(line.strip())
                    embeddings = data.get('embeddings', {})

                    # Check if has both OpenAI and nomic
                    has_openai = 'openai_text-embedding-3-small' in embeddings
                    has_nomic = 'nomic-embed-text' in embeddings
                    has_bge = 'bge-m3' in embeddings

                    if has_openai and has_nomic:
                        stats['has_openai_and_nomic'] += 1

                        if has_bge:
                            stats['has_all_three'] += 1
                        else:
                            stats['missing_bge'] += 1

                            # Get text content for embedding generation
                            abstract = data.get('abstract', '')
                            full_text = data.get('full_text', '')

                            # Use abstract if available, otherwise truncated full_text
                            text_for_embedding = abstract
                            if not text_for_embedding and full_text:
                                text_for_embedding = full_text[:8000]  # BGE-M3 token limit

                            if text_for_embedding:
                                patent_record = {
                                    'patent_id': data.get('patent_id', ''),
                                    'abstract': abstract,
                                    'full_text': full_text[:500] if full_text else '',  # Truncated for display
                                    'classification': data.get('classification', ''),
                                    'text_for_embedding': text_for_embedding,
                                    'text_length': len(text_for_embedding),
                                    'source': 'abstract' if abstract else 'full_text'
                                }
                                missing_bge_patents.append(patent_record)
                            else:
                                stats['no_text_available'] += 1

                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")
                    continue

                if line_num % 10000 == 0:
                    logger.info(f"Processed {line_num:,} patents...")

    logger.info("Analysis complete!")

    # Print summary
    print("\n" + "="*60)
    print("MISSING BGE-M3 EMBEDDINGS ANALYSIS")
    print("="*60)
    print(f"📊 Total patents analyzed: {stats['total_patents']:,}")
    print(f"🎯 Patents with OpenAI + nomic: {stats['has_openai_and_nomic']:,}")
    print(f"✅ Patents with all three models: {stats['has_all_three']:,}")
    print(f"❌ Patents missing bge-m3: {stats['missing_bge']:,}")
    print(f"📝 Patents with text for embedding: {len(missing_bge_patents):,}")
    print(f"⚠️  Patents without text: {stats['no_text_available']:,}")

    if missing_bge_patents:
        print(f"\n📋 SAMPLE PATENTS MISSING BGE-M3:")
        for i, patent in enumerate(missing_bge_patents[:5]):
            print(f"  {i+1}. {patent['patent_id']}")
            print(f"     Text source: {patent['source']}")
            print(f"     Text length: {patent['text_length']} chars")
            print(f"     Classification: {patent['classification']}")

        if len(missing_bge_patents) > 5:
            print(f"     ... and {len(missing_bge_patents) - 5:,} more")

    print("="*60)

    return missing_bge_patents, stats


def save_patents_for_embedding(patents: List[Dict], output_file: str):
    """
    Save patents to JSONL file for bge-m3 embedding generation.

    Args:
        patents: List of patent records
        output_file: Output file path
    """
    logger.info(f"Saving {len(patents)} patents to {output_file}")

    with open(output_file, 'w') as f:
        for patent in patents:
            # Create record optimized for bge-m3 embedding
            embedding_record = {
                'id': patent['patent_id'],
                'text': patent['text_for_embedding'],
                'metadata': {
                    'classification': patent['classification'],
                    'text_source': patent['source'],
                    'text_length': patent['text_length'],
                    'abstract_available': bool(patent['abstract'])
                }
            }
            f.write(json.dumps(embedding_record) + '\n')

    logger.info(f"Saved embedding input file: {output_file}")


def main():
    """Main execution"""
    master_file = "data_v2/master_patent_embeddings.jsonl"
    output_file = "data_v2/patents_needing_bge_m3.jsonl"

    if not Path(master_file).exists():
        print(f"❌ Master embeddings file not found: {master_file}")
        return

    try:
        # Find missing bge-m3 patents
        missing_patents, stats = find_missing_bge_patents(master_file)

        if not missing_patents:
            print("✅ All patents that have OpenAI + nomic already have bge-m3 embeddings!")
            return

        # Save patents for embedding generation
        save_patents_for_embedding(missing_patents, output_file)

        print(f"\n🎯 NEXT STEPS:")
        print(f"1. Use the file: {output_file}")
        print(f"2. Generate bge-m3 embeddings for {len(missing_patents):,} patents")
        print(f"3. This will give us complete coverage for all {stats['has_openai_and_nomic']:,} patents with OpenAI+nomic")
        print(f"4. Final result: {stats['has_all_three'] + len(missing_patents):,} patents with all three models")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()