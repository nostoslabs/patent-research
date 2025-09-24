#!/usr/bin/env python3
"""Check available patent data sources."""

import pandas as pd
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_parquet_files():
    """Check what's in the parquet files."""
    parquet_files = [
        "patent_embeddings_atlas.parquet",
        "patent_atlas_enhanced.parquet",
        "patent_metadata_only.parquet"
    ]

    for file_path in parquet_files:
        try:
            df = pd.read_parquet(file_path)
            logger.info(f"\n{file_path}:")
            logger.info(f"  Shape: {df.shape}")
            logger.info(f"  Columns: {list(df.columns)}")
            if 'id' in df.columns:
                logger.info(f"  Sample IDs: {df['id'].head(3).tolist()}")
            elif 'patent_id' in df.columns:
                logger.info(f"  Sample IDs: {df['patent_id'].head(3).tolist()}")
            if 'abstract' in df.columns:
                logger.info(f"  Has abstracts: Yes ({df['abstract'].notna().sum()}/{len(df)} non-null)")
            if 'publication_number' in df.columns:
                logger.info(f"  Sample pub numbers: {df['publication_number'].head(3).tolist()}")
        except Exception as e:
            logger.warning(f"Error reading {file_path}: {e}")

def check_ground_truth_patents():
    """Check which patents are in ground truth."""
    ground_truth = []
    with open("data/ground_truth/consolidated/ground_truth_10k.jsonl", 'r') as f:
        for line in f:
            if line.strip():
                ground_truth.append(json.loads(line.strip()))

    patent_ids = set()
    for gt_pair in ground_truth:
        patent_ids.add(gt_pair['patent1_id'])
        patent_ids.add(gt_pair['patent2_id'])

    logger.info(f"\nGround truth contains {len(patent_ids)} unique patents")
    logger.info(f"Sample patent IDs: {list(patent_ids)[:5]}")

    return patent_ids

def find_patent_texts(needed_patent_ids):
    """Find where patent texts are stored."""
    # Check parquet files for patents
    parquet_files = [
        "patent_embeddings_atlas.parquet",
        "patent_atlas_enhanced.parquet",
        "patent_metadata_only.parquet"
    ]

    found_patents = {}

    for file_path in parquet_files:
        try:
            df = pd.read_parquet(file_path)

            # Check if this file has the patents we need
            if 'id' in df.columns:
                id_col = 'id'
            elif 'patent_id' in df.columns:
                id_col = 'patent_id'
            else:
                continue

            # Find intersection
            file_patents = set(df[id_col].astype(str))
            intersection = needed_patent_ids & file_patents

            if intersection:
                logger.info(f"\n{file_path} contains {len(intersection)} needed patents")
                if 'abstract' in df.columns:
                    logger.info(f"  Has abstracts: Yes")
                    # Check a few patents
                    for patent_id in list(intersection)[:3]:
                        row = df[df[id_col] == patent_id]
                        if not row.empty and pd.notna(row['abstract'].iloc[0]):
                            abstract = row['abstract'].iloc[0]
                            logger.info(f"  {patent_id}: {abstract[:100]}...")
                    found_patents[file_path] = (df, id_col, intersection)

        except Exception as e:
            logger.warning(f"Error checking {file_path}: {e}")

    return found_patents

def main():
    """Main execution function."""
    logger.info("Checking patent data sources...")

    # Check what's in parquet files
    check_parquet_files()

    # Get needed patent IDs
    needed_patent_ids = check_ground_truth_patents()

    # Find where patent texts are
    found_patents = find_patent_texts(needed_patent_ids)

    if found_patents:
        logger.info(f"\nFound patent texts in {len(found_patents)} files")
        for file_path in found_patents:
            logger.info(f"  {file_path}: {len(found_patents[file_path][2])} patents")
    else:
        logger.error("No patent texts found!")

if __name__ == "__main__":
    main()