#!/usr/bin/env python3
"""
Fix Missing Classifications

Enriches the master embeddings file with missing classification data from the original patent_abstracts.json.
Many patents lost their classification metadata during embedding consolidation.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_patent_metadata(patent_abstracts_file: str) -> Dict[str, Dict]:
    """
    Load patent metadata from the original abstracts file.

    Args:
        patent_abstracts_file: Path to patent_abstracts.json

    Returns:
        Dictionary mapping patent_id to metadata
    """
    logger.info(f"Loading patent metadata from {patent_abstracts_file}")

    metadata = {}

    try:
        with open(patent_abstracts_file, 'r') as f:
            patents_list = json.load(f)

        for patent in patents_list:
            patent_id = patent.get('id', '')
            if patent_id:
                metadata[patent_id] = {
                    'classification': patent.get('classification', ''),
                    'abstract': patent.get('abstract', ''),
                    'full_text': patent.get('full_text', '')
                }

        logger.info(f"Loaded metadata for {len(metadata)} patents")

    except Exception as e:
        logger.error(f"Error loading patent metadata: {e}")

    return metadata


def analyze_classification_gaps(master_file: str, metadata: Dict[str, Dict]) -> Dict:
    """
    Analyze classification gaps in the master file.

    Args:
        master_file: Path to master embeddings file
        metadata: Patent metadata dictionary

    Returns:
        Gap analysis statistics
    """
    logger.info("Analyzing classification gaps...")

    stats = {
        'total_patents': 0,
        'missing_classifications': 0,
        'can_be_fixed': 0,
        'permanently_missing': 0,
        'already_have_classification': 0,
        'classification_distribution': {}
    }

    with open(master_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                stats['total_patents'] += 1

                try:
                    data = json.loads(line.strip())
                    patent_id = data.get('patent_id', '')
                    current_classification = data.get('classification', '')

                    if not current_classification:
                        stats['missing_classifications'] += 1

                        # Check if we can fix it
                        if patent_id in metadata and metadata[patent_id]['classification']:
                            stats['can_be_fixed'] += 1
                        else:
                            stats['permanently_missing'] += 1
                    else:
                        stats['already_have_classification'] += 1
                        # Track distribution of existing classifications
                        if current_classification in stats['classification_distribution']:
                            stats['classification_distribution'][current_classification] += 1
                        else:
                            stats['classification_distribution'][current_classification] = 1

                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")

                if line_num % 10000 == 0:
                    logger.info(f"Analyzed {line_num:,} patents...")

    logger.info("Gap analysis complete!")
    return stats


def fix_classifications(master_file: str, metadata: Dict[str, Dict]) -> str:
    """
    Fix missing classifications in the master embeddings file.

    Args:
        master_file: Path to master embeddings file
        metadata: Patent metadata dictionary

    Returns:
        Path to the updated file
    """
    logger.info("Fixing missing classifications...")

    # Create backup
    backup_file = f"{master_file}.backup_classifications_{int(time.time())}"
    Path(master_file).rename(backup_file)
    logger.info(f"Created backup: {backup_file}")

    fixed_count = 0
    updated_text_count = 0
    total_patents = 0

    with open(backup_file, 'r') as infile, open(master_file, 'w') as outfile:
        for line_num, line in enumerate(infile, 1):
            if line.strip():
                total_patents += 1

                try:
                    data = json.loads(line.strip())
                    patent_id = data.get('patent_id', '')

                    # Fix missing classification
                    if patent_id in metadata:
                        source_data = metadata[patent_id]

                        # Add/update classification if missing or better available
                        if not data.get('classification', '') and source_data['classification']:
                            data['classification'] = source_data['classification']
                            fixed_count += 1

                        # Update text if missing or better available
                        current_abstract = data.get('abstract', '')
                        source_abstract = source_data['abstract']

                        if not current_abstract and source_abstract:
                            data['abstract'] = source_abstract
                            updated_text_count += 1

                        # Update full_text if not present
                        if not data.get('full_text', '') and source_data['full_text']:
                            data['full_text'] = source_data['full_text']

                        # Update has_text metadata
                        data['metadata']['has_text'] = bool(data.get('abstract', '') or data.get('full_text', ''))

                    outfile.write(json.dumps(data) + '\n')

                except Exception as e:
                    logger.warning(f"Error fixing line {line_num}: {e}")
                    outfile.write(line)  # Write original line if error

                if line_num % 10000 == 0:
                    logger.info(f"Processed {line_num:,} patents...")

    logger.info(f"Fixed {fixed_count} missing classifications and updated {updated_text_count} abstracts")
    logger.info(f"Total patents processed: {total_patents}")

    return master_file


def verify_fixes(master_file: str) -> Dict:
    """
    Verify the classification fixes.

    Args:
        master_file: Path to updated master file

    Returns:
        Verification statistics
    """
    logger.info("Verifying classification fixes...")

    verification = {
        'total_patents': 0,
        'with_classifications': 0,
        'without_classifications': 0,
        'classification_distribution': {},
        'sample_classifications': []
    }

    with open(master_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                verification['total_patents'] += 1

                try:
                    data = json.loads(line.strip())
                    classification = data.get('classification', '')

                    if classification:
                        verification['with_classifications'] += 1

                        # Track distribution
                        if classification in verification['classification_distribution']:
                            verification['classification_distribution'][classification] += 1
                        else:
                            verification['classification_distribution'][classification] = 1

                        # Sample for display
                        if len(verification['sample_classifications']) < 5:
                            verification['sample_classifications'].append({
                                'patent_id': data.get('patent_id', ''),
                                'classification': classification,
                                'has_abstract': bool(data.get('abstract', '')),
                                'model_count': len(data.get('embeddings', {}))
                            })
                    else:
                        verification['without_classifications'] += 1

                except Exception as e:
                    logger.warning(f"Error verifying line {line_num}: {e}")

                if line_num % 10000 == 0:
                    logger.info(f"Verified {line_num:,} patents...")

    logger.info("Verification complete!")
    return verification


def print_results(before_stats: Dict, after_stats: Dict):
    """Print before/after comparison"""

    print("\n" + "="*80)
    print("CLASSIFICATION FIX RESULTS")
    print("="*80)

    print(f"\n📊 BEFORE FIXES:")
    print(f"  Total patents: {before_stats['total_patents']:,}")
    print(f"  Missing classifications: {before_stats['missing_classifications']:,} ({(before_stats['missing_classifications']/before_stats['total_patents']*100):>5.1f}%)")
    print(f"  Already had classifications: {before_stats['already_have_classification']:,}")
    print(f"  Could be fixed: {before_stats['can_be_fixed']:,}")
    print(f"  Permanently missing: {before_stats['permanently_missing']:,}")

    print(f"\n✅ AFTER FIXES:")
    print(f"  Total patents: {after_stats['total_patents']:,}")
    print(f"  With classifications: {after_stats['with_classifications']:,} ({(after_stats['with_classifications']/after_stats['total_patents']*100):>5.1f}%)")
    print(f"  Still missing: {after_stats['without_classifications']:,} ({(after_stats['without_classifications']/after_stats['total_patents']*100):>5.1f}%)")

    print(f"\n🎯 IMPROVEMENT:")
    improvement = after_stats['with_classifications'] - before_stats['already_have_classification']
    print(f"  Classifications added: {improvement:,}")
    improvement_pct = (after_stats['with_classifications'] / after_stats['total_patents']) - (before_stats['already_have_classification'] / before_stats['total_patents'])
    print(f"  Coverage improvement: {improvement_pct*100:+.1f} percentage points")

    print(f"\n📈 CLASSIFICATION DISTRIBUTION:")
    for classification, count in sorted(after_stats['classification_distribution'].items()):
        percentage = (count / after_stats['total_patents']) * 100
        print(f"  {classification:>3}: {count:>7,} ({percentage:>5.1f}%)")

    print(f"\n📋 SAMPLE FIXED PATENTS:")
    for sample in after_stats['sample_classifications']:
        print(f"  {sample['patent_id']}: class={sample['classification']}, abstract={sample['has_abstract']}, models={sample['model_count']}")

    print("="*80)


def main():
    """Main execution"""
    master_file = "data_v2/master_patent_embeddings.jsonl"
    patent_abstracts_file = "data/patent_abstracts.json"

    # Check files exist
    if not Path(master_file).exists():
        print(f"❌ Master embeddings file not found: {master_file}")
        return

    if not Path(patent_abstracts_file).exists():
        print(f"❌ Patent abstracts file not found: {patent_abstracts_file}")
        return

    try:
        print("🚀 Starting classification fix process...")

        # Load patent metadata
        metadata = load_patent_metadata(patent_abstracts_file)

        if not metadata:
            print("❌ No patent metadata loaded")
            return

        # Analyze gaps before fixing
        before_stats = analyze_classification_gaps(master_file, metadata)

        # Fix classifications
        fix_classifications(master_file, metadata)

        # Verify fixes
        after_stats = verify_fixes(master_file)

        # Print results
        print_results(before_stats, after_stats)

        print(f"\n🎯 NEXT STEPS:")
        print("1. Regenerate Atlas data with fixed classifications")
        print("2. Classification colors should now be meaningful")
        print("3. Restart Atlas to see improved visualization")

    except Exception as e:
        logger.error(f"Classification fix failed: {e}")
        raise


if __name__ == "__main__":
    main()