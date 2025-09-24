#!/usr/bin/env python3
"""
Model Intersection Analysis

Analyzes the intersection of patents across all three embedding models:
- OpenAI text-embedding-3-small
- nomic-embed-text
- bge-m3

Determines how many patents have embeddings from all models, pairs of models, etc.
"""

import json
import logging
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, Set, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_model_intersections(master_embeddings_file: str) -> Dict:
    """
    Analyze patent intersections across embedding models.

    Args:
        master_embeddings_file: Path to master_patent_embeddings.jsonl

    Returns:
        Dictionary with intersection analysis results
    """

    logger.info("Starting model intersection analysis...")

    # Track which models each patent has embeddings for
    patent_models: Dict[str, Set[str]] = defaultdict(set)
    model_counts = Counter()
    total_patents = 0

    # Read master embeddings file
    with open(master_embeddings_file, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line.strip())
                patent_id = data.get('patent_id', '')
                embeddings = data.get('embeddings', {})

                if patent_id:
                    total_patents += 1

                    # Track which models this patent has embeddings for
                    for model_name in embeddings.keys():
                        patent_models[patent_id].add(model_name)
                        model_counts[model_name] += 1

    logger.info(f"Processed {total_patents} patents")

    # Analyze intersections
    model_names = ['openai_text-embedding-3-small', 'nomic-embed-text', 'bge-m3']

    # Count patents by number of models they have
    patents_by_model_count = Counter()
    for patent_id, models in patent_models.items():
        patents_by_model_count[len(models)] += 1

    # Find specific intersections
    intersections = {}

    # All three models
    all_three = set()
    for patent_id, models in patent_models.items():
        if len(models) == 3:
            all_three.add(patent_id)
    intersections['all_three_models'] = len(all_three)

    # Pairwise intersections
    model_pairs = [
        ('openai_text-embedding-3-small', 'nomic-embed-text'),
        ('openai_text-embedding-3-small', 'bge-m3'),
        ('nomic-embed-text', 'bge-m3')
    ]

    for model1, model2 in model_pairs:
        pair_count = 0
        for patent_id, models in patent_models.items():
            if model1 in models and model2 in models:
                pair_count += 1

        pair_name = f"{model1.split('_')[0]}_{model2.split('-')[0]}"
        intersections[f'pair_{pair_name}'] = pair_count

    # Individual model counts (should match our known values)
    individual_counts = {}
    for model in model_names:
        individual_counts[model] = model_counts[model]

    # Patents unique to each model (only have that one model)
    unique_to_model = {}
    for model in model_names:
        unique_count = 0
        for patent_id, models in patent_models.items():
            if len(models) == 1 and model in models:
                unique_count += 1
        unique_to_model[model] = unique_count

    # Create comprehensive results
    results = {
        'total_patents_processed': total_patents,
        'individual_model_counts': individual_counts,
        'intersection_analysis': {
            'all_three_models': intersections['all_three_models'],
            'pairwise_intersections': {
                'openai_and_nomic': intersections.get('pair_openai_nomic', 0),
                'openai_and_bge': intersections.get('pair_openai_bge', 0),
                'nomic_and_bge': intersections.get('pair_nomic_bge', 0)
            }
        },
        'patents_by_model_count': dict(patents_by_model_count),
        'unique_to_single_model': unique_to_model,
        'coverage_analysis': {
            'percent_with_all_three': round(intersections['all_three_models'] / total_patents * 100, 2),
            'percent_with_two_or_more': round((total_patents - patents_by_model_count[1]) / total_patents * 100, 2),
            'percent_with_only_one': round(patents_by_model_count[1] / total_patents * 100, 2)
        }
    }

    return results, all_three


def print_analysis_results(results: Dict, all_three_patents: Set[str]):
    """Print formatted analysis results"""

    print("\n" + "="*80)
    print("MODEL INTERSECTION ANALYSIS")
    print("="*80)

    print(f"\n📊 TOTAL PATENTS: {results['total_patents_processed']:,}")

    print(f"\n🎯 INDIVIDUAL MODEL COUNTS:")
    for model, count in results['individual_model_counts'].items():
        print(f"  {model}: {count:,}")

    print(f"\n🔗 INTERSECTION ANALYSIS:")
    print(f"  Patents with ALL THREE models: {results['intersection_analysis']['all_three_models']:,}")

    pairwise = results['intersection_analysis']['pairwise_intersections']
    print(f"\n  Pairwise Intersections:")
    print(f"    OpenAI + nomic: {pairwise['openai_and_nomic']:,}")
    print(f"    OpenAI + bge-m3: {pairwise['openai_and_bge']:,}")
    print(f"    nomic + bge-m3: {pairwise['nomic_and_bge']:,}")

    print(f"\n📈 COVERAGE BREAKDOWN:")
    by_count = results['patents_by_model_count']
    for model_count in sorted(by_count.keys(), reverse=True):
        patent_count = by_count[model_count]
        percentage = round(patent_count / results['total_patents_processed'] * 100, 2)
        print(f"  {model_count} model(s): {patent_count:,} patents ({percentage}%)")

    print(f"\n🎯 COVERAGE PERCENTAGES:")
    coverage = results['coverage_analysis']
    print(f"  With all three models: {coverage['percent_with_all_three']}%")
    print(f"  With two or more models: {coverage['percent_with_two_or_more']}%")
    print(f"  With only one model: {coverage['percent_with_only_one']}%")

    print(f"\n🔍 UNIQUE TO SINGLE MODEL:")
    for model, count in results['unique_to_single_model'].items():
        print(f"  Only {model}: {count:,}")

    if len(all_three_patents) > 0:
        print(f"\n📋 SAMPLE PATENTS WITH ALL THREE MODELS (first 10):")
        for i, patent_id in enumerate(sorted(list(all_three_patents))[:10]):
            print(f"  {i+1}. {patent_id}")
        if len(all_three_patents) > 10:
            print(f"  ... and {len(all_three_patents) - 10:,} more")

    print("\n" + "="*80)


def main():
    """Main execution function"""

    master_embeddings_file = Path("data_v2/master_patent_embeddings.jsonl")

    if not master_embeddings_file.exists():
        print(f"❌ Master embeddings file not found: {master_embeddings_file}")
        print("Please run this script from the project root directory.")
        return

    try:
        # Run analysis
        results, all_three_patents = analyze_model_intersections(str(master_embeddings_file))

        # Print results
        print_analysis_results(results, all_three_patents)

        # Save results
        output_file = Path("analysis/model_intersection_results.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert set to list for JSON serialization
        results['all_three_patent_ids'] = sorted(list(all_three_patents))

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {output_file}")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()