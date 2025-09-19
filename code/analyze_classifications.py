#!/usr/bin/env python3
"""
Analyze Patent Classifications

Reviews the classification data in our consolidated dataset to understand:
1. Distribution of classification values
2. Missing/empty classifications
3. Data quality issues
4. Patterns in the classification system
"""

import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_classifications(master_file: str) -> dict:
    """
    Analyze classification distribution in the master embeddings file.

    Args:
        master_file: Path to master embeddings file

    Returns:
        Dictionary with analysis results
    """
    logger.info("Analyzing patent classifications...")

    classification_stats = {
        'total_patents': 0,
        'classification_counts': Counter(),
        'empty_classifications': 0,
        'null_classifications': 0,
        'classification_by_model_count': defaultdict(lambda: Counter()),
        'sample_patents': defaultdict(list)
    }

    with open(master_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                classification_stats['total_patents'] += 1

                try:
                    data = json.loads(line.strip())
                    patent_id = data.get('patent_id', '')
                    classification = data.get('classification', '')
                    embeddings = data.get('embeddings', {})
                    model_count = len(embeddings)

                    # Handle different classification formats
                    if classification is None:
                        classification = 'NULL'
                        classification_stats['null_classifications'] += 1
                    elif classification == '':
                        classification = 'EMPTY'
                        classification_stats['empty_classifications'] += 1
                    else:
                        classification = str(classification).strip()

                    # Count classifications
                    classification_stats['classification_counts'][classification] += 1

                    # Track by model count
                    classification_stats['classification_by_model_count'][model_count][classification] += 1

                    # Sample patents for each classification
                    if len(classification_stats['sample_patents'][classification]) < 5:
                        classification_stats['sample_patents'][classification].append({
                            'patent_id': patent_id,
                            'model_count': model_count,
                            'abstract': data.get('abstract', '')[:100] + '...' if data.get('abstract', '') else '',
                            'has_text': bool(data.get('abstract', '') or data.get('full_text', ''))
                        })

                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")
                    continue

                if line_num % 10000 == 0:
                    logger.info(f"Processed {line_num:,} patents...")

    logger.info("Classification analysis complete!")
    return classification_stats


def print_classification_analysis(stats: dict):
    """Print detailed classification analysis"""

    print("\n" + "="*80)
    print("PATENT CLASSIFICATION ANALYSIS")
    print("="*80)

    print(f"\n📊 TOTAL PATENTS: {stats['total_patents']:,}")

    # Overall classification distribution
    print(f"\n🏷️  CLASSIFICATION DISTRIBUTION:")
    total_classified = sum(count for cls, count in stats['classification_counts'].items()
                          if cls not in ['EMPTY', 'NULL'])

    for classification, count in stats['classification_counts'].most_common():
        percentage = (count / stats['total_patents']) * 100
        if classification in ['EMPTY', 'NULL']:
            print(f"  {classification:>10}: {count:>8,} ({percentage:>5.1f}%) ⚠️")
        else:
            print(f"  {classification:>10}: {count:>8,} ({percentage:>5.1f}%)")

    print(f"\n📈 CLASSIFICATION QUALITY:")
    print(f"  Total with valid classifications: {total_classified:,} ({(total_classified/stats['total_patents']*100):>5.1f}%)")
    print(f"  Empty classifications: {stats['empty_classifications']:,}")
    print(f"  Null classifications: {stats['null_classifications']:,}")
    print(f"  Data quality issues: {(stats['empty_classifications'] + stats['null_classifications']):,}")

    # Classification by model count
    print(f"\n🎯 CLASSIFICATION BY MODEL COVERAGE:")
    for model_count in sorted(stats['classification_by_model_count'].keys(), reverse=True):
        count_data = stats['classification_by_model_count'][model_count]
        total_for_model_count = sum(count_data.values())
        print(f"\n  Patents with {model_count} model(s) ({total_for_model_count:,} patents):")

        for classification, count in count_data.most_common(10):  # Top 10
            percentage = (count / total_for_model_count) * 100
            if classification in ['EMPTY', 'NULL']:
                print(f"    {classification:>10}: {count:>6,} ({percentage:>5.1f}%) ⚠️")
            else:
                print(f"    {classification:>10}: {count:>6,} ({percentage:>5.1f}%)")

    # Sample patents for each classification
    print(f"\n📋 SAMPLE PATENTS BY CLASSIFICATION:")
    for classification, samples in stats['sample_patents'].items():
        print(f"\n  {classification} ({len(samples)} samples):")
        for i, patent in enumerate(samples[:3]):  # Show top 3
            print(f"    {i+1}. {patent['patent_id']} (models: {patent['model_count']}, has_text: {patent['has_text']})")
            if patent['abstract']:
                print(f"       {patent['abstract']}")

    print("\n" + "="*80)


def create_classification_plots(stats: dict, output_dir: str = "figures"):
    """Create visualization plots for classification analysis"""

    logger.info("Creating classification visualization plots...")

    # Prepare data
    classifications = []
    counts = []
    colors = []

    for classification, count in stats['classification_counts'].most_common():
        classifications.append(classification)
        counts.append(count)

        # Color code problematic classifications
        if classification in ['EMPTY', 'NULL']:
            colors.append('#ff6b6b')  # Red for problems
        elif classification in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
            colors.append('#4ecdc4')  # Teal for numeric
        else:
            colors.append('#45b7d1')  # Blue for others

    # Create figure with multiple subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Patent Classification Analysis', fontsize=16, fontweight='bold')

    # Plot 1: Overall distribution (bar chart)
    bars = ax1.bar(range(len(classifications)), counts, color=colors)
    ax1.set_xlabel('Classification')
    ax1.set_ylabel('Number of Patents')
    ax1.set_title('Classification Distribution (All Patents)')
    ax1.set_xticks(range(len(classifications)))
    ax1.set_xticklabels(classifications, rotation=45, ha='right')

    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}', ha='center', va='bottom', fontsize=8)

    # Plot 2: Pie chart (excluding empty/null for clarity)
    valid_classifications = {k: v for k, v in stats['classification_counts'].items()
                           if k not in ['EMPTY', 'NULL']}

    if valid_classifications:
        ax2.pie(valid_classifications.values(), labels=valid_classifications.keys(), autopct='%1.1f%%')
        ax2.set_title('Valid Classifications Distribution')
    else:
        ax2.text(0.5, 0.5, 'No valid classifications found', ha='center', va='center')
        ax2.set_title('Valid Classifications Distribution')

    # Plot 3: Classification by model count (stacked bar)
    model_counts = sorted(stats['classification_by_model_count'].keys(), reverse=True)
    classification_names = list(stats['classification_counts'].keys())

    # Prepare stacked data
    stack_data = []
    for classification in classification_names[:10]:  # Top 10 classifications
        row = []
        for model_count in model_counts:
            count = stats['classification_by_model_count'][model_count].get(classification, 0)
            row.append(count)
        stack_data.append(row)

    # Create stacked bar chart
    bottom = np.zeros(len(model_counts))
    for i, (classification, data_row) in enumerate(zip(classification_names[:10], stack_data)):
        color = colors[i] if i < len(colors) else '#95a5a6'
        ax3.bar(model_counts, data_row, bottom=bottom, label=classification, color=color)
        bottom += data_row

    ax3.set_xlabel('Number of Models')
    ax3.set_ylabel('Number of Patents')
    ax3.set_title('Classifications by Model Coverage')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Plot 4: Data quality summary
    quality_data = {
        'Valid Classifications': sum(count for cls, count in stats['classification_counts'].items()
                                   if cls not in ['EMPTY', 'NULL']),
        'Empty Classifications': stats['empty_classifications'],
        'Null Classifications': stats['null_classifications']
    }

    quality_colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green, orange, red
    wedges, texts, autotexts = ax4.pie(quality_data.values(), labels=quality_data.keys(),
                                       autopct='%1.1f%%', colors=quality_colors)
    ax4.set_title('Data Quality Overview')

    # Style the plot
    plt.tight_layout()

    # Save plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_file = output_path / "classification_analysis.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved classification analysis plot: {plot_file}")

    plt.show()

    return str(plot_file)


def main():
    """Main execution"""
    master_file = "data_v2/master_patent_embeddings.jsonl"

    if not Path(master_file).exists():
        print(f"❌ Master embeddings file not found: {master_file}")
        return

    try:
        # Analyze classifications
        stats = analyze_classifications(master_file)

        # Print analysis
        print_classification_analysis(stats)

        # Create plots
        plot_file = create_classification_plots(stats)

        # Save detailed results
        results_file = Path("analysis/classification_analysis_results.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert Counter objects to regular dicts for JSON serialization
        json_stats = {
            'total_patents': stats['total_patents'],
            'classification_counts': dict(stats['classification_counts']),
            'empty_classifications': stats['empty_classifications'],
            'null_classifications': stats['null_classifications'],
            'classification_by_model_count': {
                str(k): dict(v) for k, v in stats['classification_by_model_count'].items()
            },
            'sample_patents': dict(stats['sample_patents'])
        }

        with open(results_file, 'w') as f:
            json.dump(json_stats, f, indent=2)

        logger.info(f"Detailed results saved to: {results_file}")

        print(f"\n🎯 SUMMARY:")
        print(f"✅ Analysis complete for {stats['total_patents']:,} patents")
        print(f"📊 Plot saved: {plot_file}")
        print(f"💾 Detailed results: {results_file}")

    except Exception as e:
        logger.error(f"Classification analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()