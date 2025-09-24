#!/usr/bin/env python3
"""
Plot Classification Distribution

Creates visualizations showing the distribution of patent classifications
and the data quality issues we discovered.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Set style
plt.style.use('default')
sns.set_palette("husl")

def create_classification_plots():
    """Create comprehensive classification distribution plots"""

    # Data based on our analysis
    classification_data = {
        'Classification': ['Unclassified', 'Class 1', 'Class 7'],
        'Count': [39474, 897, 32],
        'Percentage': [97.7, 2.2, 0.1]
    }

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Patent Classification Analysis - Data Quality Issues', fontsize=16, fontweight='bold')

    # Plot 1: Raw counts (bar chart)
    colors = ['#e74c3c', '#3498db', '#2ecc71']  # Red for unclassified, blue/green for classes
    bars = ax1.bar(classification_data['Classification'], classification_data['Count'], color=colors)
    ax1.set_ylabel('Number of Patents')
    ax1.set_title('Classification Distribution (Raw Counts)')
    ax1.tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar, count in zip(bars, classification_data['Count']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')

    # Plot 2: Pie chart showing the massive data quality issue
    ax2.pie(classification_data['Count'], labels=classification_data['Classification'],
            autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Classification Coverage - Showing Data Quality Issue')

    # Plot 3: Log scale to show actual classified data
    classified_data = {
        'Classification': ['Class 1', 'Class 7'],
        'Count': [897, 32]
    }

    bars3 = ax3.bar(classified_data['Classification'], classified_data['Count'],
                    color=['#3498db', '#2ecc71'])
    ax3.set_ylabel('Number of Patents (Classified Only)')
    ax3.set_title('Distribution of Actually Classified Patents')
    ax3.tick_params(axis='x', rotation=45)

    # Add value labels
    for bar, count in zip(bars3, classified_data['Count']):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom', fontweight='bold')

    # Plot 4: Data quality summary
    quality_summary = {
        'Data Quality': ['Has Classification', 'Missing Classification'],
        'Count': [929, 39474],
        'Percentage': [2.3, 97.7]
    }

    quality_colors = ['#2ecc71', '#e74c3c']  # Green for good, red for bad
    wedges, texts, autotexts = ax4.pie(quality_summary['Count'],
                                       labels=quality_summary['Data Quality'],
                                       autopct='%1.1f%%',
                                       colors=quality_colors,
                                       startangle=90)
    ax4.set_title('Overall Data Quality')

    # Make percentage text bold and larger
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)

    plt.tight_layout()

    # Save the plot
    output_path = Path("figures")
    output_path.mkdir(parents=True, exist_ok=True)
    plot_file = output_path / "classification_distribution_analysis.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')

    print(f"📊 Classification distribution plot saved: {plot_file}")

    # Show the plot
    plt.show()

    # Create summary statistics
    print("\n" + "="*60)
    print("CLASSIFICATION DISTRIBUTION SUMMARY")
    print("="*60)
    print(f"📊 Total Patents: 40,403")
    print(f"📈 Classification Breakdown:")
    print(f"   Unclassified: 39,474 patents (97.7%) ❌")
    print(f"   Class 1:         897 patents (2.2%)  ✅")
    print(f"   Class 7:          32 patents (0.1%)  ✅")
    print(f"")
    print(f"🎯 Key Insights:")
    print(f"   • Only 929 patents (2.3%) have any classification")
    print(f"   • Class 1 dominates with 96.6% of classified patents")
    print(f"   • Class 7 is very rare with only 32 patents")
    print(f"   • 97.7% of patents have no classification metadata")
    print(f"")
    print(f"🔍 Root Cause:")
    print(f"   • Most patents came from embedding files without metadata")
    print(f"   • Only original patent_abstracts.json had classifications")
    print(f"   • Classification system appears to be binary or minimal")
    print(f"")
    print(f"💡 Recommendation:")
    print(f"   • Use other metadata for Atlas visualization (patent ranges, text quality)")
    print(f"   • Consider creating semantic classifications using embeddings")
    print(f"   • Focus analysis on the 929 patents with rich metadata")
    print("="*60)

if __name__ == "__main__":
    create_classification_plots()