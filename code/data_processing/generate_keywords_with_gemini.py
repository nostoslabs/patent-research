#!/usr/bin/env python3
"""
Generate Keywords with Gemini

Selects 1000 random patents with all three embeddings and uses Gemini to generate
relevant keywords based on abstracts. This will create meaningful classifications
for Atlas visualization.
"""

import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
import umap
import google.generativeai as genai
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def configure_gemini():
    """Configure Gemini API"""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY environment variable not set")
        print("Please set it with: export GEMINI_API_KEY='your-api-key'")
        return None

    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-pro')


def select_random_patents(master_file: str, target_count: int = 1000) -> List[Dict]:
    """
    Select random patents that have all three embeddings.

    Args:
        master_file: Path to master embeddings file
        target_count: Number of patents to select

    Returns:
        List of selected patent data
    """
    logger.info(f"Selecting {target_count} random patents with all three models...")

    eligible_patents = []

    with open(master_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    data = json.loads(line.strip())
                    embeddings = data.get('embeddings', {})

                    # Check if has all three models
                    required_models = {
                        'openai_text-embedding-3-small',
                        'nomic-embed-text',
                        'bge-m3'
                    }

                    if required_models.issubset(embeddings.keys()):
                        # Only include patents with abstracts for keyword generation
                        abstract = data.get('abstract', '').strip()
                        if abstract and len(abstract) > 50:  # Minimum abstract length
                            eligible_patents.append(data)

                            # Stop early if we have enough candidates
                            if len(eligible_patents) >= target_count * 3:
                                break

                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")
                    continue

                if line_num % 5000 == 0:
                    logger.info(f"Processed {line_num:,} patents, found {len(eligible_patents)} eligible...")

    logger.info(f"Found {len(eligible_patents)} eligible patents with all three models and abstracts")

    # Randomly select the target count
    if len(eligible_patents) > target_count:
        selected = random.sample(eligible_patents, target_count)
    else:
        selected = eligible_patents
        logger.warning(f"Only found {len(selected)} eligible patents (requested {target_count})")

    logger.info(f"Selected {len(selected)} random patents")
    return selected


def generate_keywords_with_gemini(model, abstract: str, patent_id: str) -> Dict:
    """
    Use Gemini to generate keywords from patent abstract.

    Args:
        model: Gemini model instance
        abstract: Patent abstract text
        patent_id: Patent identifier for logging

    Returns:
        Dictionary with generated keywords and categories
    """

    prompt = f"""
Analyze this patent abstract and generate relevant technical keywords and categories:

ABSTRACT: {abstract}

Please provide:
1. PRIMARY_CATEGORY: The main technical field (e.g., "Machine Learning", "Medical Devices", "Energy Storage")
2. KEYWORDS: 3-5 specific technical keywords (comma-separated)
3. APPLICATION_AREA: The main application domain (e.g., "Healthcare", "Automotive", "Telecommunications")

Format your response as:
PRIMARY_CATEGORY: [category]
KEYWORDS: [keyword1, keyword2, keyword3, ...]
APPLICATION_AREA: [area]

Be concise and focus on the core technical concepts.
"""

    try:
        response = model.generate_content(prompt)
        result_text = response.text.strip()

        # Parse the response
        keywords_data = {
            'primary_category': 'Unknown',
            'keywords': [],
            'application_area': 'Unknown',
            'raw_response': result_text
        }

        for line in result_text.split('\n'):
            line = line.strip()
            if line.startswith('PRIMARY_CATEGORY:'):
                keywords_data['primary_category'] = line.replace('PRIMARY_CATEGORY:', '').strip()
            elif line.startswith('KEYWORDS:'):
                keywords_str = line.replace('KEYWORDS:', '').strip()
                keywords_data['keywords'] = [k.strip() for k in keywords_str.split(',')]
            elif line.startswith('APPLICATION_AREA:'):
                keywords_data['application_area'] = line.replace('APPLICATION_AREA:', '').strip()

        return keywords_data

    except Exception as e:
        logger.error(f"Error generating keywords for {patent_id}: {e}")
        return {
            'primary_category': 'Error',
            'keywords': [],
            'application_area': 'Error',
            'raw_response': f"Error: {str(e)}"
        }


def generate_keywords_for_patents(patents: List[Dict]) -> List[Dict]:
    """
    Generate keywords for all selected patents using Gemini.

    Args:
        patents: List of patent data

    Returns:
        List of patents with added keyword data
    """
    logger.info("Configuring Gemini API...")
    model = configure_gemini()

    if not model:
        return []

    logger.info(f"Generating keywords for {len(patents)} patents...")

    enhanced_patents = []

    for i, patent in enumerate(patents):
        patent_id = patent.get('patent_id', f'patent_{i}')
        abstract = patent.get('abstract', '')

        logger.info(f"Processing {i+1}/{len(patents)}: {patent_id}")

        # Generate keywords
        keywords_data = generate_keywords_with_gemini(model, abstract, patent_id)

        # Add to patent data
        patent_copy = patent.copy()
        patent_copy['gemini_keywords'] = keywords_data
        enhanced_patents.append(patent_copy)

        # Rate limiting - Gemini has usage limits
        if i < len(patents) - 1:  # Don't sleep after the last one
            time.sleep(1)  # 1 second between requests

        # Progress update
        if (i + 1) % 50 == 0:
            logger.info(f"Generated keywords for {i+1}/{len(patents)} patents...")

    logger.info("Keyword generation complete!")
    return enhanced_patents


def create_keyword_atlas_dataset(patents: List[Dict]) -> pd.DataFrame:
    """
    Create Atlas dataset using generated keywords as categories.

    Args:
        patents: Patents with generated keywords

    Returns:
        DataFrame ready for Atlas visualization
    """
    logger.info("Creating keyword-based Atlas dataset...")

    atlas_data = []

    for patent in patents:
        patent_id = patent.get('patent_id', '')
        abstract = patent.get('abstract', '')
        embeddings = patent.get('embeddings', {})
        keywords_data = patent.get('gemini_keywords', {})

        # Use OpenAI embeddings for visualization
        if 'openai_text-embedding-3-small' in embeddings:
            embedding_vector = embeddings['openai_text-embedding-3-small']['vector']

            # Create display text (first 300 chars of abstract)
            display_text = abstract[:300] + '...' if len(abstract) > 300 else abstract

            atlas_data.append({
                'patent_id': patent_id,
                'text': display_text,
                'embedding': embedding_vector,
                'primary_category': keywords_data.get('primary_category', 'Unknown'),
                'application_area': keywords_data.get('application_area', 'Unknown'),
                'keywords': ', '.join(keywords_data.get('keywords', [])),
                'model_count': len(embeddings)
            })

    if not atlas_data:
        logger.error("No valid atlas data created")
        return pd.DataFrame()

    # Create DataFrame
    df = pd.DataFrame(atlas_data)

    # Create UMAP projection
    embeddings_matrix = np.array(df['embedding'].tolist())

    logger.info(f"Creating UMAP projection for {len(df)} patents...")
    umap_reducer = umap.UMAP(n_components=2, random_state=42, metric='cosine')
    umap_result = umap_reducer.fit_transform(embeddings_matrix)

    df['umap_x'] = umap_result[:, 0]
    df['umap_y'] = umap_result[:, 1]

    # Remove embedding column for parquet compatibility
    df = df.drop('embedding', axis=1)

    logger.info("Keyword Atlas dataset created!")
    return df


def save_keyword_dataset(df: pd.DataFrame, patents_with_keywords: List[Dict], output_dir: str = "data_v2/atlas_data"):
    """Update existing Atlas dataset with keyword data"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Update the existing complete_coverage dataset
    atlas_file = output_path / "complete_coverage_enhanced_atlas.parquet"
    df.to_parquet(atlas_file, index=False)

    # Update launch script to use keyword categories
    launch_script = output_path / "launch_atlas_temp.sh"
    with open(launch_script, 'w') as f:
        f.write(f"""#!/bin/bash
echo "Launching Enhanced Patent Atlas with Gemini Keywords..."
echo "Dataset: complete_coverage_enhanced_atlas.parquet ({len(df)} patents with AI-generated categories)"
echo ""
echo "Categories available:"
echo "- primary_category: Main technical field (Gemini-generated)"
echo "- application_area: Application domain (Gemini-generated)"
echo "- keywords: Specific technical terms"
echo ""
echo "Atlas will open at http://localhost:8001"
echo "Use Ctrl+C to stop when done"
echo ""

cd data_v2/atlas_data || exit 1

uv run embedding-atlas \\
    complete_coverage_enhanced_atlas.parquet \\
    --text text \\
    --x umap_x \\
    --y umap_y \\
    --color primary_category \\
    --host localhost \\
    --port 8001
""")

    # Make executable
    launch_script.chmod(0o755)

    return atlas_file, launch_script


def print_keyword_analysis(df: pd.DataFrame):
    """Print analysis of generated keywords"""

    print("\n" + "="*60)
    print("GEMINI KEYWORD ANALYSIS")
    print("="*60)

    print(f"\n📊 DATASET SUMMARY:")
    print(f"  Total patents: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    print(f"\n🏷️  PRIMARY CATEGORIES:")
    category_counts = df['primary_category'].value_counts()
    for category, count in category_counts.head(10).items():
        percentage = (count / len(df)) * 100
        print(f"  {category:.<25} {count:>3} ({percentage:>4.1f}%)")

    print(f"\n🎯 APPLICATION AREAS:")
    area_counts = df['application_area'].value_counts()
    for area, count in area_counts.head(10).items():
        percentage = (count / len(df)) * 100
        print(f"  {area:.<25} {count:>3} ({percentage:>4.1f}%)")

    print(f"\n📈 CATEGORY DIVERSITY:")
    print(f"  Unique primary categories: {df['primary_category'].nunique()}")
    print(f"  Unique application areas: {df['application_area'].nunique()}")

    # Show some sample keywords
    print(f"\n📋 SAMPLE KEYWORDS:")
    for i, row in df.head(5).iterrows():
        print(f"  {row['patent_id']}: {row['primary_category']} | {row['keywords'][:80]}...")

    print("="*60)


def main():
    """Main execution"""
    master_file = "data_v2/master_patent_embeddings.jsonl"

    if not Path(master_file).exists():
        print(f"❌ Master embeddings file not found: {master_file}")
        return

    try:
        # Set random seed for reproducibility
        random.seed(42)

        # Select random patents
        selected_patents = select_random_patents(master_file, target_count=1000)

        if not selected_patents:
            print("❌ No eligible patents found")
            return

        # Generate keywords with Gemini
        patents_with_keywords = generate_keywords_for_patents(selected_patents)

        if not patents_with_keywords:
            print("❌ Failed to generate keywords")
            return

        # Create Atlas dataset
        keyword_df = create_keyword_atlas_dataset(patents_with_keywords)

        if keyword_df.empty:
            print("❌ Failed to create Atlas dataset")
            return

        # Update existing Atlas dataset
        atlas_file, launch_script = save_keyword_dataset(
            keyword_df, patents_with_keywords
        )

        # Print analysis
        print_keyword_analysis(keyword_df)

        print(f"\n🎯 SUCCESS!")
        print(f"✅ Updated Atlas dataset: {atlas_file}")
        print(f"✅ Launch script: {launch_script}")
        print(f"\n🚀 Launch enhanced Atlas with:")
        print(f"bash {launch_script}")

    except Exception as e:
        logger.error(f"Keyword generation failed: {e}")
        raise


if __name__ == "__main__":
    main()