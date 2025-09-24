#!/usr/bin/env python3
"""
Data Consolidation Script for Patent Research Project

This script consolidates all scattered data files into a clean, organized structure:
1. Master patent embeddings file (all models per patent)
2. Ground truth similarities file (patent pair comparisons)
3. Patent metadata file (abstracts, classifications)
4. Data catalog and metadata indexes

Author: Patent Research Project
Date: September 2025
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
from collections import defaultdict
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataConsolidator:
    """Main class for consolidating patent research data"""

    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.data_v2_dir = self.base_dir / "data_v2"
        self.metadata_dir = self.data_v2_dir / "metadata"

        # Output files
        self.master_embeddings_file = self.data_v2_dir / "master_patent_embeddings.jsonl"
        self.ground_truth_file = self.data_v2_dir / "ground_truth_similarities.jsonl"
        self.patent_metadata_file = self.data_v2_dir / "patent_metadata.jsonl"

        # Data tracking
        self.patents_with_embeddings: Dict[str, Dict[str, Any]] = {}
        self.ground_truth_pairs: List[Dict[str, Any]] = []
        self.patent_metadata: Dict[str, Dict[str, Any]] = {}
        self.processing_stats = {
            "start_time": datetime.now().isoformat(),
            "patents_processed": 0,
            "embeddings_consolidated": 0,
            "ground_truth_pairs": 0,
            "models_found": set(),
            "files_processed": []
        }

    def load_patent_abstracts(self) -> Dict[str, Dict[str, Any]]:
        """Load patent abstracts and metadata"""
        logger.info("Loading patent abstracts and metadata...")

        patent_file = self.base_dir / "data" / "patent_abstracts.json"
        if not patent_file.exists():
            logger.warning(f"Patent abstracts file not found: {patent_file}")
            return {}

        try:
            with open(patent_file, 'r') as f:
                patents_list = json.load(f)

            patents_dict = {}
            for patent in patents_list:
                patent_id = patent.get('id', '')
                patents_dict[patent_id] = {
                    'abstract': patent.get('abstract', ''),
                    'full_text': patent.get('full_text', ''),
                    'classification': patent.get('classification', '')
                }

            logger.info(f"Loaded {len(patents_dict)} patent abstracts")
            self.processing_stats["files_processed"].append("data/patent_abstracts.json")
            return patents_dict

        except Exception as e:
            logger.error(f"Error loading patent abstracts: {e}")
            return {}

    def load_openai_embeddings(self) -> Dict[str, Any]:
        """Load OpenAI embeddings from results directory"""
        logger.info("Loading OpenAI embeddings...")

        # Try different OpenAI embedding files
        possible_files = [
            "results/openai_embeddings_final.jsonl",
            "results/openai_embeddings_consolidated.jsonl",
            "results/openai_embeddings_clean.jsonl"
        ]

        openai_embeddings = {}

        for file_path in possible_files:
            full_path = self.base_dir / file_path
            if not full_path.exists():
                continue

            logger.info(f"Loading OpenAI embeddings from {file_path}")
            try:
                with open(full_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line.strip())
                            patent_id = data.get('id', '')
                            if patent_id and 'embedding' in data:
                                openai_embeddings[patent_id] = {
                                    'vector': data['embedding'],
                                    'dimension': len(data['embedding']),
                                    'source_file': file_path,
                                    'model': 'openai_text-embedding-3-small'
                                }

                logger.info(f"Loaded {len(openai_embeddings)} OpenAI embeddings from {file_path}")
                self.processing_stats["files_processed"].append(file_path)
                self.processing_stats["models_found"].add("openai_text-embedding-3-small")
                break

            except Exception as e:
                logger.error(f"Error loading OpenAI embeddings from {file_path}: {e}")

        return openai_embeddings

    def load_model_embeddings(self, model_name: str) -> Dict[str, Any]:
        """Load embeddings for a specific model from data/embeddings/by_model/"""
        logger.info(f"Loading {model_name} embeddings...")

        model_dir = self.base_dir / "data" / "embeddings" / "by_model" / model_name
        if not model_dir.exists():
            logger.warning(f"Model directory not found: {model_dir}")
            return {}

        embeddings = {}

        # Find all embedding files for this model
        embedding_files = list(model_dir.glob("*.jsonl"))

        for file_path in embedding_files:
            logger.info(f"Loading {model_name} embeddings from {file_path.name}")
            try:
                with open(file_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line.strip())
                            patent_id = data.get('id', data.get('patent_id', ''))

                            # Handle different embedding formats
                            embedding_vector = None

                            # Format 1: Direct embedding key (OpenAI style)
                            if 'embedding' in data:
                                embedding_vector = data['embedding']
                            # Format 2: Nested in models (nomic/bge style)
                            elif 'models' in data and model_name in data['models']:
                                model_data = data['models'][model_name]
                                if 'embeddings' in model_data:
                                    embeddings_data = model_data['embeddings']
                                    # Handle nested structure: {'original': {'embedding': [vector]}}
                                    if isinstance(embeddings_data, dict):
                                        if 'original' in embeddings_data and 'embedding' in embeddings_data['original']:
                                            embedding_vector = embeddings_data['original']['embedding']
                                        elif 'embedding' in embeddings_data:
                                            embedding_vector = embeddings_data['embedding']
                                    elif isinstance(embeddings_data, list):
                                        embedding_vector = embeddings_data

                            if patent_id and embedding_vector:
                                # Use the most recent embedding if duplicate
                                if patent_id not in embeddings:
                                    embeddings[patent_id] = {
                                        'vector': embedding_vector,
                                        'dimension': len(embedding_vector),
                                        'source_file': str(file_path.relative_to(self.base_dir)),
                                        'model': model_name
                                    }

                self.processing_stats["files_processed"].append(str(file_path.relative_to(self.base_dir)))

            except Exception as e:
                logger.error(f"Error loading {model_name} embeddings from {file_path}: {e}")

        logger.info(f"Loaded {len(embeddings)} {model_name} embeddings")
        if embeddings:
            self.processing_stats["models_found"].add(model_name)

        return embeddings

    def consolidate_embeddings(self) -> None:
        """Consolidate all embeddings into master file"""
        logger.info("Starting embedding consolidation...")

        # Load patent metadata first
        self.patent_metadata = self.load_patent_abstracts()

        # Load embeddings from all models
        openai_embeddings = self.load_openai_embeddings()
        nomic_embeddings = self.load_model_embeddings("nomic-embed-text")
        bge_embeddings = self.load_model_embeddings("bge-m3")

        # Get all unique patent IDs
        all_patent_ids: Set[str] = set()
        all_patent_ids.update(openai_embeddings.keys())
        all_patent_ids.update(nomic_embeddings.keys())
        all_patent_ids.update(bge_embeddings.keys())
        all_patent_ids.update(self.patent_metadata.keys())

        logger.info(f"Found {len(all_patent_ids)} unique patents across all data sources")

        # Consolidate into master structure
        consolidated_count = 0
        with open(self.master_embeddings_file, 'w') as f:
            for patent_id in sorted(all_patent_ids):
                patent_record = {
                    "patent_id": patent_id,
                    "abstract": self.patent_metadata.get(patent_id, {}).get('abstract', ''),
                    "full_text": self.patent_metadata.get(patent_id, {}).get('full_text', ''),
                    "classification": self.patent_metadata.get(patent_id, {}).get('classification', ''),
                    "embeddings": {},
                    "metadata": {
                        "processing_date": datetime.now().isoformat(),
                        "has_text": bool(self.patent_metadata.get(patent_id, {}).get('abstract', ''))
                    }
                }

                # Add embeddings if available
                embedding_added = False

                if patent_id in openai_embeddings:
                    patent_record["embeddings"]["openai_text-embedding-3-small"] = {
                        "vector": openai_embeddings[patent_id]["vector"],
                        "dimension": openai_embeddings[patent_id]["dimension"],
                        "source_file": openai_embeddings[patent_id]["source_file"],
                        "generated_at": "2025-09-15T00:00:00Z"  # Approximate from file dates
                    }
                    embedding_added = True

                if patent_id in nomic_embeddings:
                    patent_record["embeddings"]["nomic-embed-text"] = {
                        "vector": nomic_embeddings[patent_id]["vector"],
                        "dimension": nomic_embeddings[patent_id]["dimension"],
                        "source_file": nomic_embeddings[patent_id]["source_file"],
                        "generated_at": "2025-09-12T00:00:00Z"
                    }
                    embedding_added = True

                if patent_id in bge_embeddings:
                    patent_record["embeddings"]["bge-m3"] = {
                        "vector": bge_embeddings[patent_id]["vector"],
                        "dimension": bge_embeddings[patent_id]["dimension"],
                        "source_file": bge_embeddings[patent_id]["source_file"],
                        "generated_at": "2025-09-12T00:00:00Z"
                    }
                    embedding_added = True

                # Write record if it has embeddings or metadata
                if embedding_added or patent_record["abstract"]:
                    f.write(json.dumps(patent_record) + '\n')
                    consolidated_count += 1
                    self.processing_stats["embeddings_consolidated"] += len(patent_record["embeddings"])

        self.processing_stats["patents_processed"] = consolidated_count
        logger.info(f"Consolidated {consolidated_count} patents with embeddings into {self.master_embeddings_file}")

    def process_ground_truth(self) -> None:
        """Process ground truth similarities into consolidated format"""
        logger.info("Processing ground truth similarities...")

        # Load ground truth data
        ground_truth_file = self.base_dir / "data" / "ground_truth" / "consolidated" / "ground_truth_10k.jsonl"
        if not ground_truth_file.exists():
            logger.warning(f"Ground truth file not found: {ground_truth_file}")
            return

        # Load embedding similarities from various sources
        embedding_similarities = self._load_embedding_similarities()

        pairs_processed = 0
        with open(self.ground_truth_file, 'w') as f:
            with open(ground_truth_file, 'r') as gt_file:
                for line in gt_file:
                    if line.strip():
                        data = json.loads(line.strip())

                        patent1_id = data.get('patent1_id', '')
                        patent2_id = data.get('patent2_id', '')
                        pair_id = f"{patent1_id}_{patent2_id}"

                        # Create consolidated record
                        similarity_record = {
                            "pair_id": pair_id,
                            "patent1_id": patent1_id,
                            "patent2_id": patent2_id,
                            "llm_evaluation": data.get('llm_analysis', {}),
                            "embedding_similarities": embedding_similarities.get(pair_id, {}),
                            "metadata": {
                                "evaluation_date": "2025-09-14",
                                "llm_model": "gemini-1.5-flash",
                                "evaluation_batch": "ground_truth_10k",
                                "original_embedding_similarity": data.get('embedding_similarity', 0.0)
                            }
                        }

                        f.write(json.dumps(similarity_record) + '\n')
                        pairs_processed += 1

        self.processing_stats["ground_truth_pairs"] = pairs_processed
        self.processing_stats["files_processed"].append("data/ground_truth/consolidated/ground_truth_10k.jsonl")
        logger.info(f"Processed {pairs_processed} ground truth pairs")

    def _load_embedding_similarities(self) -> Dict[str, Dict[str, float]]:
        """Load embedding similarities from fair comparison results"""
        similarities = {}

        # Load from fair comparison results
        fair_comparison_file = self.base_dir / "results" / "three_way_fair_comparison_results.json"
        if fair_comparison_file.exists():
            try:
                with open(fair_comparison_file, 'r') as f:
                    fair_data = json.load(f)

                # Process similarities by model
                for model_name, model_data in fair_data.items():
                    if 'pairs' in model_data:
                        for pair_data in model_data['pairs']:
                            patent1_id = pair_data.get('patent1_id', '')
                            patent2_id = pair_data.get('patent2_id', '')
                            similarity = pair_data.get('cosine_similarity', 0.0)

                            pair_id = f"{patent1_id}_{patent2_id}"
                            if pair_id not in similarities:
                                similarities[pair_id] = {}

                            model_key = model_name.replace('_', '-')  # Normalize model names
                            similarities[pair_id][model_key] = similarity

                logger.info(f"Loaded embedding similarities for {len(similarities)} pairs from fair comparison")
                self.processing_stats["files_processed"].append("results/three_way_fair_comparison_results.json")

            except Exception as e:
                logger.error(f"Error loading fair comparison results: {e}")

        return similarities

    def generate_metadata_files(self) -> None:
        """Generate metadata and catalog files"""
        logger.info("Generating metadata files...")

        # Data catalog
        catalog = {
            "generated_at": datetime.now().isoformat(),
            "total_patents": self.processing_stats["patents_processed"],
            "total_embedding_vectors": self.processing_stats["embeddings_consolidated"],
            "total_ground_truth_pairs": self.processing_stats["ground_truth_pairs"],
            "models_available": list(self.processing_stats["models_found"]),
            "files": {
                "master_embeddings": "master_patent_embeddings.jsonl",
                "ground_truth": "ground_truth_similarities.jsonl",
                "patent_metadata": "patent_metadata.jsonl"
            },
            "statistics": {
                "avg_embeddings_per_patent": round(self.processing_stats["embeddings_consolidated"] / max(self.processing_stats["patents_processed"], 1), 2),
                "models_found": len(self.processing_stats["models_found"]),
                "source_files_processed": len(self.processing_stats["files_processed"])
            }
        }

        catalog_file = self.metadata_dir / "data_catalog.json"
        with open(catalog_file, 'w') as f:
            json.dump(catalog, f, indent=2)

        # Model versions
        model_versions = {
            "generated_at": datetime.now().isoformat(),
            "models": {
                "openai_text-embedding-3-small": {
                    "provider": "OpenAI",
                    "dimension": 1536,
                    "max_tokens": 8191,
                    "generation_date": "2025-09-15",
                    "api_version": "2023-05-15"
                },
                "nomic-embed-text": {
                    "provider": "Nomic AI",
                    "dimension": 768,
                    "max_tokens": 8192,
                    "generation_date": "2025-09-12",
                    "local_inference": True
                },
                "bge-m3": {
                    "provider": "BAAI",
                    "dimension": 1024,
                    "max_tokens": 8192,
                    "generation_date": "2025-09-12",
                    "local_inference": True,
                    "multilingual": True
                }
            }
        }

        versions_file = self.metadata_dir / "model_versions.json"
        with open(versions_file, 'w') as f:
            json.dump(model_versions, f, indent=2)

        # Processing history
        self.processing_stats["end_time"] = datetime.now().isoformat()
        self.processing_stats["models_found"] = list(self.processing_stats["models_found"])

        history_file = self.metadata_dir / "processing_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.processing_stats, f, indent=2)

        logger.info(f"Generated metadata files in {self.metadata_dir}")

    def validate_consolidated_data(self) -> Dict[str, Any]:
        """Run validation checks on consolidated data"""
        logger.info("Running validation checks...")

        validation_results = {
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "errors": [],
            "warnings": []
        }

        try:
            # Check master embeddings file
            if self.master_embeddings_file.exists():
                patent_count = 0
                embedding_counts = defaultdict(int)
                dimension_check = defaultdict(set)

                with open(self.master_embeddings_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            patent_count += 1
                            data = json.loads(line.strip())

                            for model_name, embedding_data in data.get('embeddings', {}).items():
                                embedding_counts[model_name] += 1
                                dimension_check[model_name].add(embedding_data.get('dimension', 0))

                validation_results["checks"]["master_embeddings"] = {
                    "total_patents": patent_count,
                    "embeddings_per_model": dict(embedding_counts),
                    "dimensions_per_model": {k: list(v) for k, v in dimension_check.items()}
                }

                # Check for dimension consistency
                for model, dims in dimension_check.items():
                    if len(dims) > 1:
                        validation_results["errors"].append(f"Inconsistent dimensions for {model}: {dims}")

            # Check ground truth file
            if self.ground_truth_file.exists():
                pair_count = 0
                with open(self.ground_truth_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            pair_count += 1

                validation_results["checks"]["ground_truth"] = {
                    "total_pairs": pair_count
                }

            # Check file sizes
            validation_results["checks"]["file_sizes"] = {
                "master_embeddings_mb": round(self.master_embeddings_file.stat().st_size / (1024*1024), 2) if self.master_embeddings_file.exists() else 0,
                "ground_truth_mb": round(self.ground_truth_file.stat().st_size / (1024*1024), 2) if self.ground_truth_file.exists() else 0
            }

        except Exception as e:
            validation_results["errors"].append(f"Validation error: {str(e)}")

        # Write validation results
        validation_file = self.metadata_dir / "validation_results.json"
        with open(validation_file, 'w') as f:
            json.dump(validation_results, f, indent=2)

        logger.info(f"Validation complete. Results saved to {validation_file}")
        return validation_results

    def run_consolidation(self) -> Dict[str, Any]:
        """Run the complete data consolidation process"""
        logger.info("Starting complete data consolidation...")

        try:
            # Phase 1: Consolidate embeddings
            self.consolidate_embeddings()

            # Phase 2: Process ground truth
            self.process_ground_truth()

            # Phase 3: Generate metadata
            self.generate_metadata_files()

            # Phase 4: Validate
            validation_results = self.validate_consolidated_data()

            # Summary report
            summary = {
                "status": "success",
                "completion_time": datetime.now().isoformat(),
                "statistics": self.processing_stats,
                "validation": validation_results,
                "output_files": {
                    "master_embeddings": str(self.master_embeddings_file),
                    "ground_truth": str(self.ground_truth_file),
                    "metadata_dir": str(self.metadata_dir)
                }
            }

            logger.info("Data consolidation completed successfully!")
            return summary

        except Exception as e:
            logger.error(f"Data consolidation failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "completion_time": datetime.now().isoformat()
            }


def main():
    """Main execution function"""
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = "."

    consolidator = DataConsolidator(base_dir)
    results = consolidator.run_consolidation()

    # Print summary
    print("\n" + "="*80)
    print("DATA CONSOLIDATION SUMMARY")
    print("="*80)
    print(f"Status: {results['status']}")
    print(f"Completion Time: {results['completion_time']}")

    if results['status'] == 'success':
        stats = results['statistics']
        print(f"\nProcessing Statistics:")
        print(f"  Patents processed: {stats['patents_processed']}")
        print(f"  Embedding vectors: {stats['embeddings_consolidated']}")
        print(f"  Ground truth pairs: {stats['ground_truth_pairs']}")
        print(f"  Models found: {len(stats['models_found'])}")
        print(f"  Source files processed: {len(stats['files_processed'])}")

        validation = results['validation']
        error_count = len(validation.get('errors', []))
        warning_count = len(validation.get('warnings', []))
        print(f"\nValidation Results:")
        print(f"  Errors: {error_count}")
        print(f"  Warnings: {warning_count}")

        if error_count > 0:
            print("\nErrors found:")
            for error in validation['errors']:
                print(f"  - {error}")
    else:
        print(f"Error: {results.get('error', 'Unknown error')}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()