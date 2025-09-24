"""Consolidate and organize all embedding files."""

import json
import shutil
import re
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingConsolidator:
    """Consolidate embeddings from multiple files and experiments."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.data_root = self.project_root / "data"
        
        # Master index to track all patents and their embeddings
        self.master_index = {
            "patents": {},
            "models": {},
            "experiments": {},
            "statistics": {},
            "last_updated": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Model configurations
        self.model_configs = {
            "nomic-embed-text": {"embedding_dim": 768, "context_limit": 8192},
            "bge-m3": {"embedding_dim": 1024, "context_limit": 8192},
            "embeddinggemma": {"embedding_dim": 768, "context_limit": 2000},
            "mxbai-embed-large": {"embedding_dim": 1024, "context_limit": 512}
        }
    
    def discover_all_files(self) -> Dict[str, List[Path]]:
        """Discover all data files in the project."""
        files = {
            "embedding_files": [],
            "ground_truth_files": [],
            "raw_patent_files": [],
            "comparison_files": [],
            "experiment_files": []
        }
        
        # Find embedding files
        for pattern in ["*embeddings.jsonl", "*_embeddings.jsonl"]:
            files["embedding_files"].extend(self.project_root.glob(pattern))
            files["embedding_files"].extend(self.data_root.glob(f"**/{pattern}"))
        
        # Find ground truth files
        for pattern in ["*ground_truth*.jsonl", "ground_truth_*.jsonl"]:
            files["ground_truth_files"].extend(self.project_root.glob(pattern))
            files["ground_truth_files"].extend(self.data_root.glob(f"**/{pattern}"))
        
        # Find raw patent files
        for pattern in ["patent_abstracts*.jsonl"]:
            files["raw_patent_files"].extend(self.data_root.glob(f"**/{pattern}"))
        
        # Find comparison files
        for pattern in ["*comparison*.jsonl", "*baseline*.jsonl"]:
            files["comparison_files"].extend(self.project_root.glob(pattern))
            files["comparison_files"].extend(self.data_root.glob(f"**/{pattern}"))
        
        # Find experiment tracking files
        for pattern in ["experiment_tracking.json", "*batch_results.json"]:
            files["experiment_files"].extend(self.project_root.glob(pattern))
            files["experiment_files"].extend(self.data_root.glob(f"**/{pattern}"))
        
        # Remove duplicates and filter out current job files
        for category in files:
            files[category] = list(set(files[category]))
            # Filter out current ground truth generation files
            files[category] = [f for f in files[category] 
                             if not str(f).startswith('./ground_truth_partial_')]
        
        return files
    
    def parse_embedding_filename(self, filepath: Path) -> Dict[str, Any]:
        """Parse embedding filename to extract metadata."""
        filename = filepath.name
        
        # Extract model name
        model = None
        for model_name in self.model_configs.keys():
            if model_name in filename:
                model = model_name
                break
        
        # Extract dataset info
        dataset_size = None
        dataset_type = None
        
        # Look for patterns like "100k", "10k", "500"
        size_match = re.search(r'(\d+k?)', filename)
        if size_match:
            size_str = size_match.group(1)
            if size_str.endswith('k'):
                dataset_size = int(size_str[:-1]) * 1000
            else:
                dataset_size = int(size_str)
        
        # Extract date
        date_match = re.search(r'(\d{8}_\d{6})', filename)
        date = date_match.group(1) if date_match else None
        
        # Determine experiment type
        if "production" in filename:
            experiment_type = "production"
        elif "diverse" in filename:
            experiment_type = "diverse"
        elif "validation" in filename:
            experiment_type = "validation"
        elif "original" in filename:
            experiment_type = "original"
        else:
            experiment_type = "unknown"
        
        return {
            "model": model,
            "dataset_size": dataset_size,
            "dataset_type": dataset_type,
            "experiment_type": experiment_type,
            "date": date,
            "filepath": filepath,
            "filename": filename
        }
    
    def count_records_in_file(self, filepath: Path) -> int:
        """Count number of records in a JSONL file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return sum(1 for line in f if line.strip())
        except Exception as e:
            logger.warning(f"Could not count records in {filepath}: {e}")
            return 0
    
    def analyze_embedding_file(self, filepath: Path) -> Dict[str, Any]:
        """Analyze an embedding file to understand its contents."""
        metadata = self.parse_embedding_filename(filepath)
        
        # Count records
        record_count = self.count_records_in_file(filepath)
        metadata["record_count"] = record_count
        
        # Sample first record to understand structure
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if first_line:
                    sample_record = json.loads(first_line)
                    metadata["has_multi_model"] = "models" in sample_record
                    metadata["has_direct_embedding"] = "embedding" in sample_record
                    
                    if metadata["has_multi_model"]:
                        # Multi-model format
                        metadata["available_models"] = list(sample_record.get("models", {}).keys())
                    elif metadata["has_direct_embedding"]:
                        # Direct embedding format
                        metadata["embedding_dim"] = len(sample_record["embedding"]) if isinstance(sample_record["embedding"], list) else None
        
        except Exception as e:
            logger.warning(f"Could not analyze structure of {filepath}: {e}")
        
        # Get file size
        metadata["file_size_mb"] = filepath.stat().st_size / (1024 * 1024)
        metadata["last_modified"] = filepath.stat().st_mtime
        
        return metadata
    
    def build_master_index(self, files: Dict[str, List[Path]]) -> None:
        """Build master index of all data."""
        logger.info("Building master index of all data files...")
        
        # Analyze embedding files
        embedding_analysis = []
        for filepath in files["embedding_files"]:
            analysis = self.analyze_embedding_file(filepath)
            embedding_analysis.append(analysis)
            logger.info(f"Analyzed {filepath.name}: {analysis['record_count']} records, {analysis.get('model', 'unknown')} model")
        
        # Sort by model and date for each model
        embedding_analysis.sort(key=lambda x: (x.get('model') or 'zzz', x.get('date') or '', -x.get('record_count', 0)))
        
        # Group by model
        embeddings_by_model = defaultdict(list)
        for analysis in embedding_analysis:
            model = analysis.get('model', 'unknown')
            embeddings_by_model[model].append(analysis)
        
        # Store in master index
        self.master_index["embeddings_by_model"] = dict(embeddings_by_model)
        self.master_index["all_embedding_files"] = embedding_analysis
        
        # Analyze raw patent files
        patent_files = []
        for filepath in files["raw_patent_files"]:
            count = self.count_records_in_file(filepath)
            patent_files.append({
                "filepath": str(filepath),
                "filename": filepath.name,
                "record_count": count,
                "file_size_mb": filepath.stat().st_size / (1024 * 1024)
            })
        self.master_index["raw_patent_files"] = patent_files
        
        # Analyze ground truth files
        ground_truth_files = []
        for filepath in files["ground_truth_files"]:
            if "partial" not in filepath.name:  # Skip partial files
                count = self.count_records_in_file(filepath)
                ground_truth_files.append({
                    "filepath": str(filepath),
                    "filename": filepath.name,
                    "record_count": count,
                    "file_size_mb": filepath.stat().st_size / (1024 * 1024)
                })
        self.master_index["ground_truth_files"] = ground_truth_files
        
        # Calculate statistics
        stats = {
            "total_embedding_files": len(files["embedding_files"]),
            "total_ground_truth_files": len([f for f in files["ground_truth_files"] if "partial" not in f.name]),
            "total_raw_files": len(files["raw_patent_files"]),
            "models_with_embeddings": list(embeddings_by_model.keys()),
            "largest_embedding_file": max(embedding_analysis, key=lambda x: x.get('record_count', 0), default={}),
            "total_patents_with_embeddings": {}
        }
        
        # Count unique patents per model
        for model, file_list in embeddings_by_model.items():
            total_records = sum(f.get('record_count', 0) for f in file_list)
            stats["total_patents_with_embeddings"][model] = total_records
        
        self.master_index["statistics"] = stats
    
    def identify_consolidation_opportunities(self) -> Dict[str, Any]:
        """Identify files that can be consolidated or are duplicates."""
        opportunities = {
            "potential_duplicates": [],
            "partial_runs": [],
            "consolidation_candidates": {},
            "archival_candidates": []
        }
        
        # Group files by model
        embeddings_by_model = self.master_index.get("embeddings_by_model", {})
        
        for model, file_list in embeddings_by_model.items():
            if len(file_list) > 1:
                # Sort by record count (largest first)
                file_list.sort(key=lambda x: x.get('record_count', 0), reverse=True)
                
                # The largest file is the consolidation target
                target_file = file_list[0]
                candidate_files = file_list[1:]
                
                opportunities["consolidation_candidates"][model] = {
                    "target": target_file,
                    "candidates": candidate_files,
                    "total_records": sum(f.get('record_count', 0) for f in file_list),
                    "potential_savings": sum(f.get('file_size_mb', 0) for f in candidate_files)
                }
                
                # Identify partial runs (files with significantly fewer records)
                max_records = target_file.get('record_count', 0)
                for file_info in candidate_files:
                    if file_info.get('record_count', 0) < max_records * 0.8:
                        opportunities["partial_runs"].append(file_info)
        
        return opportunities
    
    def generate_organization_plan(self) -> Dict[str, Any]:
        """Generate detailed plan for organizing files."""
        plan = {
            "raw_files": {"moves": []},
            "embeddings": {"moves": [], "consolidations": []},
            "ground_truth": {"moves": []},
            "evaluations": {"moves": []},
            "archive": {"moves": []}
        }
        
        # Plan raw file moves
        for file_info in self.master_index.get("raw_patent_files", []):
            src_path = Path(file_info["filepath"])
            if "100k" in file_info["filename"]:
                dst_path = self.data_root / "raw" / "patents_100k.jsonl"
            elif "10k" in file_info["filename"]:
                dst_path = self.data_root / "raw" / "patents_10k.jsonl" 
            else:
                dst_path = self.data_root / "raw" / file_info["filename"]
            
            if src_path != dst_path:
                plan["raw_files"]["moves"].append({
                    "src": str(src_path),
                    "dst": str(dst_path),
                    "reason": "organize raw data"
                })
        
        # Plan embedding file organization
        embeddings_by_model = self.master_index.get("embeddings_by_model", {})
        opportunities = self.identify_consolidation_opportunities()
        
        for model, file_list in embeddings_by_model.items():
            # Skip files with unknown models
            if model in [None, "unknown", "zzz"]:
                continue
                
            # Move current files to model directories
            for file_info in file_list:
                src_path = Path(file_info["filepath"])
                dst_path = self.data_root / "embeddings" / "by_model" / model / file_info["filename"]
                
                if src_path != dst_path:
                    plan["embeddings"]["moves"].append({
                        "src": str(src_path),
                        "dst": str(dst_path),
                        "model": model,
                        "records": file_info.get("record_count", 0),
                        "reason": f"organize {model} embeddings"
                    })
        
        # Plan consolidations
        for model, consolidation_info in opportunities.get("consolidation_candidates", {}).items():
            target = consolidation_info["target"]
            candidates = consolidation_info["candidates"]
            
            plan["embeddings"]["consolidations"].append({
                "model": model,
                "target_file": target["filepath"],
                "candidate_files": [f["filepath"] for f in candidates],
                "estimated_total_records": consolidation_info["total_records"],
                "potential_space_savings_mb": consolidation_info["potential_savings"]
            })
        
        # Plan ground truth moves
        for file_info in self.master_index.get("ground_truth_files", []):
            src_path = Path(file_info["filepath"])
            if "500" in file_info["filename"]:
                dst_path = self.data_root / "ground_truth" / "consolidated" / "ground_truth_500.jsonl"
            elif "100" in file_info["filename"]:
                dst_path = self.data_root / "ground_truth" / "consolidated" / "ground_truth_100.jsonl"
            else:
                dst_path = self.data_root / "ground_truth" / "consolidated" / file_info["filename"]
            
            if src_path != dst_path:
                plan["ground_truth"]["moves"].append({
                    "src": str(src_path),
                    "dst": str(dst_path),
                    "reason": "consolidate ground truth"
                })
        
        return plan
    
    def save_master_index(self):
        """Save master index to file."""
        index_path = self.data_root / "metadata" / "master_index.json"
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(self.master_index, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Master index saved to {index_path}")
    
    def save_organization_plan(self, plan: Dict[str, Any]):
        """Save organization plan to file."""
        plan_path = self.data_root / "metadata" / "organization_plan.json"
        with open(plan_path, 'w', encoding='utf-8') as f:
            json.dump(plan, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Organization plan saved to {plan_path}")
    
    def print_summary(self):
        """Print summary of discovered data."""
        stats = self.master_index.get("statistics", {})
        
        print("\n" + "="*60)
        print("DATA CONSOLIDATION ANALYSIS")
        print("="*60)
        
        print(f"📁 Total embedding files: {stats.get('total_embedding_files', 0)}")
        print(f"📊 Total ground truth files: {stats.get('total_ground_truth_files', 0)}")
        print(f"📄 Total raw patent files: {stats.get('total_raw_files', 0)}")
        
        models_list = [m for m in stats.get('models_with_embeddings', []) if m is not None]
        print(f"\n🤖 Models with embeddings: {', '.join(models_list)}")
        
        print(f"\n📈 Patents with embeddings per model:")
        for model, count in stats.get('total_patents_with_embeddings', {}).items():
            print(f"  - {model}: {count:,} patents")
        
        largest = stats.get('largest_embedding_file', {})
        if largest:
            print(f"\n🏆 Largest embedding file: {largest.get('filename', 'unknown')}")
            print(f"   Records: {largest.get('record_count', 0):,}")
            print(f"   Model: {largest.get('model', 'unknown')}")
        
        # Show consolidation opportunities
        opportunities = self.identify_consolidation_opportunities()
        
        if opportunities.get('consolidation_candidates'):
            print(f"\n💡 Consolidation opportunities:")
            for model, info in opportunities['consolidation_candidates'].items():
                print(f"  - {model}: {len(info['candidates'])} files can be merged")
                print(f"    Potential space savings: {info['potential_savings']:.1f} MB")
        
        print("="*60)


def main():
    """Main consolidation workflow."""
    consolidator = EmbeddingConsolidator()
    
    # Discover all files
    logger.info("Discovering all data files...")
    files = consolidator.discover_all_files()
    
    # Build master index
    consolidator.build_master_index(files)
    
    # Generate organization plan
    logger.info("Generating organization plan...")
    plan = consolidator.generate_organization_plan()
    
    # Save results
    consolidator.save_master_index()
    consolidator.save_organization_plan(plan)
    
    # Print summary
    consolidator.print_summary()
    
    return consolidator, plan


if __name__ == "__main__":
    main()