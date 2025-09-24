"""Orchestrator for running comprehensive multi-model embedding experiments."""

import argparse
import json
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from experiment_tracker import ExperimentTracker


class MultiModelExperimentOrchestrator:
    """Orchestrate multiple embedding experiments across models and datasets."""
    
    def __init__(self):
        self.tracker = ExperimentTracker()
        self.default_models = [
            "embeddinggemma",
            "bge-m3", 
            "mxbai-embed-large",
            "nomic-embed-text"
        ]
        self.default_strategies = ["fixed", "overlapping", "sentence"]
        self.default_aggregations = ["mean", "max", "weighted"]
    
    def check_model_availability(self, models: List[str] = None) -> Dict[str, bool]:
        """Check which models are available in Ollama."""
        if models is None:
            models = self.default_models
        
        print("Checking model availability...")
        availability = {}
        
        for model in models:
            try:
                # Try to get model info
                result = subprocess.run(
                    ["ollama", "show", model],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                availability[model] = result.returncode == 0
                status = "✅ Available" if availability[model] else "❌ Not found"
                print(f"  {model}: {status}")
                
            except subprocess.TimeoutExpired:
                availability[model] = False
                print(f"  {model}: ❌ Timeout")
            except Exception as e:
                availability[model] = False
                print(f"  {model}: ❌ Error: {e}")
        
        available_models = [m for m, avail in availability.items() if avail]
        print(f"\nAvailable models: {len(available_models)}/{len(models)}")
        
        return availability
    
    def pull_missing_models(self, models: List[str]) -> Dict[str, bool]:
        """Attempt to pull missing models."""
        availability = self.check_model_availability(models)
        missing_models = [m for m, avail in availability.items() if not avail]
        
        if not missing_models:
            return availability
        
        print(f"\nAttempting to pull {len(missing_models)} missing models...")
        
        for model in missing_models:
            print(f"Pulling {model}...")
            try:
                result = subprocess.run(
                    ["ollama", "pull", model],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                
                if result.returncode == 0:
                    availability[model] = True
                    print(f"  ✅ Successfully pulled {model}")
                else:
                    print(f"  ❌ Failed to pull {model}: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                print(f"  ⏰ Timeout pulling {model}")
            except Exception as e:
                print(f"  ❌ Error pulling {model}: {e}")
        
        return availability
    
    def create_experiment_batch(
        self,
        batch_name: str,
        dataset_file: str,
        models: List[str] = None,
        max_records_per_experiment: int = 100,
        description: str = ""
    ) -> List[str]:
        """Create a batch of experiments for different models."""
        
        if models is None:
            models = self.default_models
        
        # Check model availability
        availability = self.check_model_availability(models)
        available_models = [m for m, avail in availability.items() if avail]
        
        if not available_models:
            print("❌ No models available. Please install models first.")
            return []
        
        experiment_ids = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"\nCreating experiment batch: {batch_name}")
        print(f"Dataset: {dataset_file}")
        print(f"Models: {len(available_models)}")
        print(f"Max records per experiment: {max_records_per_experiment}")
        
        for model in available_models:
            experiment_id = f"{batch_name}_{model}_{timestamp}"
            
            try:
                experiment = self.tracker.create_experiment(
                    experiment_id=experiment_id,
                    dataset_file=dataset_file,
                    model_name=model,
                    max_records=max_records_per_experiment,
                    description=f"{description} - Model: {model}, Batch: {batch_name}"
                )
                
                experiment_ids.append(experiment_id)
                print(f"  ✅ Created experiment: {experiment_id}")
                
            except Exception as e:
                print(f"  ❌ Failed to create experiment for {model}: {e}")
        
        print(f"\nCreated {len(experiment_ids)} experiments in batch '{batch_name}'")
        return experiment_ids
    
    def run_experiment_batch(
        self,
        experiment_ids: List[str],
        parallel: bool = False,
        max_parallel: int = 2
    ) -> Dict[str, Any]:
        """Run a batch of experiments."""
        
        print(f"\n🚀 Running {len(experiment_ids)} experiments...")
        if parallel:
            print(f"   Running up to {max_parallel} experiments in parallel")
        else:
            print("   Running experiments sequentially")
        
        results = {
            "batch_start_time": time.time(),
            "experiments": {},
            "summary": {}
        }
        
        if parallel:
            # TODO: Implement parallel execution
            print("⚠️  Parallel execution not implemented yet, running sequentially")
            parallel = False
        
        # Sequential execution
        for i, experiment_id in enumerate(experiment_ids, 1):
            print(f"\n📊 Running experiment {i}/{len(experiment_ids)}: {experiment_id}")
            
            experiment = self.tracker.get_experiment(experiment_id)
            if not experiment:
                print(f"❌ Experiment {experiment_id} not found")
                results["experiments"][experiment_id] = {"error": "Experiment not found"}
                continue
            
            try:
                # Run the experiment
                start_time = time.time()
                
                output_file = f"{experiment.config.output_prefix}_embeddings.jsonl"
                
                # Use multi-model generator
                from generate_embeddings_multimodel import process_patents_multimodel
                
                process_patents_multimodel(
                    input_file=experiment.config.dataset_file,
                    output_file=output_file,
                    models=[experiment.config.model_config.name],
                    max_records=experiment.config.max_records,
                    experiment_id=experiment_id
                )
                
                execution_time = time.time() - start_time
                
                results["experiments"][experiment_id] = {
                    "status": "completed",
                    "execution_time": execution_time,
                    "output_file": output_file
                }
                
                print(f"  ✅ Completed in {execution_time/60:.1f} minutes")
                
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                results["experiments"][experiment_id] = {
                    "status": "failed",
                    "error": str(e)
                }
                
                # Mark experiment as failed
                self.tracker.complete_experiment(experiment_id, error_message=str(e))
        
        # Generate summary
        total_time = time.time() - results["batch_start_time"]
        completed = sum(1 for r in results["experiments"].values() if r.get("status") == "completed")
        failed = sum(1 for r in results["experiments"].values() if r.get("status") == "failed")
        
        results["summary"] = {
            "total_experiments": len(experiment_ids),
            "completed": completed,
            "failed": failed,
            "total_time_minutes": total_time / 60,
            "average_time_per_experiment": total_time / len(experiment_ids) / 60
        }
        
        return results
    
    def run_comprehensive_comparison(
        self,
        dataset_file: str,
        batch_name: str = "comprehensive",
        models: List[str] = None,
        max_records: int = 100,
        include_analysis: bool = True
    ) -> None:
        """Run comprehensive comparison across multiple models."""
        
        print("🔬 COMPREHENSIVE MULTI-MODEL COMPARISON")
        print("=" * 60)
        
        if models is None:
            models = self.default_models
        
        # Check and pull models
        print("Step 1: Model Setup")
        availability = self.pull_missing_models(models)
        available_models = [m for m, avail in availability.items() if avail]
        
        if not available_models:
            print("❌ No models available after pulling attempts")
            return
        
        print(f"✅ Using {len(available_models)} models: {', '.join(available_models)}")
        
        # Create experiment batch
        print("\nStep 2: Creating Experiments")
        experiment_ids = self.create_experiment_batch(
            batch_name=batch_name,
            dataset_file=dataset_file,
            models=available_models,
            max_records_per_experiment=max_records,
            description=f"Comprehensive comparison with {max_records} patents per model"
        )
        
        if not experiment_ids:
            print("❌ No experiments created")
            return
        
        # Run experiments
        print("\nStep 3: Running Experiments")
        batch_results = self.run_experiment_batch(experiment_ids)
        
        # Save batch results
        batch_results_file = f"{batch_name}_batch_results.json"
        with open(batch_results_file, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 Batch Results Summary:")
        print(f"   Completed: {batch_results['summary']['completed']}")
        print(f"   Failed: {batch_results['summary']['failed']}")
        print(f"   Total time: {batch_results['summary']['total_time_minutes']:.1f} minutes")
        print(f"   Results saved to: {batch_results_file}")
        
        if include_analysis and batch_results['summary']['completed'] > 0:
            print("\nStep 4: Cross-Model Analysis")
            self.analyze_batch_results(experiment_ids, batch_name)
    
    def analyze_batch_results(self, experiment_ids: List[str], batch_name: str) -> None:
        """Analyze results across multiple experiments."""
        print("Generating cross-model analysis...")
        
        # Collect results from all experiments
        all_results = []
        
        for experiment_id in experiment_ids:
            experiment = self.tracker.get_experiment(experiment_id)
            if not experiment or experiment.status.value != "completed":
                continue
            
            # Load experiment results
            output_file = experiment.output_files.get("embeddings")
            if output_file and Path(output_file).exists():
                with open(output_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        result = json.loads(line.strip())
                        result["experiment_id"] = experiment_id
                        result["model"] = experiment.config.model_config.name
                        all_results.append(result)
        
        if not all_results:
            print("❌ No results found for analysis")
            return
        
        # Generate comparison report
        comparison_file = f"{batch_name}_model_comparison.json"
        
        comparison_data = {
            "batch_name": batch_name,
            "timestamp": datetime.now().isoformat(),
            "experiments": experiment_ids,
            "total_results": len(all_results),
            "models_compared": list(set(r["model"] for r in all_results)),
            "results": all_results
        }
        
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Cross-model analysis saved to: {comparison_file}")
        
        # Generate summary statistics
        model_stats = {}
        for result in all_results:
            model = result["model"]
            if model not in model_stats:
                model_stats[model] = {
                    "count": 0,
                    "avg_processing_time": 0,
                    "chunking_required": 0,
                    "embedding_dims": set()
                }
            
            model_stats[model]["count"] += 1
            
            for model_name, model_data in result.get("models", {}).items():
                if model_data.get("needs_chunking"):
                    model_stats[model]["chunking_required"] += 1
                
                original_emb = model_data.get("embeddings", {}).get("original", {})
                if "embedding_dim" in original_emb:
                    model_stats[model]["embedding_dims"].add(original_emb["embedding_dim"])
        
        # Print summary
        print(f"\n📈 Model Comparison Summary:")
        print(f"{'Model':<20} {'Patents':<10} {'Avg Time':<12} {'Need Chunking':<15} {'Embed Dim'}")
        print("-" * 75)
        
        for model, stats in model_stats.items():
            chunking_pct = (stats["chunking_required"] / stats["count"]) * 100 if stats["count"] > 0 else 0
            embed_dim = list(stats["embedding_dims"])[0] if stats["embedding_dims"] else "Unknown"
            
            print(f"{model:<20} {stats['count']:<10} {'N/A':<12} {chunking_pct:<13.1f}% {embed_dim}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive multi-model embedding experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check model availability
  python run_multimodel_experiments.py status
  
  # Pull missing models
  python run_multimodel_experiments.py pull-models
  
  # Run comprehensive comparison
  python run_multimodel_experiments.py compare patent_abstracts.jsonl --batch-name test_batch --max-records 50
  
  # Create experiment batch only
  python run_multimodel_experiments.py create-batch patent_abstracts.jsonl my_batch --models embeddinggemma,bge-m3
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Status command
    subparsers.add_parser('status', help='Show experiment tracker status')
    
    # Pull models command
    pull_parser = subparsers.add_parser('pull-models', help='Pull missing models')
    pull_parser.add_argument('--models', help='Comma-separated model names')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Run comprehensive comparison')
    compare_parser.add_argument('dataset_file', help='Input dataset file')
    compare_parser.add_argument('--batch-name', default='comprehensive', help='Batch name')
    compare_parser.add_argument('--models', help='Comma-separated model names')
    compare_parser.add_argument('--max-records', type=int, default=100, help='Max records per experiment')
    compare_parser.add_argument('--no-analysis', action='store_true', help='Skip cross-model analysis')
    
    # Create batch command
    batch_parser = subparsers.add_parser('create-batch', help='Create experiment batch')
    batch_parser.add_argument('dataset_file', help='Input dataset file')
    batch_parser.add_argument('batch_name', help='Batch name')
    batch_parser.add_argument('--models', help='Comma-separated model names')
    batch_parser.add_argument('--max-records', type=int, default=100, help='Max records per experiment')
    
    args = parser.parse_args()
    
    orchestrator = MultiModelExperimentOrchestrator()
    
    if args.command == 'status':
        print(orchestrator.tracker.generate_status_report())
    
    elif args.command == 'pull-models':
        models = args.models.split(',') if args.models else orchestrator.default_models
        orchestrator.pull_missing_models(models)
    
    elif args.command == 'compare':
        if not Path(args.dataset_file).exists():
            print(f"Error: Dataset file {args.dataset_file} not found")
            return
        
        models = args.models.split(',') if args.models else None
        orchestrator.run_comprehensive_comparison(
            dataset_file=args.dataset_file,
            batch_name=args.batch_name,
            models=models,
            max_records=args.max_records,
            include_analysis=not args.no_analysis
        )
    
    elif args.command == 'create-batch':
        if not Path(args.dataset_file).exists():
            print(f"Error: Dataset file {args.dataset_file} not found")
            return
        
        models = args.models.split(',') if args.models else None
        experiment_ids = orchestrator.create_experiment_batch(
            batch_name=args.batch_name,
            dataset_file=args.dataset_file,
            models=models,
            max_records_per_experiment=args.max_records
        )
        
        print(f"\n✅ Created batch '{args.batch_name}' with {len(experiment_ids)} experiments")
        print("Use the following command to run the batch:")
        print(f"python run_multimodel_experiments.py run-batch {' '.join(experiment_ids)}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()