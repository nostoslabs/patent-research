"""Comprehensive experiment progress tracking system."""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum


class ExperimentStatus(Enum):
    """Experiment status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ModelConfig:
    """Configuration for embedding models."""
    name: str
    context_limit: int  # in tokens
    embedding_dim: int
    char_per_token: float = 4.0  # rough approximation
    
    @property
    def char_limit(self) -> int:
        return int(self.context_limit * self.char_per_token)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    experiment_id: str
    dataset_file: str
    model_config: ModelConfig
    chunking_strategies: List[str]
    aggregation_methods: List[str]
    max_records: Optional[int] = None
    output_prefix: str = "experiment"
    description: str = ""


@dataclass
class ProgressMetrics:
    """Progress tracking metrics."""
    total_patents: int
    processed_patents: int
    successful_patents: int
    failed_patents: int
    start_time: float
    current_time: float
    estimated_completion: Optional[float] = None
    
    @property
    def progress_percent(self) -> float:
        return (self.processed_patents / self.total_patents) * 100 if self.total_patents > 0 else 0
    
    @property
    def elapsed_time(self) -> float:
        return self.current_time - self.start_time
    
    @property
    def rate_per_second(self) -> float:
        return self.processed_patents / self.elapsed_time if self.elapsed_time > 0 else 0
    
    @property
    def estimated_total_time(self) -> float:
        if self.rate_per_second > 0:
            return self.total_patents / self.rate_per_second
        return 0
    
    @property
    def eta_seconds(self) -> float:
        if self.rate_per_second > 0:
            remaining = self.total_patents - self.processed_patents
            return remaining / self.rate_per_second
        return 0


@dataclass
class ExperimentRecord:
    """Complete experiment record."""
    config: ExperimentConfig
    status: ExperimentStatus
    progress: Optional[ProgressMetrics] = None
    created_at: float = 0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error_message: Optional[str] = None
    output_files: Dict[str, str] = None
    results_summary: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at == 0:
            self.created_at = time.time()
        if self.output_files is None:
            self.output_files = {}
        if self.results_summary is None:
            self.results_summary = {}


class ExperimentTracker:
    """Centralized experiment tracking system."""
    
    def __init__(self, tracking_file: str = "experiment_tracking.json"):
        self.tracking_file = Path(tracking_file)
        self.experiments: Dict[str, ExperimentRecord] = {}
        self.model_configs = self._load_default_models()
        self.load_state()
    
    def _load_default_models(self) -> Dict[str, ModelConfig]:
        """Load default model configurations."""
        return {
            "embeddinggemma": ModelConfig("embeddinggemma", 2000, 768, 4.0),
            "bge-m3": ModelConfig("bge-m3", 8192, 1024, 3.5),
            "mxbai-embed-large": ModelConfig("mxbai-embed-large", 512, 1024, 4.0),
            "nomic-embed-text": ModelConfig("nomic-embed-text", 8192, 768, 3.8),
            "all-MiniLM-L6-v2": ModelConfig("all-MiniLM-L6-v2", 512, 384, 4.2),
        }
    
    def add_model_config(self, model_config: ModelConfig) -> None:
        """Add or update model configuration."""
        self.model_configs[model_config.name] = model_config
        self.save_state()
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get model configuration by name."""
        return self.model_configs.get(model_name)
    
    def create_experiment(
        self,
        experiment_id: str,
        dataset_file: str,
        model_name: str,
        chunking_strategies: List[str] = None,
        aggregation_methods: List[str] = None,
        max_records: Optional[int] = None,
        output_prefix: str = None,
        description: str = ""
    ) -> ExperimentRecord:
        """Create a new experiment record."""
        
        if chunking_strategies is None:
            chunking_strategies = [
                "fixed_512", "fixed_768", "overlapping_550",
                "sentence_boundary_512", "sentence_boundary_768",
                "semantic"
            ]
        
        if aggregation_methods is None:
            aggregation_methods = ["mean", "max", "weighted", "attention"]
        
        if output_prefix is None:
            output_prefix = f"{experiment_id}_{model_name}"
        
        model_config = self.get_model_config(model_name)
        if not model_config:
            raise ValueError(f"Unknown model: {model_name}")
        
        config = ExperimentConfig(
            experiment_id=experiment_id,
            dataset_file=dataset_file,
            model_config=model_config,
            chunking_strategies=chunking_strategies,
            aggregation_methods=aggregation_methods,
            max_records=max_records,
            output_prefix=output_prefix,
            description=description
        )
        
        experiment = ExperimentRecord(
            config=config,
            status=ExperimentStatus.PENDING
        )
        
        self.experiments[experiment_id] = experiment
        self.save_state()
        
        return experiment
    
    def start_experiment(self, experiment_id: str, total_patents: int) -> None:
        """Mark experiment as started."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        experiment.status = ExperimentStatus.RUNNING
        experiment.started_at = time.time()
        experiment.progress = ProgressMetrics(
            total_patents=total_patents,
            processed_patents=0,
            successful_patents=0,
            failed_patents=0,
            start_time=time.time(),
            current_time=time.time()
        )
        
        self.save_state()
    
    def update_progress(
        self,
        experiment_id: str,
        processed_patents: int,
        successful_patents: int,
        failed_patents: int
    ) -> None:
        """Update experiment progress."""
        if experiment_id not in self.experiments:
            return
        
        experiment = self.experiments[experiment_id]
        if experiment.progress:
            experiment.progress.processed_patents = processed_patents
            experiment.progress.successful_patents = successful_patents
            experiment.progress.failed_patents = failed_patents
            experiment.progress.current_time = time.time()
            
            # Update ETA
            if experiment.progress.rate_per_second > 0:
                experiment.progress.estimated_completion = (
                    time.time() + experiment.progress.eta_seconds
                )
        
        self.save_state()
    
    def complete_experiment(
        self,
        experiment_id: str,
        output_files: Dict[str, str] = None,
        results_summary: Dict[str, Any] = None,
        error_message: str = None
    ) -> None:
        """Mark experiment as completed or failed."""
        if experiment_id not in self.experiments:
            return
        
        experiment = self.experiments[experiment_id]
        experiment.completed_at = time.time()
        
        if error_message:
            experiment.status = ExperimentStatus.FAILED
            experiment.error_message = error_message
        else:
            experiment.status = ExperimentStatus.COMPLETED
            if output_files:
                experiment.output_files.update(output_files)
            if results_summary:
                experiment.results_summary.update(results_summary)
        
        self.save_state()
    
    def get_experiment(self, experiment_id: str) -> Optional[ExperimentRecord]:
        """Get experiment by ID."""
        return self.experiments.get(experiment_id)
    
    def list_experiments(
        self,
        status: Optional[ExperimentStatus] = None,
        model_name: Optional[str] = None
    ) -> List[ExperimentRecord]:
        """List experiments with optional filtering."""
        experiments = list(self.experiments.values())
        
        if status:
            experiments = [e for e in experiments if e.status == status]
        
        if model_name:
            experiments = [e for e in experiments if e.config.model_config.name == model_name]
        
        # Sort by creation time (newest first)
        experiments.sort(key=lambda x: x.created_at, reverse=True)
        return experiments
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get overall progress summary."""
        running_experiments = self.list_experiments(ExperimentStatus.RUNNING)
        
        summary = {
            "total_experiments": len(self.experiments),
            "by_status": {},
            "running_experiments": [],
            "recent_completions": []
        }
        
        # Count by status
        for status in ExperimentStatus:
            count = len(self.list_experiments(status))
            summary["by_status"][status.value] = count
        
        # Running experiments details
        for exp in running_experiments:
            if exp.progress:
                exp_summary = {
                    "experiment_id": exp.config.experiment_id,
                    "model": exp.config.model_config.name,
                    "progress_percent": round(exp.progress.progress_percent, 1),
                    "eta_minutes": round(exp.progress.eta_seconds / 60, 1),
                    "rate_per_minute": round(exp.progress.rate_per_second * 60, 1),
                    "elapsed_minutes": round(exp.progress.elapsed_time / 60, 1)
                }
                summary["running_experiments"].append(exp_summary)
        
        # Recent completions (last 24 hours)
        recent_cutoff = time.time() - (24 * 60 * 60)
        completed = self.list_experiments(ExperimentStatus.COMPLETED)
        recent_completed = [
            exp for exp in completed 
            if exp.completed_at and exp.completed_at > recent_cutoff
        ]
        
        for exp in recent_completed[:5]:  # Last 5
            exp_summary = {
                "experiment_id": exp.config.experiment_id,
                "model": exp.config.model_config.name,
                "completed_ago_hours": round((time.time() - exp.completed_at) / 3600, 1),
                "total_time_minutes": round((exp.completed_at - exp.started_at) / 60, 1) if exp.started_at else None
            }
            summary["recent_completions"].append(exp_summary)
        
        return summary
    
    def save_state(self) -> None:
        """Save tracking state to JSON file."""
        # Convert to serializable format
        serializable_data = {
            "experiments": {},
            "model_configs": {}
        }
        
        for exp_id, experiment in self.experiments.items():
            serializable_data["experiments"][exp_id] = {
                "config": {
                    "experiment_id": experiment.config.experiment_id,
                    "dataset_file": experiment.config.dataset_file,
                    "model_config": asdict(experiment.config.model_config),
                    "chunking_strategies": experiment.config.chunking_strategies,
                    "aggregation_methods": experiment.config.aggregation_methods,
                    "max_records": experiment.config.max_records,
                    "output_prefix": experiment.config.output_prefix,
                    "description": experiment.config.description
                },
                "status": experiment.status.value,
                "progress": asdict(experiment.progress) if experiment.progress else None,
                "created_at": experiment.created_at,
                "started_at": experiment.started_at,
                "completed_at": experiment.completed_at,
                "error_message": experiment.error_message,
                "output_files": experiment.output_files,
                "results_summary": experiment.results_summary
            }
        
        for model_name, model_config in self.model_configs.items():
            serializable_data["model_configs"][model_name] = asdict(model_config)
        
        with open(self.tracking_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
    
    def load_state(self) -> None:
        """Load tracking state from JSON file."""
        if not self.tracking_file.exists():
            return
        
        try:
            with open(self.tracking_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load model configs
            if "model_configs" in data:
                for model_name, model_data in data["model_configs"].items():
                    self.model_configs[model_name] = ModelConfig(**model_data)
            
            # Load experiments
            if "experiments" in data:
                for exp_id, exp_data in data["experiments"].items():
                    config_data = exp_data["config"]
                    model_config = ModelConfig(**config_data["model_config"])
                    
                    config = ExperimentConfig(
                        experiment_id=config_data["experiment_id"],
                        dataset_file=config_data["dataset_file"],
                        model_config=model_config,
                        chunking_strategies=config_data["chunking_strategies"],
                        aggregation_methods=config_data["aggregation_methods"],
                        max_records=config_data.get("max_records"),
                        output_prefix=config_data["output_prefix"],
                        description=config_data.get("description", "")
                    )
                    
                    progress = None
                    if exp_data.get("progress"):
                        progress = ProgressMetrics(**exp_data["progress"])
                    
                    experiment = ExperimentRecord(
                        config=config,
                        status=ExperimentStatus(exp_data["status"]),
                        progress=progress,
                        created_at=exp_data.get("created_at", 0),
                        started_at=exp_data.get("started_at"),
                        completed_at=exp_data.get("completed_at"),
                        error_message=exp_data.get("error_message"),
                        output_files=exp_data.get("output_files", {}),
                        results_summary=exp_data.get("results_summary", {})
                    )
                    
                    self.experiments[exp_id] = experiment
        
        except Exception as e:
            print(f"Warning: Could not load tracking state: {e}")
    
    def generate_status_report(self) -> str:
        """Generate a human-readable status report."""
        summary = self.get_progress_summary()
        
        report = []
        report.append("# Experiment Progress Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Experiments: {summary['total_experiments']}")
        
        # Status breakdown
        report.append("\n## 📊 Status Summary")
        for status, count in summary["by_status"].items():
            emoji = {
                "pending": "⏳",
                "running": "🔄", 
                "completed": "✅",
                "failed": "❌",
                "cancelled": "🚫"
            }.get(status, "📋")
            report.append(f"{emoji} {status.title()}: {count}")
        
        # Running experiments
        if summary["running_experiments"]:
            report.append("\n## 🔄 Currently Running")
            report.append(f"{'Experiment':<25} {'Model':<20} {'Progress':<10} {'ETA':<10} {'Rate/min'}")
            report.append("-" * 80)
            
            for exp in summary["running_experiments"]:
                report.append(
                    f"{exp['experiment_id']:<25} {exp['model']:<20} "
                    f"{exp['progress_percent']:>7.1f}% {exp['eta_minutes']:>8.1f}m "
                    f"{exp['rate_per_minute']:>7.1f}"
                )
        
        # Recent completions
        if summary["recent_completions"]:
            report.append("\n## ✅ Recent Completions (24h)")
            report.append(f"{'Experiment':<25} {'Model':<20} {'Completed':<12} {'Duration'}")
            report.append("-" * 75)
            
            for exp in summary["recent_completions"]:
                duration = f"{exp['total_time_minutes']:.1f}m" if exp['total_time_minutes'] else "N/A"
                report.append(
                    f"{exp['experiment_id']:<25} {exp['model']:<20} "
                    f"{exp['completed_ago_hours']:>9.1f}h {duration:>10}"
                )
        
        # Model configurations
        report.append("\n## 🤖 Available Models")
        report.append(f"{'Model':<25} {'Context Limit':<15} {'Embedding Dim':<15} {'Est. Char Limit'}")
        report.append("-" * 80)
        
        for model_name, model_config in self.model_configs.items():
            report.append(
                f"{model_name:<25} {model_config.context_limit:<15} "
                f"{model_config.embedding_dim:<15} {model_config.char_limit:>10}"
            )
        
        return "\n".join(report)


def main() -> None:
    """Command-line interface for experiment tracking."""
    import sys
    
    tracker = ExperimentTracker()
    
    if len(sys.argv) < 2:
        print("Usage: python experiment_tracker.py [status|create|list]")
        return
    
    command = sys.argv[1]
    
    if command == "status":
        print(tracker.generate_status_report())
    
    elif command == "list":
        experiments = tracker.list_experiments()
        print(f"Found {len(experiments)} experiments:")
        for exp in experiments:
            print(f"  {exp.config.experiment_id}: {exp.status.value} ({exp.config.model_config.name})")
    
    elif command == "create":
        if len(sys.argv) < 5:
            print("Usage: python experiment_tracker.py create <experiment_id> <dataset_file> <model_name>")
            return
        
        experiment_id = sys.argv[2]
        dataset_file = sys.argv[3]
        model_name = sys.argv[4]
        
        try:
            experiment = tracker.create_experiment(experiment_id, dataset_file, model_name)
            print(f"Created experiment: {experiment_id}")
        except Exception as e:
            print(f"Error creating experiment: {e}")
    
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()