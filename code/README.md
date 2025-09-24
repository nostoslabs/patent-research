# Code Organization

This directory contains all Python scripts and tools for the patent research project, organized by functionality.

## Directory Structure

### 🔧 Core (`core/`)
Core system components and main functionality:
- `llm_provider_factory.py` - Multi-provider LLM system with PydanticAI
- `cross_encoder_reranker.py` - Two-stage retrieval implementation
- `reranker_enhancement_plan.py` - Advanced multi-reranker system
- `semantic_search.py` - Semantic search functionality

### 📊 Data Processing (`data_processing/`)
Data ingestion, transformation, and preparation:
- `download_patents.py` - Patent data download utilities
- `download_large_diverse_patents.py` - Large-scale patent dataset creation
- `convert_to_jsonl.py` / `convert_to_parquet.py` - Data format conversion
- `consolidate_embeddings.py` / `data_consolidation.py` - Data consolidation
- `create_atlas_data.py` / `create_better_atlas_data.py` - Atlas visualization data
- `generate_embeddings*.py` - Embedding generation scripts
- `generate_keywords_with_gemini.py` - Keyword extraction

### 📈 Analysis (`analysis/`)
Scientific analysis and evaluation:
- `scientific_analysis.py` - Statistical analysis and visualization
- `model_performance_analyzer.py` - Model performance comparison
- `model_intersection_analysis.py` - Model intersection analysis
- `comprehensive_evaluation.py` - End-to-end evaluation pipeline
- `practical_search_analysis.py` - Practical search evaluation
- `realistic_search_analysis.py` - Realistic search scenarios
- `manager_focused_analysis.py` - Manager-focused analysis

### 🔬 Experiments (`experiments/`)
Experimental workflows and ground truth generation:
- `run_multimodel_experiments.py` - Multi-model experiment runner
- `run_chunking_experiment.py` - Chunking experiment runner
- `experiment_tracker.py` - Experiment tracking and monitoring
- `ground_truth_generator.py` - LLM-based ground truth generation
- `ground_truth_batch_generator.py` - Batch ground truth generation
- `ground_truth_quality_evaluator.py` - Ground truth quality assessment
- `openai_ground_truth_evaluation.py` - OpenAI-based evaluation

### 📊 Benchmarks (`benchmarks/`)
Performance benchmarking and comparison:
- `quick_benchmark.py` - Quick performance tests
- `comprehensive_embedding_benchmark.py` - Comprehensive embedding benchmarks
- `fair_comparison_benchmark.py` - Fair comparison benchmarks
- `three_way_fair_comparison.py` - Three-way comparison
- `openai_baseline_comparison.py` - OpenAI baseline comparison
- `openai_direct_comparison.py` - Direct OpenAI comparison
- `google_patents_baseline.py` - Google Patents baseline

### 📊 Monitoring (`monitoring/`)
System monitoring and progress tracking:
- `monitor_all_batches.py` - Batch processing monitoring
- `monitor_batch_job.py` - Individual batch monitoring
- `monitor_experiments.py` - Experiment monitoring
- `monitor_ground_truth_progress.py` - Ground truth progress tracking
- `monitor_openai_batch.py` - OpenAI batch monitoring
- `simple_process_monitor.py` - Simple process monitoring
- `simple_progress_monitor.py` - Progress monitoring

### 🛠️ Utilities (`utilities/`)
Utility scripts and helper tools:
- `batch_manager.py` - Batch processing management
- `gemini_batch_client.py` - Gemini batch client
- `submit_openai_batch.py` - OpenAI batch submission
- `process_openai_batch_results.py` - OpenAI batch result processing
- `filter_openai_batch.py` / `fix_openai_batch.py` - OpenAI batch utilities
- `openai_batch_embeddings.py` - OpenAI batch embeddings
- `create_visualizations.py` - Visualization creation
- `visualize_embeddings.py` / `visualize_chunking_analysis.py` - Visualization tools
- `plot_classification_distribution.py` - Classification plotting
- `dimensionality_reduction*.py` - Dimensionality reduction tools
- `analyze_chunking_performance.py` / `analyze_classifications.py` - Analysis utilities
- `auto_complete_benchmark.py` - Auto-complete benchmarking
- `check_patent_data_sources.py` - Data source validation
- `continuous_embedding_job.py` - Continuous embedding jobs
- `empirical_openai_evaluation.py` - Empirical OpenAI evaluation
- `find_missing_bge_embeddings.py` / `fix_missing_classifications.py` - Data fixing utilities
- `generate_report.py` - Report generation
- `launch_atlas.py` - Atlas launch utility
- `process_all_chunks.py` - Chunk processing
- `main.py` - Main entry point

## Usage

### Running Core Components
```bash
# Semantic search
python core/semantic_search.py

# Reranking system
python core/cross_encoder_reranker.py

# Multi-provider LLM system
python core/llm_provider_factory.py
```

### Data Processing
```bash
# Download patents
python data_processing/download_large_diverse_patents.py 10k

# Generate embeddings
python data_processing/generate_embeddings_multimodel.py

# Consolidate data
python data_processing/data_consolidation.py
```

### Running Experiments
```bash
# Multi-model experiments
python experiments/run_multimodel_experiments.py

# Ground truth generation
python experiments/ground_truth_generator.py

# Chunking experiments
python experiments/run_chunking_experiment.py
```

### Analysis
```bash
# Scientific analysis
python analysis/scientific_analysis.py

# Model performance analysis
python analysis/model_performance_analyzer.py

# Comprehensive evaluation
python analysis/comprehensive_evaluation.py
```

### Monitoring
```bash
# Monitor experiments
python monitoring/monitor_experiments.py

# Monitor ground truth progress
python monitoring/monitor_ground_truth_progress.py
```

## Organization Benefits

- **Clear separation of concerns** - Each directory has a specific purpose
- **Easy navigation** - Find scripts by functionality
- **Logical grouping** - Related scripts are together
- **Scalable structure** - Easy to add new scripts in appropriate categories
- **Maintainable** - Clear organization makes maintenance easier
