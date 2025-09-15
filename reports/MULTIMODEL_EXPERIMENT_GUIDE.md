# Multi-Model Embedding Experiment Guide

This comprehensive guide explains how to run experiments across multiple embedding models with full progress tracking and automated analysis.

## 🎯 Overview

The multi-model framework allows you to:
- **Test multiple embedding models** (embeddinggemma, bge-m3, mxbai-embed-large, nomic-embed-text, etc.)
- **Compare chunking strategies** across different model context windows
- **Track experiment progress** with detailed metrics and ETA
- **Analyze results comprehensively** with cross-model comparisons
- **Scale to large datasets** with parallel processing support

## 🤖 Supported Models

| Model | Context Limit | Embedding Dim | Char Limit | Best Use Case |
|-------|--------------|---------------|------------|---------------|
| **embeddinggemma** | 2K tokens | 768D | ~8K chars | Fast, lightweight |
| **bge-m3** | 8K tokens | 1024D | ~28K chars | High quality, multilingual |
| **mxbai-embed-large** | 512 tokens | 1024D | ~2K chars | High precision |
| **nomic-embed-text** | 8K tokens | 768D | ~30K chars | Long context, efficient |
| **all-MiniLM-L6-v2** | 512 tokens | 384D | ~2K chars | Very fast, small |

## 📊 Progress Tracking System

The framework includes comprehensive experiment tracking:

### Experiment States
- **🟡 Pending**: Created but not started
- **🔄 Running**: Currently processing
- **✅ Completed**: Successfully finished
- **❌ Failed**: Encountered errors
- **🚫 Cancelled**: Manually stopped

### Tracked Metrics
- **Progress percentage** and ETA
- **Processing rate** (patents/minute)
- **Success/failure counts**
- **Model-specific performance**
- **Memory and time usage**

## 🚀 Quick Start

### 1. Check System Status
```bash
# Check experiment tracker status
uv run python run_multimodel_experiments.py status

# Check model availability
uv run python experiment_tracker.py status
```

### 2. Install Missing Models
```bash
# Pull default models
uv run python run_multimodel_experiments.py pull-models

# Pull specific models
uv run python run_multimodel_experiments.py pull-models --models "bge-m3,nomic-embed-text"
```

### 3. Run Quick Test (Recommended First)
```bash
# Test with 25 patents across available models
uv run python run_multimodel_experiments.py compare patent_abstracts.jsonl \
  --batch-name quick_test --max-records 25
```

### 4. Run Comprehensive Comparison
```bash
# Compare all models on 100 patents
uv run python run_multimodel_experiments.py compare patent_abstracts.jsonl \
  --batch-name full_comparison --max-records 100

# Compare specific models on diverse dataset
uv run python run_multimodel_experiments.py compare patent_abstracts_10k_diverse.jsonl \
  --batch-name diverse_test --models "embeddinggemma,bge-m3" --max-records 500
```

## 📋 Detailed Workflows

### Workflow 1: Single Model Deep Dive
```bash
# Create experiment for single model with comprehensive chunking
uv run python experiment_tracker.py create deep_dive_gemma patent_abstracts.jsonl embeddinggemma

# Run with full chunking analysis
uv run python generate_embeddings_multimodel.py patent_abstracts.jsonl \
  gemma_deep_dive.jsonl embeddinggemma 100 deep_dive_gemma

# Analyze results
uv run python analyze_chunking_performance.py gemma_deep_dive.jsonl
```

### Workflow 2: Cross-Model Comparison
```bash
# Create batch of experiments
uv run python run_multimodel_experiments.py create-batch \
  patent_abstracts.jsonl model_comparison --models "embeddinggemma,bge-m3,nomic-embed-text"

# Run comprehensive comparison
uv run python run_multimodel_experiments.py compare patent_abstracts.jsonl \
  --batch-name model_comparison --max-records 100
```

### Workflow 3: Large Scale Analysis
```bash
# Run on diverse 10k dataset with all available models
uv run python run_multimodel_experiments.py compare patent_abstracts_10k_diverse.jsonl \
  --batch-name large_scale --max-records 1000
```

## 📈 Progress Monitoring

### Real-time Status
```bash
# Watch experiment progress
watch -n 5 'uv run python experiment_tracker.py status'

# Check specific experiment
uv run python experiment_tracker.py list
```

### Progress Dashboard
The tracker provides:
- **Overall progress**: Experiments by status
- **Running experiments**: Progress %, ETA, processing rate
- **Recent completions**: Last 24 hours
- **Model performance**: Comparative metrics

### Example Status Output
```
# Experiment Progress Report

Generated: 2024-09-12 14:30:15
Total Experiments: 12

## 📊 Status Summary
⏳ Pending: 2
🔄 Running: 3  
✅ Completed: 6
❌ Failed: 1

## 🔄 Currently Running
Experiment               Model                Progress   ETA      Rate/min
--------------------------------------------------------------------------------
diverse_test_bge-m3      bge-m3               67.5%      4.2m     8.3
diverse_test_gemma       embeddinggemma       45.2%      8.1m     12.1
large_scale_nomic        nomic-embed-text     23.8%      15.3m    5.2

## ✅ Recent Completions (24h)  
Experiment               Model                Completed    Duration
--------------------------------------------------------------------------
quick_test_gemma         embeddinggemma       2.1h         3.2m
quick_test_bge           bge-m3               2.5h         5.8m
```

## 🔧 Advanced Usage

### Custom Model Configuration
```python
# Add new model to tracker
from experiment_tracker import ExperimentTracker, ModelConfig

tracker = ExperimentTracker()
tracker.add_model_config(ModelConfig(
    name="custom-model",
    context_limit=4096,
    embedding_dim=512,
    char_per_token=3.5
))
```

### Parallel Processing
```bash
# Run multiple experiments in parallel (coming soon)
uv run python run_multimodel_experiments.py compare patent_abstracts.jsonl \
  --batch-name parallel_test --parallel --max-parallel 3
```

### Experiment Analysis
```bash
# Analyze specific experiment
uv run python analyze_chunking_performance.py experiment_embeddings.jsonl

# Generate visualizations
uv run python visualize_chunking_analysis.py experiment_analysis.json

# Cross-model comparison
uv run python compare_model_results.py batch_results.json
```

## 📊 Analysis Outputs

### Per-Model Analysis
Each experiment generates:
- **Clustering performance** (silhouette scores)
- **Chunking effectiveness** (when needed)
- **Processing efficiency** (time, memory)
- **Quality metrics** (semantic preservation)

### Cross-Model Comparison
Batch experiments generate:
- **Model performance rankings**
- **Context window utilization**
- **Chunking strategy effectiveness**
- **Speed vs quality trade-offs**

### Visualization Dashboard
Interactive charts showing:
- **Performance heatmaps** across models
- **Processing time comparisons**
- **Quality vs efficiency plots**
- **Context window analysis**

## 🎯 Example Results

### Expected Performance Hierarchy
1. **bge-m3**: Highest quality, good context window
2. **nomic-embed-text**: Best context window, good quality
3. **embeddinggemma**: Fast, lightweight, good balance
4. **mxbai-embed-large**: High precision, limited context
5. **all-MiniLM-L6-v2**: Fastest, smallest, basic quality

### Context Window Impact
- **Short abstracts** (<2K chars): All models similar performance
- **Medium abstracts** (2K-8K chars): Context window matters
- **Long abstracts** (>8K chars): bge-m3, nomic-embed-text excel

## 🔍 Troubleshooting

### Common Issues

**Model not found**
```bash
# Check available models
ollama list

# Pull missing model
ollama pull bge-m3
```

**Memory errors**
```bash
# Reduce batch size
--max-records 50

# Use chunking strategies
# (automatically handled for long texts)
```

**Slow processing**
```bash
# Check system resources
htop

# Use faster models for testing
--models embeddinggemma

# Reduce sample size
--max-records 25
```

**Experiment tracking issues**
```bash
# Reset tracker state
rm experiment_tracking.json

# Check tracker status
uv run python experiment_tracker.py status
```

## 📚 File Structure

```
patent_research/
├── experiment_tracker.py              # Core tracking system
├── generate_embeddings_multimodel.py  # Multi-model embedding generator
├── run_multimodel_experiments.py      # Experiment orchestrator
├── analyze_chunking_performance.py    # Performance analysis
├── visualize_chunking_analysis.py     # Visualization generator
├── MULTIMODEL_EXPERIMENT_GUIDE.md     # This guide
├── experiment_tracking.json           # Persistent experiment state
└── results/
    ├── batch_name_batch_results.json  # Batch execution results
    ├── batch_name_model_comparison.json # Cross-model analysis
    └── experiment_*_embeddings.jsonl  # Individual experiment results
```

## 🎪 Complete Example Workflow

```bash
# 1. Initial setup and testing
uv run python run_multimodel_experiments.py status
uv run python run_multimodel_experiments.py pull-models

# 2. Quick validation test
uv run python run_multimodel_experiments.py compare patent_abstracts.jsonl \
  --batch-name validation --max-records 25

# 3. Review validation results
cat validation_batch_results.json
uv run python visualize_chunking_analysis.py validation_model_comparison.json

# 4. Full comparison on diverse dataset
uv run python run_multimodel_experiments.py compare patent_abstracts_10k_diverse.jsonl \
  --batch-name production --max-records 500

# 5. Monitor progress
watch -n 10 'uv run python experiment_tracker.py status'

# 6. Analyze final results
cat production_batch_results.json
open chunking_analysis_visualizations/analysis_dashboard.html

# 7. Select best model and implement in production
```

## 💡 Best Practices

1. **Start small**: Always test with 25-50 patents first
2. **Check resources**: Monitor memory and CPU usage
3. **Use appropriate models**: Match model context window to your data
4. **Track everything**: Use experiment tracking for all runs
5. **Compare systematically**: Use batch experiments for fair comparison
6. **Analyze thoroughly**: Review both quantitative metrics and visualizations
7. **Document decisions**: Save experiment configurations and results

---

**Ready to compare embedding models? Start with the validation test!**

```bash
uv run python run_multimodel_experiments.py compare patent_abstracts.jsonl --batch-name my_validation --max-records 25
```