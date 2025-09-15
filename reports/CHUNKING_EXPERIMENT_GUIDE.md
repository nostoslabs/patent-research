# Comprehensive Chunking Strategy Experiment Guide

This guide explains how to use the comprehensive chunking strategy experimental framework to optimize embeddings for long patent abstracts.

## 🎯 Problem Statement

**Issue**: Patent abstracts can be very long (up to 30,000+ characters), but EmbeddingGemma has a 2K token limit (~8,000 characters). This means:
- 11% of original dataset abstracts are truncated
- 7% of diverse dataset abstracts are truncated  
- We lose semantic information from the truncated portions

**Solution**: Implement multiple chunking and aggregation strategies to preserve full semantic content while maintaining embedding quality.

## 🔬 Experimental Framework

### Chunking Strategies Tested

1. **Fixed-size chunking**: 
   - `fixed_512`: 512 token chunks (~2,048 chars)
   - `fixed_768`: 768 token chunks (~3,072 chars)

2. **Overlapping chunking**:
   - `overlapping_550`: 550 tokens with 50 token overlap

3. **Sentence-boundary chunking**:
   - `sentence_boundary_512`: Target 512 tokens, split at sentence boundaries
   - `sentence_boundary_768`: Target 768 tokens, split at sentence boundaries

4. **Semantic chunking**:
   - `semantic`: Split when cosine similarity between adjacent sentences drops below 0.7

### Aggregation Methods Tested

For each chunking strategy, multiple aggregation methods are tested:

1. **Mean pooling**: Simple average of all chunk embeddings
2. **Max pooling**: Element-wise maximum across chunks
3. **Weighted average**: Position-based weighting (first/last chunks more important)
4. **Attention-weighted**: Similarity-based weighting using mean embedding as query

### Storage Format

Each patent is processed to generate this comprehensive structure:

```json
{
  "id": "patent_123",
  "text_length": 15000,
  "estimated_tokens": 3750,
  "embeddings": {
    "original": {
      "embedding": [...],
      "processing_time": 0.25,
      "was_truncated": true
    },
    "chunking_strategies": {
      "fixed_512_mean": {
        "num_chunks": 4,
        "chunks": [...],
        "aggregations": {
          "mean": [...],
          "max": [...],
          "weighted": [...],
          "attention": [...]
        },
        "total_processing_time": 2.1
      }
      // ... other strategies
    }
  }
}
```

## 🚀 Usage

### Quick Test (25 patents)
```bash
# Test the framework quickly
uv run python run_chunking_experiment.py patent_abstracts.jsonl --max-records 25 --prefix test
```

### Full Analysis (Original Dataset)
```bash
# Analyze all 929 patents from original dataset
uv run python run_chunking_experiment.py patent_abstracts.jsonl --prefix original_full
```

### Diverse Dataset Analysis
```bash
# Analyze 10k diverse patents (will take longer)
uv run python run_chunking_experiment.py patent_abstracts_10k_diverse.jsonl --prefix diverse_analysis
```

### Custom Configuration
```bash
# Custom model and output
uv run python run_chunking_experiment.py patent_abstracts.jsonl \
  --model embeddinggemma \
  --max-records 500 \
  --prefix custom_experiment
```

### Generate Summary Only
```bash
# Create summary from existing results
uv run python run_chunking_experiment.py patent_abstracts.jsonl \
  --summary-only --prefix existing_experiment
```

## 📊 Analysis Metrics

The framework evaluates each strategy across multiple dimensions:

### 1. Clustering Performance
- **Metric**: Silhouette score (higher is better)
- **Methods**: K-means (3,5,7,10 clusters), DBSCAN (eps=0.3,0.5,0.7)
- **Interpretation**:
  - >0.7: Excellent clustering
  - 0.5-0.7: Good clustering  
  - 0.3-0.5: Fair clustering
  - <0.3: Poor clustering

### 2. Semantic Similarity Preservation
- **Metric**: Correlation with original embeddings
- **Method**: Compare pairwise cosine similarities
- **Interpretation**:
  - >0.9: Excellent preservation
  - 0.8-0.9: Good preservation
  - 0.7-0.8: Fair preservation
  - <0.7: Poor preservation

### 3. Processing Efficiency
- **Metric**: Processing time per patent
- **Comparison**: Overhead vs original embedding generation

### 4. Chunk Statistics
- **Metrics**: Average chunk count, size, token distribution
- **Purpose**: Understand chunking behavior patterns

## 📈 Generated Outputs

### 1. Analysis Results (`prefix_analysis.json`)
Complete numerical results including:
- Clustering scores for all strategy/aggregation combinations
- Similarity preservation metrics
- Processing time statistics  
- Chunk distribution data

### 2. Performance Report (`prefix_report.md`)
Human-readable markdown report with:
- Strategy rankings
- Performance recommendations
- Quality assessments
- Statistical summaries

### 3. Interactive Visualizations (`chunking_analysis_visualizations/`)
- `clustering_performance.html`: Bar chart of clustering scores
- `similarity_preservation.html`: Correlation comparison
- `processing_efficiency.html`: Time overhead analysis
- `chunk_statistics.html`: Chunking pattern analysis
- `performance_heatmap.html`: Multi-metric comparison
- `analysis_dashboard.html`: Comprehensive overview

### 4. Experimental Data (`prefix_embeddings.jsonl`)
Complete embedding data for all strategies (large file - for further analysis)

## 🎯 Expected Results

Based on research, we expect:

1. **Best clustering**: UMAP-like strategies (sentence-boundary or semantic)
2. **Best preservation**: Strategies with smaller, overlapping chunks
3. **Best efficiency**: Simple fixed-size chunking
4. **Best aggregation**: Mean or weighted average pooling

## ⚡ Performance Considerations

### Resource Usage
- **Memory**: ~500MB peak for 1000 patents
- **Storage**: ~50MB per 1000 patents (full experimental data)
- **Time**: ~5-10 minutes per patent (with all strategies)

### Optimization Tips
1. **Start small**: Use `--max-records 25` for quick testing
2. **Skip semantic chunking**: Remove for faster processing (requires sentence-level embedding)
3. **Focus on top strategies**: After initial analysis, test only promising strategies on larger datasets

## 🔧 Script Details

### Core Scripts

1. **`run_chunking_experiment.py`**: Main workflow orchestrator
2. **`generate_embeddings_experimental.py`**: Embedding generation with all strategies  
3. **`analyze_chunking_performance.py`**: Performance analysis and metrics
4. **`visualize_chunking_analysis.py`**: Interactive visualization generation

### Supporting Classes

- **`TextChunker`**: Implements all chunking strategies
- **`EmbeddingAggregator`**: Implements all aggregation methods
- **`ExperimentalEmbeddingGenerator`**: Orchestrates embedding generation
- **`ChunkingAnalyzer`**: Comprehensive performance analysis
- **`ChunkingVisualizer`**: Chart and dashboard generation

## 🎪 Example Workflow

```bash
# 1. Quick test to validate framework
uv run python run_chunking_experiment.py patent_abstracts.jsonl --max-records 25 --prefix test

# 2. Check test results
cat test_report.md
open chunking_analysis_visualizations/analysis_dashboard.html

# 3. Run full analysis on dataset of choice
uv run python run_chunking_experiment.py patent_abstracts_10k_diverse.jsonl --prefix production

# 4. Review results and select best strategy
cat production_report.md

# 5. Implement best strategy in production embedding pipeline
```

## 🏆 Using Results

After completing the experiment:

1. **Identify the best performing strategy** from the report
2. **Implement that strategy** in your production embedding pipeline
3. **Use aggregated embeddings** instead of truncated original embeddings
4. **Monitor performance** on downstream tasks (search, clustering, etc.)

## 🔍 Troubleshooting

### Common Issues

**Memory errors**: Reduce `--max-records` or process in batches

**Ollama connection errors**: Ensure Ollama is running and embeddinggemma model is available

**Missing visualizations**: Check for plotly installation: `uv add plotly`

**Slow semantic chunking**: Skip semantic strategy for faster processing

### Validation

The framework includes validation checks:
- ✅ Model connectivity testing
- ✅ Embedding dimension verification  
- ✅ Progress saving every 10 records
- ✅ Error handling for failed chunks
- ✅ Data consistency checks

## 📚 Next Steps

1. **Run experiments** on your dataset
2. **Analyze results** using the generated reports
3. **Select optimal strategy** based on your priorities (quality vs speed)  
4. **Integrate into production** embedding pipeline
5. **Validate improvements** on downstream tasks
6. **Scale to larger datasets** as needed

---

**Ready to optimize your patent embeddings? Start with the quick test!**

```bash
uv run python run_chunking_experiment.py patent_abstracts.jsonl --max-records 25 --prefix my_test
```