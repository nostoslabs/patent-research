# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Neural Patent Similarity Search Research** project - a production-ready system for evaluating and implementing neural embedding models for patent similarity search. The system processes 40,403 patents with 52,209 embedding vectors across 4 models, with 9,988 LLM-evaluated ground truth similarity pairs.

## Key Commands

### Environment Setup
```bash
# Install dependencies (uses uv package manager)
uv sync

# Install Ollama models for embeddings
ollama pull nomic-embed-text
ollama pull bge-m3
ollama pull embeddinggemma
ollama pull mxbai-embed-large
```

### Running Core Operations
```bash
# Generate embeddings for multiple models
uv run python code/experiments/run_multimodel_experiments.py compare data/patent_abstracts.jsonl

# Create ground truth dataset
uv run python code/experiments/ground_truth_generator.py data/patent_abstracts_with_embeddings.jsonl --pairs 100

# Run comprehensive benchmark
uv run python code/benchmarks/comprehensive_embedding_benchmark.py

# Execute scientific analysis
uv run python code/analysis/scientific_analysis.py
```

### Data Processing
```bash
# Consolidate scattered data files
uv run python code/data_processing/data_consolidation.py

# Generate missing embeddings for specific model
uv run python code/data_processing/generate_missing_bge_embeddings.py

# Convert datasets to different formats
uv run python code/data_processing/convert_to_parquet.py
```

### Linting and Code Quality
```bash
# Run linter (configured in pyproject.toml)
uv run ruff check .
uv run ruff format .
```

## Architecture Overview

### Core Two-Stage Search Pipeline
1. **Stage 1**: Fast embedding-based retrieval using cosine similarity (top-20 results)
2. **Stage 2**: Cross-encoder reranking using BGE-reranker or LLM fallback (top-10 results)

The system achieves <4 second end-to-end processing with production throughput of ~1,000 queries/hour.

### Key Components

**Core System (`code/core/`)**:
- `semantic_search.py` - Main search engine with embedding similarity
- `cross_encoder_reranker.py` - Two-stage retrieval system with reranking
- `llm_provider_factory.py` - Multi-provider LLM integration (OpenAI, Google, Anthropic, Ollama)

**Data Processing Pipeline (`code/data_processing/`)**:
- `data_consolidation.py` - Master data consolidation and organization
- `generate_embeddings_multimodel.py` - Multi-model embedding generation
- `consolidate_embeddings.py` - Merge embeddings from different sources

**Analysis & Experiments (`code/analysis/`, `code/experiments/`)**:
- `scientific_analysis.py` - Complete statistical analysis and reporting
- `ground_truth_generator.py` - LLM-based similarity pair generation
- `model_intersection_analysis.py` - Cross-model performance comparison

### Data Architecture

**Primary Dataset (`data_v2/`)**:
- `master_patent_embeddings.jsonl.xz` (284MB) - Consolidated embeddings for 40,403 patents
- `ground_truth_similarities.jsonl` - 9,988 LLM-evaluated similarity pairs
- `atlas_data/` - Visualization data for Apple Embedding Atlas

**File Handling**: The system uses `code/utilities/file_utils.py` for automatic compressed file detection:
- Supports .xz and .gz compression transparently
- Auto-detects best available file version (compressed vs uncompressed)
- Use `smart_jsonl_reader()` for reading JSONL files

### Model Performance Results

**Recommended Production Configuration**:
- **Primary Embedding**: `nomic-embed-text` (optimal balance: 81.5/100 score, 329 patents/min)
- **Reranker**: `bge-reranker-base` (36x faster than LLM reranking)
- **LLM Fallback**: `gemini-1.5-flash` for quality-critical queries

**Key Research Findings**:
- Weak correlation (r=0.275) between embedding similarity and human/LLM assessment
- BGE reranker processes 18 queries/minute vs 0.5 queries/minute for LLM reranking
- Statistical significance confirmed (Spearman ρ=0.358, p=0.011)

## Development Patterns

### Adding New Embedding Models
1. Install model in Ollama: `ollama pull model-name`
2. Add configuration to `run_multimodel_experiments.py`
3. Update analysis scripts in `code/analysis/`

### Working with Compressed Data
Always use the file utilities for data access:
```python
from code.utilities.file_utils import smart_jsonl_reader

# Automatically handles compressed files
with smart_jsonl_reader("data_v2/master_patent_embeddings.jsonl") as jsonl_data:
    for patent in jsonl_data:
        process_patent(patent)
```

### LLM Integration
Use the factory pattern for multi-provider LLM access:
```python
from code.core.llm_provider_factory import get_llm_client

# Supports OpenAI, Google, Anthropic, Ollama
llm = get_llm_client("gemini-1.5-flash")
```

### Ground Truth Generation
The system uses structured LLM evaluation for similarity assessment:
```python
# Generates similarity scores with confidence levels and reasoning
pairs = await generate_ground_truth_pairs(patent_list, num_pairs=100)
```

## Important Implementation Notes

- **Scale**: System is designed for 100K+ patent processing with efficient memory management
- **Chunking**: Only 5% of patents require chunking with 8K context window models
- **Production Ready**: Real-time search capabilities with cost-efficient local inference
- **Scientific Rigor**: All analysis includes statistical validation and reproducible results
- **Modular Design**: Supports both research experimentation and production deployment

## Testing

Run the comprehensive benchmark suite:
```bash
# Test all embedding models
uv run python code/benchmarks/comprehensive_embedding_benchmark.py

# Validate reranker performance
uv run python code/benchmarks/fair_comparison_benchmark.py

# Scientific reproducibility test
uv run python code/analysis/comprehensive_evaluation.py
```