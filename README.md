# Neural Patent Similarity Search Research

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Comprehensive research on neural embedding models for patent similarity search, featuring production-ready implementations and scientific analysis.

## 🎯 Project Overview

This repository contains a complete research pipeline for evaluating and implementing neural patent similarity search systems. The project evaluates 4 embedding models across 1,625+ patents and provides production-ready implementations with sub-4 second query processing.

### Key Achievements
- **36× speedup** with dedicated rerankers vs. LLM approaches
- **Production-ready system** processing 329 patents/minute  
- **Scientific validation** with proper statistical analysis
- **Novel insights** into embedding-LLM similarity disconnect (r=0.275)

---

## 📁 Project Structure

```
patent_research/
├── README.md                    # This file
├── pyproject.toml              # Python dependencies
├── .python-version             # Python version specification
├── 
├── code/                       # Implementation code
│   ├── llm_provider_factory.py      # PydanticAI multi-provider LLM system
│   ├── reranker_enhancement_plan.py # Advanced multi-reranker system  
│   ├── cross_encoder_reranker.py    # Two-stage retrieval implementation
│   ├── ground_truth_generator.py    # LLM-based similarity evaluation
│   ├── google_patents_baseline.py   # Classification-based search baseline
│   ├── scientific_analysis.py       # Statistical analysis & visualization
│   ├── run_multimodel_experiments.py # Batch experiment runner
│   ├── model_performance_analyzer.py # Performance analysis & ranking
│   ├── comprehensive_evaluation.py   # End-to-end evaluation pipeline
│   └── download_large_diverse_patents.py # Dataset preparation
│
├── data/                       # Datasets and results
│   ├── patent_abstracts.jsonl           # Base patent dataset
│   ├── patent_abstracts_10k_diverse.jsonl # 10K diverse patents
│   ├── patent_abstracts_100k_diverse.jsonl # 100K diverse patents  
│   ├── patent_ground_truth_100.jsonl     # LLM-evaluated pairs
│   ├── baseline_comparison_100_queries.jsonl # Baseline study results
│   ├── *_embeddings.jsonl               # Generated embeddings
│   ├── *_batch_results.json             # Experiment results
│   └── patent_similarity_statistical_analysis.json # Statistics
│
├── reports/                    # Research reports and analysis
│   ├── SCIENTIFIC_PATENT_SEARCH_ANALYSIS.md  # 📄 Main scientific paper
│   ├── PROJECT_SUMMARY.md                    # Complete project overview
│   ├── MODEL_PERFORMANCE_ANALYSIS.md         # Detailed model comparison
│   ├── RERANKER_PERFORMANCE_ANALYSIS.md      # Reranker evaluation
│   ├── MODEL_COMPARISON_SUMMARY.md           # Executive summary
│   └── CHUNKING_EXPERIMENT_GUIDE.md          # Chunking analysis
│
├── visualizations/             # Generated plots and charts
│   ├── patent_similarity_analysis.png   # 4-panel scientific visualization
│   ├── correlation_heatmap.png          # Correlation matrix
│   └── chunking_analysis_visualizations/ # Chunking experiment plots
│
└── analysis/                   # Additional analysis files
    └── [temporary analysis files]
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **uv** package manager (recommended) or pip
- **Ollama** with embedding models installed
- **API Keys** (optional): OpenAI, Google, Anthropic, Cohere

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd patent_research

# Install dependencies with uv (recommended)
uv sync

# OR install with pip
pip install -r requirements.txt

# Install Ollama embedding models
ollama pull nomic-embed-text
ollama pull bge-m3
ollama pull embeddinggemma  
ollama pull mxbai-embed-large
```

### Environment Setup

Create a `.env` file with your API keys (optional):
```env
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
COHERE_API_KEY=your_cohere_key_here
```

---

## 🔬 Usage Examples

### 1. Quick Patent Similarity Search

```python
from code.cross_encoder_reranker import EmbeddingSearchEngine

# Load embeddings
searcher = EmbeddingSearchEngine("data/patent_abstracts_with_embeddings.jsonl")

# Find similar patents
results = searcher.search_similar("patent_12345", top_k=10)
for result in results:
    print(f"{result.patent_id}: {result.similarity:.3f}")
```

### 2. Advanced Reranking

```python
from code.reranker_enhancement_plan import AdvancedRerankerSystem

# Initialize with BGE reranker
system = AdvancedRerankerSystem(
    embeddings_file="data/patent_abstracts_with_embeddings.jsonl",
    reranker_type="bge-reranker-base"
)

# Get initial results and rerank
initial_results = system.embedding_searcher.search_similar("patent_12345", 20)
reranked = await system.rerank_results(query_patent, initial_results, 10)
```

### 3. Run Complete Evaluation

```bash
# Generate embeddings for all models
uv run python code/run_multimodel_experiments.py compare data/patent_abstracts.jsonl

# Create ground truth dataset
uv run python code/ground_truth_generator.py data/patent_abstracts_with_embeddings.jsonl --pairs 100

# Run baseline comparison  
uv run python code/google_patents_baseline.py data/patent_abstracts_with_embeddings.jsonl --sample 100

# Generate scientific analysis
uv run python code/scientific_analysis.py
```

---

## 📊 Key Research Findings

### Embedding Model Rankings

| Model | Speed (pat/min) | Overall Score | Context Window | Production Ready |
|-------|-----------------|---------------|----------------|------------------|
| **nomic-embed-text** | 329 | 81.5/100 | 8,192 tokens | ✅ **Yes** |
| **bge-m3** | 133 | 66.0/100 | 8,192 tokens | ✅ Yes |
| **embeddinggemma** | 99 | 27.2/100 | 2,048 tokens | ⚠️ Limited |
| **mxbai-embed-large** | 50 | 24.8/100 | 512 tokens | ❌ No |

### Reranker Performance

- **BGE-reranker-base**: 36× faster than LLM reranking (3.3s vs 120s per query)
- **Production throughput**: 18 queries/minute with high-quality results
- **Recommendation**: Use BGE for speed, LLM fallback for explainability

### Statistical Validation

- **Sample size**: 50 patent pairs with structured LLM evaluation
- **Correlation**: r=0.275 (weak) between embedding and LLM similarity
- **Baseline overlap**: 15.2% between embedding and classification search
- **Significance**: Spearman ρ=0.358, p=0.011 (statistically significant)

---

## 🏗️ Production Architecture

### Recommended Configuration

```python
# Production-ready two-stage system
EMBEDDING_MODEL = "nomic-embed-text"    # 329 patents/min
RERANKER_MODEL = "bge-reranker-base"    # 18 queries/min  
FALLBACK_LLM = "gemini-1.5-flash"      # For quality-critical queries

# Performance targets achieved:
# - Sub-4 second end-to-end processing
# - Real-time search on 10K+ patent corpora  
# - Cost-efficient local inference
```

### System Performance

- **Processing**: 10,000 patents in ~30 minutes (embedding generation)
- **Query speed**: <4 seconds end-to-end with reranking
- **Throughput**: ~1,000 queries/hour production capacity
- **Memory**: Minimal chunking (5% of patents) with 8K context models

---

## 🔬 Reproducing Research

### Complete Research Pipeline

```bash
# 1. Download and prepare datasets
uv run python code/download_large_diverse_patents.py 10k
uv run python code/download_large_diverse_patents.py 100k

# 2. Generate embeddings for all models
uv run python code/run_multimodel_experiments.py compare patent_abstracts_10k_diverse.jsonl --batch-name research

# 3. Create ground truth evaluation
uv run python code/ground_truth_generator.py data/patent_abstracts_with_embeddings.jsonl --pairs 200

# 4. Run baseline comparisons
uv run python code/google_patents_baseline.py data/patent_abstracts_with_embeddings.jsonl --sample 100

# 5. Test reranker systems
uv run python code/reranker_enhancement_plan.py data/patent_abstracts_with_embeddings.jsonl --benchmark

# 6. Generate scientific analysis
uv run python code/scientific_analysis.py
uv run python code/model_performance_analyzer.py

# 7. Create comprehensive evaluation
uv run python code/comprehensive_evaluation.py
```

### Expected Outputs

- **Embeddings**: Generated for all 4 models across datasets
- **Performance metrics**: Speed, accuracy, context efficiency  
- **Statistical analysis**: Correlations, distributions, significance tests
- **Visualizations**: Scientific plots and correlation heatmaps
- **Reports**: Publication-ready analysis documents

---

## 📚 Key Documentation

### Scientific Papers
- **[SCIENTIFIC_PATENT_SEARCH_ANALYSIS.md](reports/SCIENTIFIC_PATENT_SEARCH_ANALYSIS.md)**: Main research paper with methodology, results, and analysis
- **[PROJECT_SUMMARY.md](reports/PROJECT_SUMMARY.md)**: Complete project overview and achievements

### Technical Documentation  
- **[MODEL_PERFORMANCE_ANALYSIS.md](reports/MODEL_PERFORMANCE_ANALYSIS.md)**: Detailed embedding model comparison
- **[RERANKER_PERFORMANCE_ANALYSIS.md](reports/RERANKER_PERFORMANCE_ANALYSIS.md)**: Reranker evaluation results

### Implementation Guides
- **[code/llm_provider_factory.py](code/llm_provider_factory.py)**: Multi-provider LLM integration
- **[code/reranker_enhancement_plan.py](code/reranker_enhancement_plan.py)**: Advanced reranking implementation

---

## 🛠️ Development & Extension

### Adding New Embedding Models

1. **Install model in Ollama**: `ollama pull model-name`
2. **Add to experiment runner**: Edit `code/run_multimodel_experiments.py`
3. **Update analysis**: Add model to `code/model_performance_analyzer.py`

### Extending Rerankers

1. **Implement reranker**: Add to `code/reranker_enhancement_plan.py`  
2. **Add configuration**: Update `RERANKER_CONFIGS` dictionary
3. **Test performance**: Run benchmark comparison

### Custom Datasets

```python
# Prepare your patent data in JSONL format:
{"id": "patent_id", "abstract": "patent abstract text", "classification": "patent_class"}

# Then run the complete pipeline:
uv run python code/run_multimodel_experiments.py compare your_dataset.jsonl
```

---

## 🤝 Contributing

### Research Contributions Needed

1. **Expand ground truth**: Increase sample size to n≥200 for robust statistics
2. **Human validation**: Compare LLM judgments against patent experts  
3. **Full document analysis**: Include claims, figures, and specifications
4. **Domain specialization**: Fine-tune models for specific technical areas

### Code Contributions

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/new-feature`
3. **Make changes** and test thoroughly
4. **Submit pull request** with detailed description

### Standards

- **Python 3.8+** compatibility
- **Type hints** for all functions
- **Comprehensive docstrings**
- **Unit tests** for new functionality
- **Scientific rigor** in analysis

---

## 📊 Performance Benchmarks

### System Requirements

- **CPU**: Multi-core recommended (embedding generation)
- **Memory**: 8GB+ RAM for large datasets  
- **GPU**: Optional (can accelerate embedding models)
- **Storage**: 10GB+ for complete datasets and results

### Scaling Considerations

- **10K patents**: Real-time search, <1GB storage
- **100K patents**: Near real-time, ~5GB storage  
- **1M+ patents**: Requires optimization, distributed processing

---

## 📝 License & Citation

### License
This project is licensed under the MIT License - see LICENSE file for details.

### Citation
If you use this research in your work, please cite:

```bibtex
@misc{patent_similarity_research_2025,
  title={Neural Patent Similarity Search: A Comprehensive Evaluation of Embedding Models and Reranking Approaches},
  author={[Author Names]},
  year={2025},
  note={Available at: [Repository URL]}
}
```

---

## 🚨 Limitations & Future Work

### Current Limitations

1. **Ground truth sample size**: n=50 pairs insufficient for robust correlation analysis
2. **Single LLM evaluator**: Results dependent on Gemini-1.5-Flash capabilities
3. **Abstract-only analysis**: Missing claims, figures, and technical specifications  
4. **Limited domain coverage**: 7 classification groups may not represent full diversity

### Immediate Next Steps

1. **Expand ground truth to n≥200** with stratified sampling
2. **Multi-evaluator validation** (human experts + multiple LLMs)  
3. **Full patent document analysis** incorporating all sections
4. **Cross-domain validation** across different technical fields

### Research Extensions

1. **Hybrid search systems** combining embedding + classification + graph approaches
2. **Explainable similarity** for patent professional workflows
3. **Temporal analysis** of patent similarity evolution  
4. **Multi-modal integration** with figures and chemical structures

---

## 📞 Support & Contact

### Getting Help

1. **Check documentation** in `reports/` directory
2. **Review code comments** for implementation details
3. **Run example scripts** to understand usage patterns
4. **Open GitHub issues** for bugs or feature requests

### Research Collaboration

We welcome collaborations on:
- **Patent similarity research**
- **LLM evaluation methodologies**  
- **Production deployment case studies**
- **Domain-specific applications**

---

**🎓 This repository provides a complete foundation for neural patent similarity research with production-ready implementations, scientific validation, and extensible architecture for future development.**