# AI Coding Agent Guide

**Patent Research Project - AI Agent Navigation Guide**
*For AI coding assistants working on this project*

---

## 🤖 Quick Reference for AI Agents

### **Project Type**: Neural Patent Similarity Search Research
### **Main Language**: Python 3.11+
### **Key Frameworks**: PydanticAI, Ollama, Sentence Transformers
### **Data Scale**: 40,403 patents, 52,209 embedding vectors, 9,988 ground truth pairs

---

## 📁 Directory Navigation Map

```
patent_research/
├── 🐍 code/                    # ALL PYTHON CODE HERE
│   ├── core/                   # Core system (4 files)
│   ├── data_processing/        # Data handling (8 files)
│   ├── analysis/               # Analysis tools (7 files)
│   ├── experiments/            # Research workflows (7 files)
│   ├── benchmarks/             # Performance testing (8 files)
│   ├── monitoring/             # System monitoring (7 files)
│   └── utilities/              # Helper tools (20+ files)
│
├── 🗄️ data_v2/                 # CONSOLIDATED DATA HERE
│   ├── master_patent_embeddings.jsonl    # 875MB main dataset
│   ├── ground_truth_similarities.jsonl  # 11.5MB LLM evaluations
│   ├── atlas_data/             # Visualization data
│   └── batch_processing/       # OpenAI API batch files
│
├── 📊 reports/                  # ALL DOCUMENTATION HERE
│   ├── scientific/             # Research papers (3 files)
│   ├── technical/              # Technical docs (7 files)
│   ├── analysis/               # Analysis outputs (4 files)
│   └── images/                 # Generated visualizations
│
├── 📈 figures/                 # Generated plots and charts
├── 📋 analysis/                # Analysis results and outputs
└── 🔧 scripts/                 # Utility scripts
```

---

## 🎯 Common AI Agent Tasks

### **1. Finding Code by Function**
```bash
# Core system components
code/core/llm_provider_factory.py          # Multi-provider LLM system
code/core/cross_encoder_reranker.py        # Two-stage retrieval
code/core/reranker_enhancement_plan.py     # Advanced reranking
code/core/semantic_search.py              # Semantic search

# Data processing
code/data_processing/download_large_diverse_patents.py  # Patent downloads
code/data_processing/generate_embeddings_multimodel.py   # Embedding generation
code/data_processing/data_consolidation.py              # Data consolidation

# Analysis tools
code/analysis/scientific_analysis.py      # Statistical analysis
code/analysis/model_performance_analyzer.py # Model comparison
code/analysis/comprehensive_evaluation.py  # End-to-end evaluation

# Experiments
code/experiments/run_multimodel_experiments.py # Multi-model experiments
code/experiments/ground_truth_generator.py    # Ground truth generation
code/experiments/run_chunking_experiment.py   # Chunking experiments

# Benchmarks
code/benchmarks/comprehensive_embedding_benchmark.py # Embedding benchmarks
code/benchmarks/fair_comparison_benchmark.py         # Fair comparisons
code/benchmarks/google_patents_baseline.py           # Baseline comparison

# Monitoring
code/monitoring/monitor_experiments.py      # Experiment monitoring
code/monitoring/monitor_ground_truth_progress.py # Progress tracking

# Utilities
code/utilities/create_visualizations.py    # Visualization creation
code/utilities/batch_manager.py            # Batch processing
code/utilities/launch_atlas.py             # Atlas launch
```

### **2. Data Access Patterns**
```python
# Master data files
MASTER_EMBEDDINGS = "data_v2/master_patent_embeddings.jsonl"
GROUND_TRUTH = "data_v2/ground_truth_similarities.jsonl"

# Atlas data
ATLAS_DATA = "data_v2/atlas_data/patent_atlas_enhanced.parquet"

# Batch processing
BATCH_DIR = "data_v2/batch_processing/"
```

### **3. Import Patterns**
```python
# Core system imports
from code.core.llm_provider_factory import LLMProviderFactory
from code.core.cross_encoder_reranker import EmbeddingSearchEngine

# Data processing imports
from code.data_processing.generate_embeddings_multimodel import EmbeddingGenerator
from code.data_processing.data_consolidation import DataConsolidator

# Analysis imports
from code.analysis.scientific_analysis import ScientificAnalyzer
from code.analysis.model_performance_analyzer import ModelPerformanceAnalyzer

# Experiment imports
from code.experiments.run_multimodel_experiments import MultiModelExperimentRunner
from code.experiments.ground_truth_generator import GroundTruthGenerator

# Benchmark imports
from code.benchmarks.comprehensive_embedding_benchmark import EmbeddingBenchmark
from code.benchmarks.fair_comparison_benchmark import FairComparisonBenchmark

# Monitoring imports
from code.monitoring.monitor_experiments import ExperimentMonitor
from code.monitoring.monitor_ground_truth_progress import GroundTruthProgressMonitor

# Utility imports
from code.utilities.create_visualizations import VisualizationCreator
from code.utilities.batch_manager import BatchManager
```

---

## 🔍 File Location Quick Reference

### **By Function**
| Function | Location | Key Files |
|----------|----------|-----------|
| **LLM Integration** | `code/core/` | `llm_provider_factory.py` |
| **Embedding Generation** | `code/data_processing/` | `generate_embeddings_multimodel.py` |
| **Data Analysis** | `code/analysis/` | `scientific_analysis.py` |
| **Experiments** | `code/experiments/` | `run_multimodel_experiments.py` |
| **Benchmarks** | `code/benchmarks/` | `comprehensive_embedding_benchmark.py` |
| **Monitoring** | `code/monitoring/` | `monitor_experiments.py` |
| **Visualizations** | `code/utilities/` | `create_visualizations.py` |

### **By Data Type**
| Data Type | Location | Key Files |
|-----------|----------|-----------|
| **Master Embeddings** | `data_v2/` | `master_patent_embeddings.jsonl` |
| **Ground Truth** | `data_v2/` | `ground_truth_similarities.jsonl` |
| **Atlas Data** | `data_v2/atlas_data/` | `patent_atlas_enhanced.parquet` |
| **Batch Processing** | `data_v2/batch_processing/` | `openai_batch_*.jsonl` |

### **By Documentation Type**
| Doc Type | Location | Key Files |
|----------|----------|-----------|
| **Scientific Papers** | `reports/scientific/` | `SCIENTIFIC_PATENT_SEARCH_ANALYSIS.md` |
| **Technical Docs** | `reports/technical/` | `MODEL_PERFORMANCE_ANALYSIS.md` |
| **Analysis Outputs** | `reports/analysis/` | `Better_Patent_Search_Solution_Plan.md` |
| **Images** | `reports/images/` | `*.png` files |

---

## 🚨 Critical Rules for AI Agents

### **❌ NEVER DO THESE**
1. **Don't create Python files in root directory** - Use `code/` with appropriate subdirectory
2. **Don't mix file categories** - Keep related files together
3. **Don't delete or move files** without understanding the organization
4. **Don't create new directories** without following established patterns
5. **Don't put data files in code directories** - Use `data_v2/`
6. **Don't put code files in data directories** - Use `code/`

### **✅ ALWAYS DO THESE**
1. **Use the organized directory structure** - Follow the established pattern
2. **Check existing README files** - They contain important guidance
3. **Follow naming conventions** - Use descriptive, consistent names
4. **Maintain logical organization** - Group related functionality
5. **Update documentation** - Keep README files current
6. **Preserve data integrity** - Don't modify master data files

---

## 🛠️ Common Development Patterns

### **Creating New Scripts**
```python
# Standard script structure
import sys
import os
import json
from pathlib import Path

# Add code directory to path
sys.path.append('code')

# Import from organized directories
from core.llm_provider_factory import LLMProviderFactory
from analysis.scientific_analysis import ScientificAnalyzer

# Data file paths
DATA_DIR = Path("data_v2")
MASTER_FILE = DATA_DIR / "master_patent_embeddings.jsonl"
GROUND_TRUTH_FILE = DATA_DIR / "ground_truth_similarities.jsonl"

def main():
    # Your code here
    pass

if __name__ == "__main__":
    main()
```

### **Loading Data**
```python
# Load patent embeddings
def load_patent_embeddings(file_path="data_v2/master_patent_embeddings.jsonl"):
    patents = []
    with open(file_path, 'r') as f:
        for line in f:
            patent = json.loads(line)
            patents.append(patent)
    return patents

# Load ground truth similarities
def load_ground_truth(file_path="data_v2/ground_truth_similarities.jsonl"):
    similarities = []
    with open(file_path, 'r') as f:
        for line in f:
            pair = json.loads(line)
            similarities.append(pair)
    return similarities
```

### **Running Experiments**
```python
# Standard experiment pattern
from experiments.run_multimodel_experiments import MultiModelExperimentRunner
from experiments.ground_truth_generator import GroundTruthGenerator

# Run multi-model experiments
experiment_runner = MultiModelExperimentRunner()
results = experiment_runner.run_experiments()

# Generate ground truth
gt_generator = GroundTruthGenerator()
ground_truth = gt_generator.generate_ground_truth()
```

---

## 📊 Data Access Patterns

### **Master Data Files**
```python
# Patent embeddings (875MB)
MASTER_EMBEDDINGS = "data_v2/master_patent_embeddings.jsonl"

# Ground truth similarities (11.5MB)
GROUND_TRUTH = "data_v2/ground_truth_similarities.jsonl"

# Atlas visualization data
ATLAS_DATA = "data_v2/atlas_data/patent_atlas_enhanced.parquet"
```

### **Data Statistics**
- **40,403 patents** in master dataset
- **52,209 embedding vectors** across 3 models
- **9,988 ground truth pairs** for evaluation
- **3 models available**: bge-m3, openai_text-embedding-3-small, nomic-embed-text

### **Data Loading Examples**
```python
import json
import pandas as pd

# Load JSONL data
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Load Parquet data
def load_parquet(file_path):
    return pd.read_parquet(file_path)
```

---

## 🔧 Tool Usage Patterns

### **Embedding Generation**
```python
from data_processing.generate_embeddings_multimodel import EmbeddingGenerator

generator = EmbeddingGenerator()
embeddings = generator.generate_embeddings(
    model_name="nomic-embed-text",
    data_file="data_v2/master_patent_embeddings.jsonl"
)
```

### **Analysis Tools**
```python
from analysis.scientific_analysis import ScientificAnalyzer
from analysis.model_performance_analyzer import ModelPerformanceAnalyzer

# Scientific analysis
analyzer = ScientificAnalyzer()
results = analyzer.analyze_similarities()

# Model performance analysis
perf_analyzer = ModelPerformanceAnalyzer()
performance = perf_analyzer.compare_models()
```

### **Monitoring Tools**
```python
from monitoring.monitor_experiments import ExperimentMonitor
from monitoring.monitor_ground_truth_progress import GroundTruthProgressMonitor

# Monitor experiments
monitor = ExperimentMonitor()
status = monitor.check_experiment_status()

# Monitor ground truth progress
gt_monitor = GroundTruthProgressMonitor()
progress = gt_monitor.check_progress()
```

---

## 📚 Documentation Navigation

### **Start Here**
1. `README.md` - Main project overview
2. `DEVELOPER_ONBOARDING_GUIDE.md` - Complete developer guide
3. `AI_AGENT_GUIDE.md` - This guide for AI agents

### **Code Documentation**
1. `code/README.md` - Code organization guide
2. `reports/README.md` - Report structure guide
3. `data_v2/README.md` - Data organization guide

### **Research Documentation**
1. `reports/scientific/SCIENTIFIC_PATENT_SEARCH_ANALYSIS.md` - Main research paper
2. `reports/scientific/PROJECT_SUMMARY.md` - Project overview
3. `reports/technical/MODEL_PERFORMANCE_ANALYSIS.md` - Model analysis

---

## 🎯 Task-Specific Guidance

### **Adding New Features**
1. **Identify the category** (core, data_processing, analysis, etc.)
2. **Place in appropriate directory** within `code/`
3. **Follow existing patterns** for imports and structure
4. **Update relevant README** files
5. **Test with existing data** and tools

### **Modifying Existing Code**
1. **Understand the organization** before making changes
2. **Check related files** in the same directory
3. **Maintain consistency** with existing patterns
4. **Update documentation** if needed
5. **Test thoroughly** with existing data

### **Working with Data**
1. **Use master data files** in `data_v2/`
2. **Don't modify original data** without backup
3. **Follow established loading patterns**
4. **Check data organization guide** for guidance
5. **Preserve data integrity** at all times

### **Creating Documentation**
1. **Place in appropriate directory** within `reports/`
2. **Follow established naming conventions**
3. **Update relevant README** files
4. **Maintain consistency** with existing docs
5. **Include clear navigation** and usage examples

---

## 🚨 Emergency Navigation

### **If You're Lost**
1. **Check the main README** - `README.md`
2. **Read the developer guide** - `DEVELOPER_ONBOARDING_GUIDE.md`
3. **Review organization summary** - `PROJECT_ORGANIZATION_SUMMARY.md`
4. **Check directory README files** - Each major directory has one
5. **Look at existing code patterns** - Study similar files

### **If You Need to Find Something**
1. **Use the directory structure** - Files are organized by purpose
2. **Check the quick reference** - Above in this guide
3. **Look at import patterns** - They show file locations
4. **Read the documentation** - README files explain organization
5. **Study existing examples** - Similar files show patterns

---

**🎓 This project represents a comprehensive research effort with production-ready implementations, scientific validation, and extensible architecture. The organization supports efficient development, easy collaboration, and long-term maintenance. Follow the established patterns and maintain the logical organization for best results.**
