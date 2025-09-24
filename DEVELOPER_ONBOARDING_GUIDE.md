# Developer Onboarding Guide

**Patent Research Project - Complete Developer Guide**
*For new team members and AI coding agents*

---

## 🎯 Project Overview

This is a comprehensive research project on **neural patent similarity search** using state-of-the-art embedding models. The project evaluates 4 embedding models across 100,000+ patents with extensive ground truth validation using LLM evaluation.

### Key Research Achievements
- **91,725+ patent embeddings** across 4 models (nomic-embed-text, bge-m3, embeddinggemma, mxbai-embed-large)
- **9,988 ground truth pairs** with LLM evaluation (66% complete)
- **Production-ready pipeline** with embedding generation and evaluation
- **Scientific analysis** with statistical validation and performance comparison

---

## 📁 Project Structure Overview

```
patent_research/
├── README.md                           # Main project documentation
├── DEVELOPER_ONBOARDING_GUIDE.md      # This file - start here!
├── PROJECT_ORGANIZATION_SUMMARY.md     # Complete organization details
├── pyproject.toml                      # Python dependencies
├── uv.lock                            # Dependency lock file
├── ruff.toml                          # Code formatting configuration
│
├── code/                              # 🐍 ALL PYTHON CODE HERE
│   ├── core/                          # Core system components
│   ├── data_processing/               # Data handling & transformation
│   ├── analysis/                      # Scientific analysis & evaluation
│   ├── experiments/                   # Research workflows
│   ├── benchmarks/                    # Performance testing
│   ├── monitoring/                     # System monitoring
│   ├── utilities/                      # Helper tools
│   └── README.md                      # Code organization guide
│
├── data/                              # Original data (preserved)
├── data_v2/                           # 🗄️ CONSOLIDATED DATA HERE
│   ├── master_patent_embeddings.jsonl # 875MB - Main dataset
│   ├── ground_truth_similarities.jsonl # 11.5MB - LLM evaluations
│   ├── atlas_data/                    # Visualization data
│   ├── batch_processing/              # OpenAI API batch files
│   └── README.md                      # Data organization guide
│
├── reports/                           # 📊 ALL DOCUMENTATION HERE
│   ├── scientific/                    # Main research papers
│   ├── technical/                     # Technical documentation
│   ├── analysis/                      # Analysis outputs
│   ├── images/                        # Generated visualizations
│   └── README.md                      # Report organization guide
│
├── figures/                           # Generated plots and charts
├── analysis/                          # Analysis results and outputs
├── results/                           # Experiment results
├── scripts/                           # Utility scripts
└── archive/                           # Data backups
```

---

## 🚀 Quick Start for New Developers

### 1. **Environment Setup**
```bash
# Clone and navigate to project
cd patent_research

# Install dependencies (recommended: use uv)
uv sync

# OR install with pip
pip install -r requirements.txt

# Install Ollama embedding models
ollama pull nomic-embed-text
ollama pull bge-m3
ollama pull embeddinggemma
ollama pull mxbai-embed-large
```

### 2. **Understanding the Codebase**
```bash
# Start with the main documentation
cat README.md                    # Project overview
cat DEVELOPER_ONBOARDING_GUIDE.md  # This guide
cat PROJECT_ORGANIZATION_SUMMARY.md # Organization details

# Understand code structure
cat code/README.md              # Code organization
cat reports/README.md            # Report structure
cat data_v2/README.md            # Data organization
```

### 3. **Key Entry Points**
- **Main research paper**: `reports/scientific/SCIENTIFIC_PATENT_SEARCH_ANALYSIS.md`
- **Project summary**: `reports/scientific/PROJECT_SUMMARY.md`
- **Core system**: `code/core/` directory
- **Data access**: `data_v2/master_patent_embeddings.jsonl`

---

## 🐍 Code Organization Deep Dive

### **Core System (`code/core/`)**
**Purpose**: Core system components and main functionality
**Key Files**:
- `llm_provider_factory.py` - Multi-provider LLM system with PydanticAI
- `cross_encoder_reranker.py` - Two-stage retrieval implementation
- `reranker_enhancement_plan.py` - Advanced multi-reranker system
- `semantic_search.py` - Semantic search functionality

**Usage**:
```python
# Import core components
from code.core.llm_provider_factory import LLMProviderFactory
from code.core.cross_encoder_reranker import EmbeddingSearchEngine
```

### **Data Processing (`code/data_processing/`)**
**Purpose**: Data ingestion, transformation, and preparation
**Key Files**:
- `download_large_diverse_patents.py` - Large-scale patent dataset creation
- `generate_embeddings_multimodel.py` - Multi-model embedding generation
- `data_consolidation.py` - Data consolidation and organization
- `create_atlas_data.py` - Atlas visualization data creation

**Usage**:
```bash
# Download patents
python code/data_processing/download_large_diverse_patents.py 10k

# Generate embeddings
python code/data_processing/generate_embeddings_multimodel.py
```

### **Analysis (`code/analysis/`)**
**Purpose**: Scientific analysis and evaluation
**Key Files**:
- `scientific_analysis.py` - Statistical analysis and visualization
- `model_performance_analyzer.py` - Model performance comparison
- `comprehensive_evaluation.py` - End-to-end evaluation pipeline

**Usage**:
```bash
# Run scientific analysis
python code/analysis/scientific_analysis.py

# Model performance analysis
python code/analysis/model_performance_analyzer.py
```

### **Experiments (`code/experiments/`)**
**Purpose**: Experimental workflows and ground truth generation
**Key Files**:
- `run_multimodel_experiments.py` - Multi-model experiment runner
- `ground_truth_generator.py` - LLM-based ground truth generation
- `run_chunking_experiment.py` - Chunking experiment runner

**Usage**:
```bash
# Run multi-model experiments
python code/experiments/run_multimodel_experiments.py

# Generate ground truth
python code/experiments/ground_truth_generator.py
```

### **Benchmarks (`code/benchmarks/`)**
**Purpose**: Performance benchmarking and comparison
**Key Files**:
- `comprehensive_embedding_benchmark.py` - Comprehensive embedding benchmarks
- `fair_comparison_benchmark.py` - Fair comparison benchmarks
- `google_patents_baseline.py` - Google Patents baseline comparison

**Usage**:
```bash
# Run benchmarks
python code/benchmarks/comprehensive_embedding_benchmark.py
```

### **Monitoring (`code/monitoring/`)**
**Purpose**: System monitoring and progress tracking
**Key Files**:
- `monitor_experiments.py` - Experiment monitoring
- `monitor_ground_truth_progress.py` - Ground truth progress tracking
- `monitor_all_batches.py` - Batch processing monitoring

**Usage**:
```bash
# Monitor experiments
python code/monitoring/monitor_experiments.py
```

### **Utilities (`code/utilities/`)**
**Purpose**: Helper tools and utilities
**Key Files**:
- `create_visualizations.py` - Visualization creation
- `batch_manager.py` - Batch processing management
- `launch_atlas.py` - Atlas launch utility

**Usage**:
```bash
# Create visualizations
python code/utilities/create_visualizations.py

# Launch Atlas
python code/utilities/launch_atlas.py
```

---

## 🗄️ Data Organization Guide

### **Master Data Files**
- `data_v2/master_patent_embeddings.jsonl` (875MB) - Consolidated patent embeddings
- `data_v2/ground_truth_similarities.jsonl` (11.5MB) - LLM-evaluated similarity pairs

### **Data Statistics**
- **40,403 patents** in master dataset
- **52,209 embedding vectors** across 3 models
- **9,988 ground truth pairs** for evaluation
- **3 models available**: bge-m3, openai_text-embedding-3-small, nomic-embed-text

### **Loading Data**
```python
import json

# Load patent embeddings
with open('data_v2/master_patent_embeddings.jsonl', 'r') as f:
    for line in f:
        patent = json.loads(line)
        # Process patent data

# Load ground truth similarities
with open('data_v2/ground_truth_similarities.jsonl', 'r') as f:
    for line in f:
        pair = json.loads(line)
        # Process similarity pair
```

---

## 📊 Reports and Documentation

### **Scientific Papers (`reports/scientific/`)**
- `SCIENTIFIC_PATENT_SEARCH_ANALYSIS.md` - Main research paper
- `PROJECT_SUMMARY.md` - Complete project overview
- `FINAL_ANALYSIS_REPORT.md` - Final comprehensive analysis

### **Technical Documentation (`reports/technical/`)**
- `MODEL_PERFORMANCE_ANALYSIS.md` - Model performance comparison
- `RERANKER_PERFORMANCE_ANALYSIS.md` - Reranker evaluation
- `COMPREHENSIVE_PATENT_SEARCH_ANALYSIS.md` - Comprehensive search analysis

### **Analysis Outputs (`reports/analysis/`)**
- `Better_Patent_Search_Solution_Plan.md` - Solution planning
- `EXPERIMENT_SUMMARY.md` - Experiment results

---

## 🔧 Common Development Tasks

### **Running Experiments**
```bash
# Multi-model experiments
python code/experiments/run_multimodel_experiments.py

# Ground truth generation
python code/experiments/ground_truth_generator.py

# Chunking experiments
python code/experiments/run_chunking_experiment.py
```

### **Data Processing**
```bash
# Download patents
python code/data_processing/download_large_diverse_patents.py 10k

# Generate embeddings
python code/data_processing/generate_embeddings_multimodel.py

# Consolidate data
python code/data_processing/data_consolidation.py
```

### **Analysis and Evaluation**
```bash
# Scientific analysis
python code/analysis/scientific_analysis.py

# Model performance analysis
python code/analysis/model_performance_analyzer.py

# Comprehensive evaluation
python code/analysis/comprehensive_evaluation.py
```

### **Monitoring and Utilities**
```bash
# Monitor experiments
python code/monitoring/monitor_experiments.py

# Create visualizations
python code/utilities/create_visualizations.py

# Launch Atlas
python code/utilities/launch_atlas.py
```

---

## 🤖 AI Coding Agent Guidelines

### **File Location Patterns**
- **Core system code**: `code/core/`
- **Data processing**: `code/data_processing/`
- **Analysis tools**: `code/analysis/`
- **Experiments**: `code/experiments/`
- **Benchmarks**: `code/benchmarks/`
- **Monitoring**: `code/monitoring/`
- **Utilities**: `code/utilities/`

### **Import Patterns**
```python
# Core system imports
from code.core.llm_provider_factory import LLMProviderFactory
from code.core.cross_encoder_reranker import EmbeddingSearchEngine

# Data processing imports
from code.data_processing.generate_embeddings_multimodel import EmbeddingGenerator

# Analysis imports
from code.analysis.scientific_analysis import ScientificAnalyzer
```

### **Data Access Patterns**
```python
# Master data files
MASTER_EMBEDDINGS = "data_v2/master_patent_embeddings.jsonl"
GROUND_TRUTH = "data_v2/ground_truth_similarities.jsonl"

# Atlas data
ATLAS_DATA = "data_v2/atlas_data/patent_atlas_enhanced.parquet"
```

### **Common Script Patterns**
```python
# Standard script structure
import sys
import os
sys.path.append('code')

# Import from organized directories
from core.llm_provider_factory import LLMProviderFactory
from analysis.scientific_analysis import ScientificAnalyzer

# Data file paths
DATA_DIR = "data_v2"
MASTER_FILE = os.path.join(DATA_DIR, "master_patent_embeddings.jsonl")
```

---

## 🚨 Important Notes for AI Agents

### **DO NOT**
- ❌ Create Python files in the root directory
- ❌ Mix files from different categories
- ❌ Delete or move files without understanding the organization
- ❌ Create new directories without following the established pattern

### **DO**
- ✅ Use the organized directory structure
- ✅ Follow the established naming conventions
- ✅ Check existing README files for guidance
- ✅ Maintain the logical organization

### **File Organization Rules**
1. **All Python code** goes in `code/` with appropriate subdirectory
2. **All documentation** goes in `reports/` with appropriate subdirectory
3. **All data** goes in `data_v2/` with appropriate subdirectory
4. **Generated figures** go in `figures/` or `reports/images/`

---

## 📚 Key Documentation Files

### **Start Here**
1. `README.md` - Main project overview
2. `DEVELOPER_ONBOARDING_GUIDE.md` - This guide
3. `PROJECT_ORGANIZATION_SUMMARY.md` - Complete organization details

### **Code Documentation**
1. `code/README.md` - Code organization guide
2. `reports/README.md` - Report structure guide
3. `data_v2/README.md` - Data organization guide

### **Research Documentation**
1. `reports/scientific/SCIENTIFIC_PATENT_SEARCH_ANALYSIS.md` - Main research paper
2. `reports/scientific/PROJECT_SUMMARY.md` - Project overview
3. `reports/technical/MODEL_PERFORMANCE_ANALYSIS.md` - Model analysis

---

## 🎯 Next Steps for New Developers

### **Week 1: Understanding**
1. Read all README files
2. Explore the code organization
3. Understand the data structure
4. Review the main research papers

### **Week 2: Hands-on**
1. Run example scripts
2. Generate some embeddings
3. Create visualizations
4. Run analysis tools

### **Week 3: Contribution**
1. Identify areas for improvement
2. Add new features following organization patterns
3. Update documentation as needed
4. Maintain the established structure

---

## 📞 Getting Help

### **Documentation**
- Check README files in each directory
- Review the organization summary
- Look at existing code patterns

### **Code Examples**
- Study existing scripts in each category
- Follow established import patterns
- Use the organized directory structure

### **Data Access**
- Use the master data files in `data_v2/`
- Follow the established data loading patterns
- Check the data organization guide

---

**🎓 This project represents a comprehensive research effort with production-ready implementations, scientific validation, and extensible architecture. The organization supports efficient development, easy collaboration, and long-term maintenance.**
