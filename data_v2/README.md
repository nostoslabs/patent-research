# Data Organization

This directory contains the consolidated and organized patent research data.

## Directory Structure

### 📊 Master Data Files
- `master_patent_embeddings.jsonl` (875MB) - Consolidated patent embeddings across all models
- `ground_truth_similarities.jsonl` (11.5MB) - LLM-evaluated patent similarity pairs
- `patents_needing_bge_m3.jsonl` - Patents requiring BGE-M3 embeddings

### 🗂️ Atlas Data (`atlas_data/`)
Parquet files for Apple Embedding Atlas visualization:
- `patent_atlas_enhanced.parquet` - Enhanced patent data for Atlas
- `patent_embeddings_atlas.parquet` - Embedding data for Atlas
- `patent_metadata_only.parquet` - Metadata-only Atlas data
- `model_comparison_*.parquet` - Model comparison visualizations
- `complete_coverage_enhanced_atlas.parquet` - Complete coverage data
- `text_quality_enhanced_atlas.parquet` - Text quality analysis

### 🔄 Batch Processing (`batch_processing/`)
OpenAI API batch processing files:
- `openai_batch_*.jsonl` - Batch request and result files
- `openai_batch_info.json` - Batch processing metadata
- `openai_batch_monitor.log` - Processing logs

### 📚 Legacy Data (`legacy_data/`)
Archived data from previous versions:
- `old_partial_files/` - Legacy partial data files
- `reduction_results/` - Dimensionality reduction results
- `visualizations/` - Legacy visualization outputs

### 📋 Metadata (`metadata/`)
Data catalogs and processing information:
- `data_catalog.json` - Master data index and statistics
- `model_versions.json` - Model version information
- `processing_history.json` - Data processing history
- `validation_results.json` - Data validation results

## Data Statistics

### Current Scale
- **40,403 patents** in master dataset
- **52,209 embedding vectors** across 3 models
- **9,988 ground truth pairs** for evaluation
- **3 models available**: bge-m3, openai_text-embedding-3-small, nomic-embed-text

### File Sizes
- Master embeddings: 875MB
- Ground truth: 11.5MB
- Atlas data: ~200MB total
- Batch processing: ~50MB

## Usage

### Loading Master Data
```python
import json

# Load patent embeddings
with open('master_patent_embeddings.jsonl', 'r') as f:
    for line in f:
        patent = json.loads(line)
        # Process patent data
```

### Accessing Ground Truth
```python
# Load ground truth similarities
with open('ground_truth_similarities.jsonl', 'r') as f:
    for line in f:
        pair = json.loads(line)
        # Process similarity pair
```

### Atlas Visualization
```python
import pandas as pd

# Load Atlas data
atlas_data = pd.read_parquet('atlas_data/patent_atlas_enhanced.parquet')
```

## Data Quality

### Validation Status
- ✅ All embeddings validated for consistency
- ✅ Ground truth pairs verified for quality
- ✅ Metadata cataloged and indexed
- ✅ Processing history tracked

### Model Coverage
| Model | Patents | Dimensions | Status |
|-------|---------|------------|-------|
| nomic-embed-text | 40,220 | 768 | Complete |
| bge-m3 | 6,139 | 1024 | Complete |
| openai_text-embedding-3-small | 5,850 | 1536 | Complete |

## Maintenance

### Regular Tasks
- Monitor batch processing logs
- Validate new embeddings against master index
- Update metadata catalogs
- Archive completed processing runs

### Backup Strategy
- Master files backed up to `archive/` directory
- Incremental backups for large datasets
- Git LFS for version control of large files
