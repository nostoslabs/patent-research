# Patent Research with Embeddings - Usage Guide

## 🚀 Quick Start

All scripts are ready to run! Here's the recommended usage order:

### 1. View Interactive Visualizations
```bash
# Open the HTML files in your browser
open visualizations/umap_by_classification.html
open visualizations/tsne_by_classification.html
open visualizations/pca_explained_variance.html
```

### 2. Launch Apple Embedding Atlas (Interactive Exploration)
```bash
# Option 1: Use the shell script
./launch_atlas.sh

# Option 2: Run Python command
python -c "from launch_atlas import launch_atlas_directly; launch_atlas_directly('patent_atlas_enhanced.parquet')"
```

### 3. Run Semantic Search
```bash
python semantic_search.py
```

## 📁 Generated Files

### Data Files
- `patent_abstracts.jsonl` - Original 929 patent abstracts
- `patent_abstracts_with_embeddings.jsonl` - Patents with 768D embeddings (17MB)
- `patent_embeddings_atlas.parquet` - Parquet format for Atlas (5MB)
- `patent_atlas_enhanced.parquet` - Enhanced with coordinates and metadata (5MB)

### Analysis Results
- `reduction_results/` - Dimensionality reduction results
  - `pca_embeddings.npy` - 50D PCA embeddings
  - `tsne_embeddings.npy` - 2D t-SNE embeddings  
  - `umap_embeddings.npy` - 2D UMAP embeddings
  - `mds_embeddings.npy` - 2D MDS embeddings (subset)
  - `reduction_results.json` - Analysis metadata and clustering scores

### Visualizations
- `visualizations/*.html` - Interactive plots
  - Classification distribution
  - PCA explained variance
  - 2D/3D scatter plots colored by classification
  - Patent similarity heatmaps
  - Clustering comparisons

## 🔧 Scripts Overview

| Script | Purpose | Usage |
|--------|---------|-------|
| `download_patents.py` | Download & sample patents | `python download_patents.py` |
| `generate_embeddings.py` | Create embeddings with embeddinggemma | `python generate_embeddings.py` |
| `convert_to_parquet.py` | Convert to Atlas format | `python convert_to_parquet.py` |
| `dimensionality_reduction.py` | PCA, t-SNE, UMAP, MDS analysis | `python dimensionality_reduction.py` |
| `visualize_embeddings.py` | Generate interactive plots | `python visualize_embeddings.py` |
| `launch_atlas.py` | Setup Apple Embedding Atlas | `python launch_atlas.py` |
| `semantic_search.py` | Semantic search interface | `python semantic_search.py` |

## 📊 Analysis Results Summary

### Dataset
- **929 patents** with valid abstracts from BigPatent dataset
- **768-dimensional** embeddings using embeddinggemma model
- Average abstract length: ~3,108 characters

### Dimensionality Reduction
- **PCA**: 50 components explain 91.9% of variance
- **t-SNE**: 2D projection with KL divergence 0.509
- **UMAP**: 2D projection with good cluster separation
- **MDS**: 2D projection (500 sample subset due to computational cost)

### Clustering Results (Silhouette Scores)
- **UMAP + K-means (5 clusters)**: 0.868 (best)
- **UMAP + DBSCAN**: 0.836
- **t-SNE + K-means (3 clusters)**: 0.729
- **MDS + K-means (3 clusters)**: 0.602

## 🎯 Recommended Exploration Workflow

1. **Start with visualizations** - Open `umap_by_classification.html` to see the overall structure
2. **Use Apple Embedding Atlas** - Launch with `./launch_atlas.sh` for interactive exploration
3. **Try semantic search** - Run `python semantic_search.py` and search for:
   - "machine learning artificial intelligence"
   - "medical device heart surgery" 
   - "battery energy storage lithium"
4. **Find similar patents** - Use the `similar <patent_id>` command in semantic search
5. **Analyze classifications** - Use the `analyze` command to see intra/inter-class similarities

## 🔍 Search Tips

### Semantic Search Examples
```
> search machine learning neural network
> search renewable energy solar panels
> search drug delivery pharmaceutical
> similar patent_1824
> analyze
```

### Atlas Exploration
- **Zoom and pan** to explore clusters
- **Click points** to see patent details
- **Filter by classification** using the sidebar
- **Search by text** using the search bar

## ⚡ Performance Notes

- **Embedding generation**: ~0.28s per patent (uses local Ollama)
- **Search latency**: <1s for similarity search across 929 patents
- **Atlas loading**: ~2-3s for 929 patents with pre-computed coordinates
- **Memory usage**: ~200MB for full dataset in memory

## 🔄 Regenerating Data

To regenerate embeddings or analysis:

```bash
# Re-download patents (if needed)
python download_patents.py

# Regenerate embeddings
python generate_all_embeddings.py

# Rerun analysis
python dimensionality_reduction.py
python visualize_embeddings.py
python convert_to_parquet.py
python launch_atlas.py
```

## 🎨 Customization

### Adding More Embedding Models
1. Modify `generate_embeddings.py` to use different models
2. Update the `model_name` parameter in search scripts

### Different Visualization Parameters
- Edit `dimensionality_reduction.py` for different t-SNE perplexity, UMAP neighbors
- Modify `visualize_embeddings.py` for different plot styles

### Extended Analysis
- Add more clustering algorithms in `dimensionality_reduction.py`
- Implement custom similarity metrics in `semantic_search.py`

---

**Happy exploring! 🚀📊🔬**