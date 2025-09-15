# Chunking Strategy Performance Analysis Report

Generated on: 2025-09-12 06:38:34
Dataset: 25 patents

============================================================

## 🎯 Clustering Performance Rankings

Rank  Strategy                       Silhouette Score Best Config
--------------------------------------------------------------------------------
🥇     original                       0.534           3 🟡 Good

## 📊 Semantic Similarity Preservation

Strategy                            Correlation with Original Quality
--------------------------------------------------------------------------------

## ⚡ Processing Efficiency

**Original Embedding**: 0.15s avg (±0.08s)

**Chunking Strategies**:
  fixed_768: 0.45s avg (3.0x overhead)
  sentence_boundary_768: 0.49s avg (3.2x overhead)
  sentence_boundary_512: 0.57s avg (3.8x overhead)
  fixed_512: 0.57s avg (3.8x overhead)
  overlapping_550: 0.60s avg (4.0x overhead)
  semantic: 7.20s avg (47.9x overhead)

## 📏 Chunk Statistics

Strategy                  Avg Chunks   Avg Size     Avg Tokens  
----------------------------------------------------------------------
fixed_512                 6.0          203          454         
fixed_768                 4.0          203          681         
overlapping_550           6.0          203          496         
sentence_boundary_512     6.0          203          448         
sentence_boundary_768     4.5          203          597         
semantic                  38.5         155          70          

## 🎯 Recommendations

**Best Overall Performance**: original
- Silhouette Score: 0.534
- Configuration: {'n_clusters': 3, 'silhouette_score': 0.5335870735942791, 'inertia': 326687.76378193375}
- ✅ Good clustering quality