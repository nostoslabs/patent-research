# Comprehensive Embedding Model Benchmark Report

**Ground Truth Dataset**: 9988 LLM-evaluated patent pairs
**Models Evaluated**: 3

## Model Performance Rankings

| Model | Pearson r | Spearman ρ | p-value | Sample Size | Avg Embedding Sim | Avg LLM Score |
|-------|-----------|------------|---------|-------------|-------------------|---------------|
| **text-embedding-3-small** | 0.556 | 0.559 | 0.00e+00 | 8,494 | 0.312 | 0.153 |
| **nomic-embed-text** | 0.516 | 0.509 | 0.00e+00 | 9,988 | 0.472 | 0.150 |
| **bge-m3** | 0.137 | 0.200 | 2.87e-27 | 6,176 | 0.528 | 0.155 |

## Key Findings

1. **Best Performing Model**: text-embedding-3-small (r = 0.556)
2. **Largest Coverage**: nomic-embed-text (9,988 pairs)

## Statistical Significance

All correlations with p < 0.05 are statistically significant:
- **bge-m3**: r = 0.137, p = 2.87e-27
- **nomic-embed-text**: r = 0.516, p = 0.00e+00
- **text-embedding-3-small**: r = 0.556, p = 0.00e+00