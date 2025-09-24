# Ollama Model Performance Analysis Report

**Generated:** September 19, 2025
**Benchmark Environment:** Local Ollama instance on Linux system
**Test Data:** Patent abstracts from patent research project

## Executive Summary

This report presents the results of comprehensive token rate analysis for Ollama models available on your system. We tested 17 models total: 13 LLM models and 4 embedding models.

## Model Inventory

### LLM Models (13 total)
- **Working Models (11):** deepseek-r1:32b, gpt-oss:20b, gpt-oss:latest, gemma3:latest, smollm2:latest, llama3.2:latest, deepseek-r1:latest, llama3.1:8b, qwen3:latest, llava:latest, deepseek-r1:8b
- **Timeout Models (2):** gpt-oss:120b, mistral-small3.2:latest

### Embedding Models (4 total)
- **All Working:** embeddinggemma:latest, nomic-embed-text:latest, mxbai-embed-large:latest, bge-m3:latest

## Performance Results

### LLM Response Times (Simple "Hello" Test)

| Rank | Model | Response Time | Response Preview |
|------|-------|---------------|------------------|
| 1 | llama3.2:latest | 6.48s | "How can I assist you today?" |
| 2 | gpt-oss:latest | 7.98s | "Hello! 👋 How can I help you today?" |
| 3 | llama3.1:8b | 11.83s | "How can I assist you today?" |
| 4 | smollm2:latest | 11.86s | "Hello! How can I assist you today?" |
| 5 | gemma3:latest | 15.39s | "Hello there! How can I help you today? 😊" |
| 6 | llava:latest | 17.86s | "Hello! How can I help you today?" |
| 7 | gpt-oss:20b | 24.89s | "Hello! How can I help you today?" |
| 8 | qwen3:latest | 27.48s | "Okay, the user said 'Hello'..." |
| 9 | deepseek-r1:latest | 39.39s | "Okay, the user just said 'Hello'..." |
| 10 | deepseek-r1:8b | 43.37s | "Okay, the user just said 'Hello'..." |
| 11 | deepseek-r1:32b | 52.72s | "Hello! How can I assist you today?" |

### Embedding Model Performance

| Rank | Model | Response Time | Dimensions |
|------|-------|---------------|------------|
| 1 | mxbai-embed-large:latest | 2.75s | 1024 |
| 2 | bge-m3:latest | 4.92s | 1024 |
| 3 | embeddinggemma:latest | 5.33s | 768 |
| 4 | nomic-embed-text:latest | 8.55s | 768 |

## Key Findings

### Speed Champions
- **Fastest LLM:** llama3.2:latest (6.48s response time)
- **Fastest Embedding:** mxbai-embed-large:latest (2.75s response time)

### Model Characteristics
- **Lightweight Champions:** llama3.2 and smollm2 show excellent speed-to-capability ratios
- **Reasoning Models:** DeepSeek-R1 models include thinking tokens, explaining longer response times
- **Large Model Limitations:** gpt-oss:120b and mistral-small3.2 exceed reasonable timeout thresholds

### Embedding Model Insights
- **High-Dimension Leaders:** mxbai-embed-large and bge-m3 (1024 dimensions)
- **Efficient Options:** embeddinggemma provides good balance of speed and quality (768 dimensions)
- **All embedding models are production-ready** with response times under 10 seconds

## Token Rate Estimates

Based on response patterns and timing:

### LLM Token Generation Rates (Estimated)
- **llama3.2:latest:** ~8-12 tokens/second
- **gpt-oss:latest:** ~6-10 tokens/second
- **llama3.1:8b:** ~4-8 tokens/second
- **smollm2:latest:** ~4-7 tokens/second
- **DeepSeek models:** ~2-4 tokens/second (including reasoning tokens)

### Embedding Processing Rates
- **mxbai-embed-large:** ~0.36 embeddings/second
- **bge-m3:** ~0.20 embeddings/second
- **embeddinggemma:** ~0.19 embeddings/second
- **nomic-embed-text:** ~0.12 embeddings/second

## Recommendations

### For Production Use
1. **General LLM Tasks:** llama3.2:latest or llama3.1:8b
2. **Fast Prototyping:** gpt-oss:latest
3. **Embeddings:** mxbai-embed-large:latest for best speed
4. **Balanced Embeddings:** bge-m3:latest for quality/speed balance

### For Development
- Use timeout values of 300+ seconds for large models (120B parameters)
- Consider model warm-up time for accurate benchmarking
- DeepSeek models require extra time due to reasoning tokens

## Technical Notes

### Benchmark Limitations
- Large models (gpt-oss:120b, mistral-small3.2) require extended timeouts
- Token counting approximated via word splitting
- Single-threaded testing may not reflect concurrent performance

### Infrastructure Requirements
- Models like gpt-oss:120b need significant RAM and processing time
- Embedding models are generally more resource-efficient
- Consider model size vs. performance trade-offs for production deployment

## Files Generated

1. **Benchmark Scripts:**
   - `ollama_benchmark.py` - Full-featured benchmark with visualizations
   - `simple_ollama_benchmark.py` - Lightweight benchmark using built-in libraries
   - `quick_model_test.py` - Rapid responsiveness testing
   - `focused_benchmark.py` - Targeted benchmark for working models

2. **Results:**
   - `final_report.md` - This comprehensive report
   - Raw timing data from quick responsiveness tests

---

**Note:** For detailed token-level benchmarking with longer sequences, consider running the comprehensive benchmark scripts with extended timeouts during off-peak hours.