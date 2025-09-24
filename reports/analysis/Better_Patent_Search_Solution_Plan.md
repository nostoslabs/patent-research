# Better Patent Search Solution Research & Implementation Plan

**Date:** September 12, 2025  
**Current Performance:** r=0.636 (OpenAI text-embedding-3-small)  
**Target Performance:** r=0.75-0.80  

---

## Executive Summary

Based on comprehensive research of the latest embedding models, hybrid search techniques, and reranking approaches, this plan outlines how to significantly improve patent search quality beyond the current OpenAI text-embedding-3-small implementation.

---

## Phase 1: Upgrade Embedding Model

### Option A: OpenAI text-embedding-3-large
- **MTEB Score:** 64.6% (vs 62.3% for small)
- **Dimensions:** 3072 (double the current 1536)
- **Cost:** $0.00013 per 1k tokens (6.5x more expensive but still minimal)
- **Expected Improvement:** ~5-10% better correlation
- **Implementation:** Direct API swap, minimal code changes
- **Pros:** Easy upgrade path, proven reliability
- **Cons:** Higher cost, still may not be sufficient

### Option B: NVIDIA NV-Embed-v2
- **MTEB Score:** 69.32% (current leader)
- **Architecture:** Fine-tuned Mistral 7B base
- **Expected Improvement:** 10-15% better than current
- **Implementation:** May require API access or special deployment
- **Pros:** Best general-purpose performance
- **Cons:** Deployment complexity, potential licensing issues

### Option C: Qwen3 Embedding 8B
- **MTEB Score:** 70.58 (multilingual leader)
- **Strengths:** Excellent for international patent searches
- **Expected Improvement:** 12-18% better, especially for non-English
- **Implementation:** Available via API providers
- **Pros:** Best multilingual performance
- **Cons:** Large model size, higher latency

---

## Phase 2: Implement Hybrid Search

### 2.1 Add BM25 Sparse Retrieval
**Purpose:** Capture exact technical term matches critical for patents

**Key Benefits:**
- Excellent for patent numbers (US12345678B2)
- Chemical formulas (C₂H₅OH)
- Specific technical terminology
- Abbreviations and acronyms

**Implementation:**
```python
# Pseudo-code for hybrid scoring
bm25_score = compute_bm25(query, document)
dense_score = cosine_similarity(query_embedding, doc_embedding)
final_score = alpha * dense_score + (1-alpha) * bm25_score
```

### 2.2 Three-Way Retrieval System
Based on IBM research showing optimal RAG performance:

1. **BM25:** Traditional keyword matching
2. **Dense Vectors:** Semantic similarity (current approach)
3. **Sparse Vectors:** Learned sparse representations (SPLADE)

**Fusion Methods:**
- Reciprocal Rank Fusion (RRF)
- Weighted linear combination
- Learned ensemble weights

**Expected Improvement:** 15-25% better recall and precision

---

## Phase 3: Advanced Reranking

### 3.1 Replace GPT with Specialized Rerankers

#### Top Options:

**Mixedbread mxbai-rerank-large-v2**
- **BEIR Score:** 57.49
- **Parameters:** 1.5B
- **Context:** 8k tokens (32k compatible)
- **License:** Apache 2.0 (open source)
- **Cost:** Self-hosted, no API fees

**BGE-reranker-v2-m3**
- **Strengths:** Multilingual support
- **Architecture:** Based on M3 backbone
- **Performance:** Excellent on technical documents
- **Deployment:** Hugging Face compatible

**Cohere Rerank 3 Nimble**
- **Speed:** Optimized for production
- **Languages:** 100+ supported
- **API:** Easy integration
- **Cost:** Usage-based pricing

### 3.2 ColBERT Late Interaction
**Architecture:** Token-level matching between query and documents

**Benefits for Patents:**
- Fine-grained technical term matching
- Better handling of long documents
- Preserves exact terminology importance

**Implementation Considerations:**
- Requires separate token embeddings
- Higher storage requirements
- Significantly better precision for technical queries

---

## Phase 4: Patent-Specific Optimizations

### 4.1 Domain-Specific Fine-Tuning
**Dataset Creation:**
- Use existing ground truth patent pairs
- Leverage patent citation networks
- Include examiner rejections/allowances

**Fine-Tuning Approach:**
- Contrastive learning on patent pairs
- Hard negative mining from similar but distinct patents
- Multi-task learning (classification + similarity)

### 4.2 Multi-Field Indexing Strategy
**Separate Embeddings for:**
- Title (high weight for exact matches)
- Abstract (balanced semantic understanding)
- Claims (technical precision)
- Description (comprehensive context)

**Field Weighting:**
```python
scores = {
    'title': 0.3 * title_similarity,
    'abstract': 0.4 * abstract_similarity,
    'claims': 0.2 * claims_similarity,
    'description': 0.1 * description_similarity
}
final_score = sum(scores.values())
```

### 4.3 Query Understanding Enhancements
**Patent-Specific Features:**
- CPC/IPC classification detection
- Technical term expansion using patent ontologies
- Citation network analysis for related patents
- Inventor/assignee entity recognition

---

## Implementation Timeline

### Week 1-2: Quick Wins
- [ ] Upgrade to text-embedding-3-large
- [ ] Implement basic BM25 hybrid search
- [ ] Replace GPT with BGE reranker
- **Expected Gain:** r=0.68-0.70

### Week 3-4: Core Improvements
- [ ] Full hybrid search pipeline with RRF
- [ ] Test and integrate best reranking model
- [ ] Implement basic query understanding
- **Expected Gain:** r=0.72-0.74

### Month 2: Advanced Features
- [ ] ColBERT late interaction integration
- [ ] Multi-field indexing implementation
- [ ] Patent-specific query processing
- **Expected Gain:** r=0.75-0.78

### Month 3: Optimization & Fine-Tuning
- [ ] Fine-tune on patent-specific data
- [ ] A/B testing and parameter optimization
- [ ] Production deployment and monitoring
- **Target Achievement:** r=0.78-0.80

---

## Evaluation Framework

### Metrics to Track
1. **Correlation with Ground Truth:** Primary metric (current: r=0.636)
2. **Mean Reciprocal Rank (MRR):** For known-item searches
3. **nDCG@10:** Normalized discounted cumulative gain
4. **Latency:** Query response time
5. **Cost per Query:** API and compute costs

### Testing Protocol
1. Maintain 500-pair ground truth test set
2. Weekly evaluation runs
3. A/B testing on 10% of production traffic
4. User satisfaction surveys

---

## Cost-Benefit Analysis

### Current Costs (per 100K searches/year)
- OpenAI text-embedding-3-small: ~$10
- GPT reranking: ~$500-2000
- **Total:** ~$510-2010

### Proposed Solution Costs
- text-embedding-3-large: ~$65
- BGE reranker (self-hosted): ~$50 (compute)
- BM25 (integrated): ~$10 (compute)
- **Total:** ~$125

### ROI
- **Cost Reduction:** 75-94% lower than current
- **Quality Improvement:** 22-26% better correlation
- **User Experience:** Significantly better relevance

---

## Risk Mitigation

### Technical Risks
- **Model compatibility:** Test all models with existing infrastructure
- **Latency increases:** Implement caching and optimization
- **Storage requirements:** Plan for increased index size

### Business Risks
- **User adaptation:** Gradual rollout with feedback loops
- **Cost overruns:** Start with self-hosted open-source models
- **Quality regression:** Maintain rollback capability

---

## Success Criteria

1. **Primary:** Achieve r≥0.75 correlation with ground truth
2. **Secondary:** Maintain <200ms query latency
3. **Tertiary:** Reduce reranking costs by >90%
4. **User Metric:** >20% improvement in click-through rate

---

## Recommended Starting Point

**Immediate Action (Week 1):**
1. Test OpenAI text-embedding-3-large (simple upgrade)
2. Implement BM25 hybrid search (proven technique)
3. Deploy BGE-reranker-base (quick win on cost)

These three changes alone should improve correlation to ~r=0.70 while reducing costs by 80%.

---

## Conclusion

The current OpenAI text-embedding-3-small performance (r=0.636) can be significantly improved through a combination of:
- Better embedding models (up to r=0.70)
- Hybrid search techniques (adds 0.05-0.08)
- Advanced reranking (adds 0.03-0.05)
- Patent-specific optimizations (adds 0.02-0.04)

The full implementation should achieve r=0.75-0.80, representing a 20-25% improvement in search quality while reducing operational costs.