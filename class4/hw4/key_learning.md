## Key Learnings

### Performance Insights

1. **Speed vs Quality Trade-off**
   - Basic search is 2-3x faster but reranking provides measurably better precision
   - For production systems, consider hybrid approach: basic for simple queries, rerank for complex

2. **Query Type Matters**
   - Factual queries: All endpoints perform similarly
   - Conceptual queries: Reranking shows clear advantage
   - Specific technical queries: Metadata filtering helps narrow results

3. **Cosine Similarity (IndexFlatIP)**
   - Using normalized embeddings with inner product gives better semantic matching
   - Similarity scores are more interpretable (0-1 range)
   - More robust than L2 distance for text embeddings

4. **Cross-Encoder Reranking**
   - Significant quality improvement for ambiguous queries
   - Latency increase is acceptable for most applications
   - Particularly effective for comparative and conceptual queries

5. **Metadata Filtering**
   - Year filtering effectively narrows results to recent research
   - Recency boost (0.3) provides subtle preference for newer papers
   - Useful for literature reviews and staying current


## Recommendations

### When to Use Each Endpoint

**Use `/search` when:**
- Speed is critical (< 50ms required)
- Query is straightforward
- Exploring broad topics
- Building autocomplete or suggestions

**Use `/search/rerank` when:**
- Quality is more important than speed
- Query is complex or nuanced
- Comparative questions
- Final result selection for user-facing features

**Use `/search/metadata` when:**
- Need recent papers (year_min filter)
- Tracking specific authors
- Building literature reviews
- Time-bounded research questions

### Optimization Tips

1. **Caching:** Cache frequent queries for basic search
2. **Hybrid Approach:** Use basic search first, rerank top-k if needed
3. **Batch Processing:** For offline analysis, use reranking by default
4. **Monitor Latency:** Set SLAs: basic < 100ms, rerank < 200ms
5. **A/B Testing:** Compare endpoints for your specific use case

