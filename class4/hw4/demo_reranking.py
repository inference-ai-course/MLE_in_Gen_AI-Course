"""
Demo: Compare Regular Search vs. Reranking
===========================================
This script demonstrates the improvement from using cross-encoder reranking.
"""

import requests
import time


def compare_search_methods(query: str, k: int = 3, base_url: str = "http://localhost:8000"):
    """
    Compare regular search with reranked search.

    Args:
        query: Search query
        k: Number of results to return
        base_url: API base URL
    """
    print("="*80)
    print(f"Query: {query}")
    print("="*80)

    # Regular search
    print("\n1. REGULAR SEARCH (FAISS only)")
    print("-"*80)
    start = time.time()
    response = requests.get(
        f"{base_url}/search",
        params={"q": query, "k": k}
    )
    regular_time = time.time() - start
    regular_results = response.json()

    for result in regular_results['results']:
        print(f"\nRank {result['rank']} (Similarity: {result['similarity']:.4f})")
        print(f"Paper: {result['paper_title'][:70]}...")
        print(f"Text: {result['chunk'][:200]}...")

    print(f"\nResponse time: {regular_time*1000:.2f}ms")

    # Reranked search
    print("\n\n2. RERANKED SEARCH (FAISS + Cross-Encoder)")
    print("-"*80)
    start = time.time()
    response = requests.get(
        f"{base_url}/search/rerank",
        params={"q": query, "k": k, "initial_k": 20}
    )
    rerank_time = time.time() - start
    rerank_results = response.json()

    for result in rerank_results['results']:
        print(f"\nRank {result['rank']} (CE Score: {result['similarity']:.4f})")
        print(f"Paper: {result['paper_title'][:70]}...")
        print(f"Text: {result['chunk'][:200]}...")

    print(f"\nResponse time: {rerank_time*1000:.2f}ms")

    # Comparison
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print(f"Regular search time:  {regular_time*1000:.2f}ms")
    print(f"Reranked search time: {rerank_time*1000:.2f}ms")
    print(f"Time overhead:        {(rerank_time-regular_time)*1000:.2f}ms ({(rerank_time/regular_time-1)*100:.1f}% slower)")
    print("\nNote: Reranking is slower but generally provides more accurate results.")
    print("="*80)


def main():
    """Run comparison demos"""
    base_url = "http://localhost:8000"

    # Check if server is running
    try:
        response = requests.get(f"{base_url}/health", timeout=2)
        print("✓ Server is running!\n")
    except requests.exceptions.ConnectionError:
        print("\n✗ Server is not running!")
        print("\nPlease start the server first:")
        print("  python main.py\n")
        return

    # Demo queries
    queries = [
        "What are transformer models in natural language processing?",
        "How does BERT improve language understanding?",
        "Explain attention mechanisms in neural networks",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n\n{'#'*80}")
        print(f"# Demo {i}")
        print(f"{'#'*80}\n")
        compare_search_methods(query, k=3, base_url=base_url)

        if i < len(queries):
            print("\n\nPress Enter to continue to next demo...")
            input()

    print("\n\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
Reranking improves search quality by:
1. First retrieving more candidates (e.g., 20) with fast FAISS search
2. Then using a cross-encoder to re-score and select the best k results

Trade-offs:
- ✓ Better precision and relevance
- ✓ More context-aware scoring
- ✗ Slower (requires cross-encoder inference)
- ✗ More memory (loads additional model)

Recommendation:
- Use /search for fast, approximate results
- Use /search/rerank when quality > speed is important
    """)


if __name__ == "__main__":
    main()
