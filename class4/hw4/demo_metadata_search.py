"""
Demo: Metadata-Based Search
============================
This script demonstrates how to use metadata filtering and boosting
to improve search results based on publication year, authors, and recency.
"""

import requests
import json
from typing import Dict, List


def print_results(results: List[Dict], title: str):
    """Pretty print search results"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")

    for result in results:
        metadata = result.get('metadata', {})
        print(f"\nRank {result['rank']} (Similarity: {result['similarity']:.4f})")
        print(f"Paper: {result['paper_title'][:70]}...")

        # Print metadata if available
        if 'published_year' in metadata:
            print(f"Year: {metadata['published_year']}")
        if 'authors' in metadata and metadata['authors']:
            authors_str = ', '.join(metadata['authors'][:3])
            if len(metadata['authors']) > 3:
                authors_str += f" (and {len(metadata['authors']) - 3} more)"
            print(f"Authors: {authors_str}")

        print(f"Text: {result['chunk'][:150]}...")


def demo_year_filtering(base_url: str):
    """Demo: Filter results by publication year"""
    query = "transformer models in natural language processing"

    print(f"\n{'#'*80}")
    print(f"# DEMO 1: Year Filtering")
    print(f"# Query: {query}")
    print(f"{'#'*80}")

    # Search without year filter
    print("\n1. WITHOUT YEAR FILTER (all years)")
    response = requests.get(
        f"{base_url}/search",
        params={"q": query, "k": 3}
    )
    results = response.json()['results']
    print_results(results, "All Papers")

    # Search with recent papers only
    print("\n\n2. WITH YEAR FILTER (2020 and later)")
    response = requests.get(
        f"{base_url}/search/metadata",
        params={
            "q": query,
            "k": 3,
            "year_min": 2020
        }
    )
    results = response.json()['results']
    print_results(results, "Recent Papers Only (≥2020)")


def demo_recency_boost(base_url: str):
    """Demo: Boost scores for recent papers"""
    query = "attention mechanisms"

    print(f"\n\n{'#'*80}")
    print(f"# DEMO 2: Recency Boosting")
    print(f"# Query: {query}")
    print(f"{'#'*80}")

    # Search without boost
    print("\n1. WITHOUT RECENCY BOOST")
    response = requests.get(
        f"{base_url}/search",
        params={"q": query, "k": 5}
    )
    results = response.json()['results']
    print_results(results, "No Recency Boost")

    # Search with moderate boost
    print("\n\n2. WITH MODERATE RECENCY BOOST (0.3)")
    response = requests.get(
        f"{base_url}/search/metadata",
        params={
            "q": query,
            "k": 5,
            "recency_boost": 0.3
        }
    )
    results = response.json()['results']
    print_results(results, "Moderate Recency Boost (newer papers ranked higher)")

    # Search with strong boost
    print("\n\n3. WITH STRONG RECENCY BOOST (0.8)")
    response = requests.get(
        f"{base_url}/search/metadata",
        params={
            "q": query,
            "k": 5,
            "recency_boost": 0.8
        }
    )
    results = response.json()['results']
    print_results(results, "Strong Recency Boost (heavily favor recent papers)")


def demo_author_filtering(base_url: str):
    """Demo: Filter results by author names"""
    query = "language models"

    print(f"\n\n{'#'*80}")
    print(f"# DEMO 3: Author Filtering")
    print(f"# Query: {query}")
    print(f"{'#'*80}")

    # First, get some results to see available authors
    print("\n1. ALL AUTHORS")
    response = requests.get(
        f"{base_url}/search",
        params={"q": query, "k": 5}
    )
    results = response.json()['results']
    print_results(results, "All Authors")

    # Extract some author names from results
    if results:
        # Get first author from first result
        first_authors = []
        for result in results:
            # Make a request to get full metadata
            # In real scenario, you'd know the author name you want to filter by
            print(f"\n\nNote: In a real scenario, you would filter by known author names.")
            print(f"Example: /search/metadata?q={query}&authors=Vaswani")
            break


def demo_combined_filtering(base_url: str):
    """Demo: Combine multiple filters"""
    query = "BERT and transformers"

    print(f"\n\n{'#'*80}")
    print(f"# DEMO 4: Combined Filtering")
    print(f"# Query: {query}")
    print(f"{'#'*80}")

    print("\n1. COMBINED: Recent papers (2019+) with recency boost (0.4)")
    response = requests.get(
        f"{base_url}/search/metadata",
        params={
            "q": query,
            "k": 5,
            "year_min": 2019,
            "recency_boost": 0.4
        }
    )
    results = response.json()['results']
    print_results(results, "Recent + Boosted")

    print("\n\n2. COMBINED: Papers from 2018-2020 only")
    response = requests.get(
        f"{base_url}/search/metadata",
        params={
            "q": query,
            "k": 5,
            "year_min": 2018,
            "year_max": 2020
        }
    )
    results = response.json()['results']
    print_results(results, "2018-2020 Only")


def main():
    """Run all metadata search demos"""
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

    print(f"\n{'='*80}")
    print(f"METADATA SEARCH DEMONSTRATIONS")
    print(f"{'='*80}")
    print(f"""
This demo shows how metadata filtering and boosting can improve search results:

1. Year Filtering: Find only papers from specific years
2. Recency Boosting: Rank newer papers higher
3. Author Filtering: Find papers by specific authors
4. Combined: Use multiple filters together
    """)

    # Run demos
    demo_year_filtering(base_url)

    input("\n\nPress Enter to continue to next demo...")
    demo_recency_boost(base_url)

    input("\n\nPress Enter to continue to next demo...")
    demo_combined_filtering(base_url)

    print(f"\n\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print("""
Metadata filtering and boosting provides:

✓ Year Filtering: Focus on recent or historical papers
✓ Recency Boost: Automatically prefer newer research
✓ Author Filtering: Find papers by specific researchers
✓ Flexible Combination: Mix and match filters as needed

Use Cases:
- Literature review: Find recent papers on a topic
- Following researchers: Track specific authors' work
- Historical analysis: Compare papers from different eras
- Citation context: Find contemporary related work

API Examples:
- Recent only: /search/metadata?q=transformers&year_min=2020
- With boost: /search/metadata?q=attention&recency_boost=0.5
- By author: /search/metadata?q=NLP&authors=Vaswani,Devlin
- Combined: /search/metadata?q=BERT&year_min=2019&year_max=2021&recency_boost=0.3
    """)


if __name__ == "__main__":
    main()
