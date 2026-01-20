"""
RAG Pipeline for arXiv cs.CL Papers
=====================================
This script implements a complete Retrieval-Augmented Generation pipeline:
1. Data Collection: Download 50 arXiv cs.CL papers
2. Text Extraction: Extract text from PDFs using PyMuPDF
3. Text Chunking: Split into chunks of d512 tokens
4. Embedding Generation: Create embeddings using sentence-transformers
5. FAISS Index: Build and save a FAISS index for efficient retrieval
"""

import os
import warnings

# Suppress multiprocessing resource tracker warnings
warnings.filterwarnings("ignore", category=UserWarning, module="resource_tracker")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# Disable PyTorch multiprocessing to prevent segfaults on Python 3.13
import torch
torch.set_num_threads(1)

import json
import pickle
import time
from typing import List, Dict, Tuple
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
import faiss
import arxiv
from sentence_transformers import SentenceTransformer, CrossEncoder
from tqdm import tqdm


class ArxivRAGPipeline:
    """Complete RAG pipeline for arXiv papers"""

    def __init__(self,
                 data_dir: str = "./arxiv_data",
                 model_name: str = "all-MiniLM-L6-v2",
                 max_tokens: int = 512,
                 overlap: int = 50):
        """
        Initialize the RAG pipeline.

        Args:
            data_dir: Directory to store downloaded papers and generated files
            model_name: Sentence transformer model name
            max_tokens: Maximum tokens per chunk
            overlap: Overlap between chunks
        """
        self.data_dir = Path(data_dir)
        self.pdf_dir = self.data_dir / "pdfs"
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

        self.model_name = model_name
        self.max_tokens = max_tokens
        self.overlap = overlap

        # Initialize embedding model
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

        # Initialize cross-encoder for reranking (lazy loading)
        self.cross_encoder = None
        self.cross_encoder_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

        # Storage for processed data
        self.papers = []  # List of paper metadata
        self.chunks = []  # List of text chunks
        self.chunk_metadata = []  # Metadata for each chunk (paper_id, chunk_id, etc.)
        self.embeddings = None  # Numpy array of embeddings
        self.index = None  # FAISS index

    def download_papers(self, num_papers: int = 50, category: str = "cs.CL"):
        """
        Download papers from arXiv.

        Args:
            num_papers: Number of papers to download
            category: arXiv category (default: cs.CL - Computation and Language)
        """
        print(f"\n{'='*60}")
        print(f"Step 1: Downloading {num_papers} papers from arXiv ({category})")
        print(f"{'='*60}")

        # Search for papers
        search = arxiv.Search(
            query=f"cat:{category}",
            max_results=num_papers,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

        downloaded_count = 0
        for result in tqdm(search.results(), desc="Downloading papers", total=num_papers):
            try:
                # Create safe filename
                paper_id = result.entry_id.split('/')[-1]
                pdf_path = self.pdf_dir / f"{paper_id}.pdf"

                # Skip if already downloaded
                if pdf_path.exists():
                    print(f"   Already exists: {paper_id}")
                else:
                    # Download PDF
                    result.download_pdf(dirpath=str(self.pdf_dir), filename=f"{paper_id}.pdf")
                    print(f"   Downloaded: {paper_id}")
                    time.sleep(1)  # Be nice to arXiv servers

                # Store metadata
                self.papers.append({
                    "paper_id": paper_id,
                    "title": result.title,
                    "authors": [author.name for author in result.authors],
                    "summary": result.summary,
                    "published": str(result.published),
                    "pdf_path": str(pdf_path)
                })

                downloaded_count += 1
                if downloaded_count >= num_papers:
                    break

            except Exception as e:
                print(f"   Error downloading {result.entry_id}: {e}")
                continue

        print(f"\n Successfully downloaded {len(self.papers)} papers")

        # Save paper metadata
        metadata_path = self.data_dir / "papers_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.papers, f, indent=2)
        print(f" Saved metadata to {metadata_path}")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text as a single string
        """
        try:
            doc = fitz.open(pdf_path)
            pages = []

            for page in doc:
                page_text = page.get_text()
                # Basic cleaning: remove excessive whitespace
                page_text = " ".join(page_text.split())
                pages.append(page_text)

            full_text = "\n\n".join(pages)
            doc.close()
            return full_text
        except Exception as e:
            print(f"   Error extracting text from {pdf_path}: {e}")
            return ""

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks
        """
        # Simple word-based tokenization
        tokens = text.split()
        chunks = []
        step = self.max_tokens - self.overlap

        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i:i + self.max_tokens]
            if len(chunk_tokens) > 0:  # Skip empty chunks
                chunks.append(" ".join(chunk_tokens))

        return chunks

    def process_papers(self):
        """
        Extract text and create chunks from all downloaded papers.
        """
        print(f"\n{'='*60}")
        print(f"Step 2: Extracting text and creating chunks")
        print(f"{'='*60}")

        if not self.papers:
            # Try to load from saved metadata
            metadata_path = self.data_dir / "papers_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.papers = json.load(f)
            else:
                raise ValueError("No papers found. Run download_papers() first.")

        total_chunks = 0
        for paper in tqdm(self.papers, desc="Processing papers"):
            pdf_path = paper['pdf_path']

            # Extract text
            text = self.extract_text_from_pdf(pdf_path)
            if not text:
                continue

            # Create chunks
            paper_chunks = self.chunk_text(text)

            # Store chunks with metadata
            for chunk_id, chunk in enumerate(paper_chunks):
                self.chunks.append(chunk)
                # Extract year from published date (format: YYYY-MM-DD HH:MM:SS)
                year = int(paper['published'].split('-')[0]) if paper['published'] else None

                self.chunk_metadata.append({
                    "paper_id": paper['paper_id'],
                    "paper_title": paper['title'],
                    "chunk_id": chunk_id,
                    "total_chunks": len(paper_chunks),
                    "authors": paper['authors'],  # List of author names
                    "published_year": year,  # Year as integer
                    "published_date": paper['published']  # Full date string
                })

            total_chunks += len(paper_chunks)

        print(f"\n Created {total_chunks} chunks from {len(self.papers)} papers")
        print(f"  Average chunks per paper: {total_chunks / len(self.papers):.1f}")

        # Save chunks
        chunks_path = self.data_dir / "chunks.json"
        with open(chunks_path, 'w') as f:
            json.dump({
                "chunks": self.chunks,
                "metadata": self.chunk_metadata
            }, f, indent=2)
        print(f" Saved chunks to {chunks_path}")

    def generate_embeddings(self):
        """
        Generate embeddings for all chunks using sentence-transformers.
        """
        print(f"\n{'='*60}")
        print(f"Step 3: Generating embeddings")
        print(f"{'='*60}")

        if not self.chunks:
            raise ValueError("No chunks found. Run process_papers() first.")

        print(f"Encoding {len(self.chunks)} chunks...")
        # Use single process to avoid segfault with Python 3.13
        import torch
        self.embeddings = self.model.encode(
            self.chunks,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=16,  # Smaller batch size for stability
            device='cpu',    # Force CPU to avoid GPU issues
            normalize_embeddings=True  # Normalize for cosine similarity
        )

        print(f"\n Generated embeddings: shape {self.embeddings.shape}")

        # Save embeddings
        embeddings_path = self.data_dir / "embeddings.npy"
        np.save(embeddings_path, self.embeddings)
        print(f" Saved embeddings to {embeddings_path}")

    def build_faiss_index(self):
        """
        Build FAISS index from embeddings.
        """
        print(f"\n{'='*60}")
        print(f"Step 4: Building FAISS index")
        print(f"{'='*60}")

        if self.embeddings is None:
            # Try to load from saved file
            embeddings_path = self.data_dir / "embeddings.npy"
            if embeddings_path.exists():
                self.embeddings = np.load(embeddings_path)
            else:
                raise ValueError("No embeddings found. Run generate_embeddings() first.")

        # Build FAISS index
        dim = self.embeddings.shape[1]
        # Use Inner Product for cosine similarity (embeddings are normalized)
        self.index = faiss.IndexFlatIP(dim)  # Inner Product (cosine similarity)
        self.index.add(self.embeddings.astype('float32'))

        print(f" Built FAISS index with {self.index.ntotal} vectors (dim={dim})")

        # Save index
        index_path = self.data_dir / "faiss_index.bin"
        faiss.write_index(self.index, str(index_path))
        print(f" Saved FAISS index to {index_path}")

    def search(self, query: str, k: int = 3) -> List[Dict]:
        """
        Search for top-k most relevant chunks.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of result dictionaries with chunk text and metadata
        """
        if self.index is None:
            raise ValueError("No index found. Run build_faiss_index() first.")

        # Embed query (normalize for cosine similarity)
        query_embedding = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

        # Search (returns cosine similarities with IndexFlatIP)
        similarities, indices = self.index.search(query_embedding.astype('float32'), k)

        # Format results
        results = []
        for i, (idx, similarity) in enumerate(zip(indices[0], similarities[0])):
            results.append({
                "rank": i + 1,
                "chunk": self.chunks[idx],
                "metadata": self.chunk_metadata[idx],
                "similarity": float(similarity)  # Cosine similarity (higher = more similar)
            })

        return results

    def _load_cross_encoder(self):
        """
        Lazy load the cross-encoder model for reranking.
        """
        if self.cross_encoder is None:
            print(f"Loading cross-encoder model: {self.cross_encoder_name}")
            self.cross_encoder = CrossEncoder(self.cross_encoder_name)
        return self.cross_encoder

    def search_with_rerank(self, query: str, k: int = 3, initial_k: int = 20) -> List[Dict]:
        """
        Search for top-k most relevant chunks with reranking.

        This implements a two-stage retrieval:
        1. First stage: Use FAISS to get initial_k candidates (fast, approximate)
        2. Second stage: Use cross-encoder to rerank and select top k (slow, accurate)

        Args:
            query: Search query
            k: Number of final results to return
            initial_k: Number of candidates to retrieve before reranking (should be > k)

        Returns:
            List of result dictionaries with chunk text and metadata, reranked by cross-encoder scores
        """
        if self.index is None:
            raise ValueError("No index found. Run build_faiss_index() first.")

        # Stage 1: Fast retrieval with FAISS
        # Get more candidates than needed (e.g., 20) for reranking
        query_embedding = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        similarities, indices = self.index.search(query_embedding.astype('float32'), initial_k)

        # Collect candidates
        candidates = []
        for idx in indices[0]:
            candidates.append({
                "idx": idx,
                "chunk": self.chunks[idx],
                "metadata": self.chunk_metadata[idx]
            })

        # Stage 2: Rerank with cross-encoder
        # Load cross-encoder (lazy loading)
        cross_encoder = self._load_cross_encoder()

        # Prepare query-document pairs for cross-encoder
        query_doc_pairs = [[query, candidate["chunk"]] for candidate in candidates]

        # Get cross-encoder scores
        print(f"Reranking {len(candidates)} candidates with cross-encoder...")
        ce_scores = cross_encoder.predict(query_doc_pairs)

        # Add scores to candidates
        for candidate, score in zip(candidates, ce_scores):
            candidate["ce_score"] = float(score)

        # Sort by cross-encoder score (higher is better)
        candidates.sort(key=lambda x: x["ce_score"], reverse=True)

        # Format top-k results
        results = []
        for i, candidate in enumerate(candidates[:k]):
            results.append({
                "rank": i + 1,
                "chunk": candidate["chunk"],
                "metadata": candidate["metadata"],
                "similarity": candidate["ce_score"],  # Cross-encoder score
                "ce_score": candidate["ce_score"],
                "reranked": True
            })

        return results

    def search_with_metadata(
        self,
        query: str,
        k: int = 3,
        initial_k: int = 50,
        year_min: int = None,
        year_max: int = None,
        authors: List[str] = None,
        recency_boost: float = 0.0
    ) -> List[Dict]:
        """
        Search with metadata filtering and score boosting.

        Args:
            query: Search query
            k: Number of final results to return
            initial_k: Number of candidates to retrieve before filtering
            year_min: Minimum publication year (inclusive)
            year_max: Maximum publication year (inclusive)
            authors: List of author names to filter by (case-insensitive, partial match)
            recency_boost: Boost factor for recent papers (0.0-1.0)
                          0.0 = no boost, 0.5 = moderate, 1.0 = strong boost

        Returns:
            List of result dictionaries with metadata-filtered and boosted scores

        Example:
            # Find recent papers about transformers
            results = pipeline.search_with_metadata(
                "transformer models",
                k=5,
                year_min=2020,
                recency_boost=0.3
            )
        """
        if self.index is None:
            raise ValueError("No index found. Run build_faiss_index() first.")

        # Retrieve more candidates for filtering
        query_embedding = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        similarities, indices = self.index.search(query_embedding.astype('float32'), initial_k)

        # Collect and filter candidates
        candidates = []
        for idx, similarity in zip(indices[0], similarities[0]):
            metadata = self.chunk_metadata[idx]

            # Apply metadata filters
            if year_min and metadata.get('published_year'):
                if metadata['published_year'] < year_min:
                    continue

            if year_max and metadata.get('published_year'):
                if metadata['published_year'] > year_max:
                    continue

            if authors:
                # Check if any author matches (case-insensitive, partial match)
                author_list = metadata.get('authors', [])
                author_match = False
                for filter_author in authors:
                    for paper_author in author_list:
                        if filter_author.lower() in paper_author.lower():
                            author_match = True
                            break
                    if author_match:
                        break

                if not author_match:
                    continue

            # Base similarity score
            base_score = float(similarity)

            # Apply recency boost if specified
            if recency_boost > 0 and metadata.get('published_year'):
                # Normalize year to 0-1 range (assume papers from 2010-2024)
                year = metadata['published_year']
                year_normalized = (year - 2010) / (2024 - 2010)
                year_normalized = max(0, min(1, year_normalized))  # Clamp to 0-1

                # Boost score based on recency
                boosted_score = base_score + (recency_boost * year_normalized)
            else:
                boosted_score = base_score

            candidates.append({
                "idx": idx,
                "chunk": self.chunks[idx],
                "metadata": metadata,
                "base_similarity": base_score,
                "boosted_similarity": boosted_score
            })

        # Sort by boosted similarity
        candidates.sort(key=lambda x: x["boosted_similarity"], reverse=True)

        # Format top-k results
        results = []
        for i, candidate in enumerate(candidates[:k]):
            results.append({
                "rank": i + 1,
                "chunk": candidate["chunk"],
                "metadata": candidate["metadata"],
                "similarity": candidate["boosted_similarity"],
                "base_similarity": candidate["base_similarity"],
                "metadata_filtered": True
            })

        return results

    def generate_retrieval_report(self, output_path: str = None):
        """
        Generate a retrieval report with example queries.

        Args:
            output_path: Path to save the report (default: data_dir/retrieval_report.txt)
        """
        print(f"\n{'='*60}")
        print(f"Step 5: Generating retrieval report")
        print(f"{'='*60}")

        if output_path is None:
            output_path = self.data_dir / "retrieval_report.txt"

        # Example queries
        example_queries = [
            "What are the latest advances in transformer models?",
            "How does BERT improve language understanding?",
            "What is the role of attention mechanisms in NLP?",
            "Explain neural machine translation architectures",
            "What are the challenges in multilingual language models?"
        ]

        report_lines = []
        report_lines.append("="*80)
        report_lines.append("RETRIEVAL PERFORMANCE REPORT")
        report_lines.append("="*80)
        report_lines.append(f"\nModel: {self.model_name}")
        report_lines.append(f"Total chunks: {len(self.chunks)}")
        report_lines.append(f"Total papers: {len(self.papers)}")
        report_lines.append(f"Embedding dimension: {self.embeddings.shape[1]}")
        report_lines.append("\n" + "="*80)

        for i, query in enumerate(example_queries, 1):
            results = self.search(query, k=3)

            report_lines.append(f"\n\nQUERY {i}: {query}")
            report_lines.append("-" * 80)

            for result in results:
                report_lines.append(f"\nRank {result['rank']} (Similarity: {result['similarity']:.4f})")
                report_lines.append(f"Paper: {result['metadata']['paper_title']}")
                report_lines.append(f"Chunk {result['metadata']['chunk_id'] + 1}/{result['metadata']['total_chunks']}")
                report_lines.append(f"\nText: {result['chunk'][:300]}...")
                report_lines.append("-" * 80)

        # Write report
        report_text = "\n".join(report_lines)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(f" Saved retrieval report to {output_path}")

        # Also print to console
        print("\n" + report_text[:2000] + "\n...(see full report in file)")

    def save_for_api(self):
        """
        Save all necessary files for the FastAPI service.
        """
        print(f"\n{'='*60}")
        print(f"Preparing files for FastAPI service")
        print(f"{'='*60}")

        # Save chunks with metadata for easy loading
        api_data = {
            "chunks": self.chunks,
            "chunk_metadata": self.chunk_metadata,
            "model_name": self.model_name
        }

        api_data_path = self.data_dir / "api_data.pkl"
        with open(api_data_path, 'wb') as f:
            pickle.dump(api_data, f)

        print(f" Saved API data to {api_data_path}")
        print(f"\nFiles ready for FastAPI:")
        print(f"  1. FAISS index: {self.data_dir / 'faiss_index.bin'}")
        print(f"  2. API data: {api_data_path}")
        print(f"  3. Model: {self.model_name}")

    def run_full_pipeline(self, num_papers: int = 50):
        """
        Run the complete RAG pipeline.

        Args:
            num_papers: Number of papers to download
        """
        print(f"\n{'#'*80}")
        print(f"# ARXIV RAG PIPELINE - COMPLETE WORKFLOW")
        print(f"{'#'*80}\n")

        # Step 1: Download papers
        self.download_papers(num_papers=num_papers)

        # Step 2: Process papers (extract text and chunk)
        self.process_papers()

        # Step 3: Generate embeddings
        self.generate_embeddings()

        # Step 4: Build FAISS index
        self.build_faiss_index()

        # Step 5: Generate retrieval report
        self.generate_retrieval_report()

        # Step 6: Prepare API files
        self.save_for_api()

        print(f"\n{'#'*80}")
        print(f"# PIPELINE COMPLETE!")
        print(f"{'#'*80}")
        print(f"\n All deliverables generated:")
        print(f"  1. Data & Index: {self.data_dir}")
        print(f"  2. Retrieval Report: {self.data_dir / 'retrieval_report.txt'}")
        print(f"  3. FastAPI files ready in: {self.data_dir}")
        print(f"\nNext steps:")
        print(f"  Run the FastAPI service with: python main.py")


def main():
    """Main execution function"""
    # Initialize pipeline
    pipeline = ArxivRAGPipeline(
        data_dir="./arxiv_data",
        model_name="all-mpnet-base-v2",
        max_tokens=512,
        overlap=50
    )

    # Run complete pipeline
    pipeline.run_full_pipeline(num_papers=50)

    # Optional: Test search
    print("\n" + "="*80)
    print("Testing search functionality:")
    print("="*80)
    test_query = "What are transformer models?"
    results = pipeline.search(test_query, k=3)
    print(f"\nQuery: {test_query}")
    for result in results:
        print(f"\nRank {result['rank']}: {result['metadata']['paper_title']}")
        print(f"Similarity: {result['similarity']:.4f}")
        print(f"Text: {result['chunk'][:200]}...")


if __name__ == "__main__":
    main()