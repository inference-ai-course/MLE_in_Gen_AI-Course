"""
End-to-End Data Cleaner
Merges outputs from Tasks 1-3, performs language detection, HTML stripping,
deduplication using MinHash, PII removal, and repetitive n-gram removal.
"""

import os
import json
import re
from typing import List, Dict, Tuple
from collections import Counter
import glob

# Third-party imports
from langdetect import detect, LangDetectException
from datasketch import MinHash, MinHashLSH
from bs4 import BeautifulSoup
import html


class DataCleaner:
    """End-to-end data cleaning and deduplication."""

    def __init__(self, similarity_threshold: float = 0.7):
        """
        Initialize the data cleaner.

        Args:
            similarity_threshold: MinHash similarity threshold for deduplication (0-1)
        """
        self.similarity_threshold = similarity_threshold
        self.stats = {
            'total_documents': 0,
            'after_language_filter': 0,
            'after_html_cleaning': 0,
            'after_deduplication': 0,
            'after_pii_removal': 0,
            'after_ngram_removal': 0,
            'total_tokens_original': 0,
            'total_tokens_final': 0,
            'removal_percentage': 0.0
        }

    def load_arxiv_data(self, json_file: str = "arxiv_clean.json") -> List[Dict[str, str]]:
        """Load data from arXiv JSON file."""
        documents = []

        try:
            # Try reading in chunks since file is large
            with open(json_file, 'r', encoding='utf-8') as f:
                content = f.read()
                data = json.loads(content)

            for item in data:
                doc = {
                    'source': 'arxiv',
                    'text': item.get('abstract', ''),
                    'metadata': {
                        'title': item.get('title', ''),
                        'url': item.get('url', ''),
                        'authors': item.get('authors', []),
                        'date': item.get('date', '')
                    }
                }
                if doc['text']:
                    documents.append(doc)

            print(f"Loaded {len(documents)} documents from arXiv")

        except Exception as e:
            print(f"Error loading arXiv data: {e}")

        return documents

    def load_ocr_data(self, ocr_dir: str = "pdf_ocr") -> List[Dict[str, str]]:
        """Load data from OCR text files."""
        documents = []

        try:
            txt_files = glob.glob(os.path.join(ocr_dir, "*.txt"))

            for txt_file in txt_files:
                with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()

                doc = {
                    'source': 'ocr',
                    'text': text,
                    'metadata': {
                        'filename': os.path.basename(txt_file)
                    }
                }
                documents.append(doc)

            print(f"Loaded {len(documents)} documents from OCR")

        except Exception as e:
            print(f"Error loading OCR data: {e}")

        return documents

    def load_transcripts(self, jsonl_file: str = "talks_transcripts.jsonl") -> List[Dict[str, str]]:
        """Load data from transcripts JSONL file."""
        documents = []

        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)

                        doc = {
                            'source': 'transcript',
                            'text': item.get('full_text', ''),
                            'metadata': {
                                'title': item.get('title', ''),
                                'url': item.get('url', ''),
                                'duration': item.get('duration', 0)
                            }
                        }
                        if doc['text']:
                            documents.append(doc)

            print(f"Loaded {len(documents)} documents from transcripts")

        except Exception as e:
            print(f"Error loading transcript data: {e}")

        return documents

    def detect_language(self, documents: List[Dict[str, str]], target_lang: str = 'en') -> List[Dict[str, str]]:
        """
        Filter documents by language.

        Args:
            documents: List of document dictionaries
            target_lang: Target language code (default: 'en' for English)

        Returns:
            Filtered list of documents
        """
        filtered = []

        for doc in documents:
            text = doc['text']

            # Skip very short texts
            if len(text) < 20:
                continue

            try:
                lang = detect(text)
                if lang == target_lang:
                    filtered.append(doc)
            except LangDetectException:
                # If detection fails, skip the document
                continue

        print(f"Language filter: {len(documents)} → {len(filtered)} documents (kept {target_lang} only)")
        return filtered

    def strip_html(self, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Remove HTML tags and decode HTML entities.

        Args:
            documents: List of document dictionaries

        Returns:
            Documents with cleaned text
        """
        cleaned = []

        for doc in documents:
            text = doc['text']

            # Decode HTML entities
            text = html.unescape(text)

            # Remove HTML tags using BeautifulSoup
            soup = BeautifulSoup(text, 'html.parser')
            text = soup.get_text()

            # Remove OCR page separators (e.g., "===... PAGE 1 ===...")
            text = re.sub(r'={20,}\s*PAGE\s+\d+\s*={20,}', '', text)

            # Remove arXiv metadata headers (e.g., "Computer Science > ... Title:")
            text = re.sub(r'Computer Science\s*>\s*[^\[]+\[Submitted on[^\]]+\]\s*Title:', '', text)

            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()

            doc['text'] = text
            cleaned.append(doc)

        print(f"HTML cleaning: Processed {len(cleaned)} documents")
        return cleaned

    def create_minhash(self, text: str, num_perm: int = 128) -> MinHash:
        """
        Create MinHash signature for a text.

        Args:
            text: Input text
            num_perm: Number of permutations for MinHash

        Returns:
            MinHash object
        """
        m = MinHash(num_perm=num_perm)

        # Tokenize by words and create shingles (3-grams)
        words = text.lower().split()
        shingles = [' '.join(words[i:i+3]) for i in range(len(words)-2)]

        for shingle in shingles:
            m.update(shingle.encode('utf-8'))

        return m

    def deduplicate_minhash(self, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Deduplicate documents using MinHashLSH.

        Args:
            documents: List of document dictionaries

        Returns:
            Deduplicated list of documents
        """
        # Create LSH index
        lsh = MinHashLSH(threshold=self.similarity_threshold, num_perm=128)

        unique_docs = []
        seen_ids = set()

        for idx, doc in enumerate(documents):
            text = doc['text']

            # Create MinHash signature
            minhash = self.create_minhash(text)

            # Query for similar documents
            result = lsh.query(minhash)

            if len(result) == 0:
                # No similar document found, this is unique
                doc_id = f"doc_{idx}"
                lsh.insert(doc_id, minhash)
                unique_docs.append(doc)
                seen_ids.add(doc_id)

        print(f"Deduplication: {len(documents)} → {len(unique_docs)} documents "
              f"(removed {len(documents) - len(unique_docs)} duplicates)")

        return unique_docs

    def remove_pii(self, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Remove personally identifiable information (PII).

        Args:
            documents: List of document dictionaries

        Returns:
            Documents with PII removed
        """
        # Regex patterns for PII
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b'
        credit_card_pattern = r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
        ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'

        cleaned = []
        total_removals = 0

        for doc in documents:
            text = doc['text']
            original_text = text

            # Remove emails
            text = re.sub(email_pattern, '[EMAIL]', text)

            # Remove phone numbers
            text = re.sub(phone_pattern, '[PHONE]', text)

            # Remove credit card numbers
            text = re.sub(credit_card_pattern, '[CC]', text)

            # Remove SSNs
            text = re.sub(ssn_pattern, '[SSN]', text)

            if text != original_text:
                total_removals += 1

            doc['text'] = text
            cleaned.append(doc)

        print(f"PII removal: Found and masked PII in {total_removals} documents")
        return cleaned

    def remove_repetitive_ngrams(self, documents: List[Dict[str, str]],
                                 n: int = 3, threshold: int = 5) -> List[Dict[str, str]]:
        """
        Remove repetitive n-grams that appear too frequently.

        Args:
            documents: List of document dictionaries
            n: N-gram size (default: 3)
            threshold: Maximum allowed repetitions (default: 5)

        Returns:
            Documents with repetitive n-grams removed
        """
        cleaned = []

        for doc in documents:
            text = doc['text']
            words = text.split()

            # Create n-grams
            ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]

            # Count n-gram occurrences
            ngram_counts = Counter(ngrams)

            # Find repetitive n-grams
            repetitive = {ng for ng, count in ngram_counts.items() if count > threshold}

            # Remove repetitive n-grams
            if repetitive:
                new_words = []
                i = 0
                while i < len(words):
                    # Check if current position starts a repetitive n-gram
                    if i <= len(words) - n:
                        current_ngram = tuple(words[i:i+n])
                        if current_ngram in repetitive:
                            # Skip this n-gram
                            i += n
                            continue

                    new_words.append(words[i])
                    i += 1

                text = ' '.join(new_words)

            doc['text'] = text
            cleaned.append(doc)

        print(f"N-gram removal: Processed {len(cleaned)} documents")
        return cleaned

    def count_tokens(self, documents: List[Dict[str, str]]) -> int:
        """Count total tokens in documents."""
        total = 0
        for doc in documents:
            total += len(doc['text'].split())
        return total

    def process_all(self) -> Tuple[List[Dict[str, str]], Dict]:
        """
        Run the complete cleaning pipeline.

        Returns:
            Tuple of (cleaned documents, statistics)
        """
        print("="*80)
        print("Starting Data Cleaning Pipeline")
        print("="*80)
        print()

        # Step 1: Load data from all sources
        print("Step 1: Loading data...")
        arxiv_docs = self.load_arxiv_data()
        ocr_docs = self.load_ocr_data()
        transcript_docs = self.load_transcripts()

        all_docs = arxiv_docs + ocr_docs + transcript_docs
        self.stats['total_documents'] = len(all_docs)
        self.stats['total_tokens_original'] = self.count_tokens(all_docs)
        print(f"Total documents loaded: {len(all_docs)}")
        print()

        # Step 2: Language detection
        print("Step 2: Language detection...")
        all_docs = self.detect_language(all_docs, target_lang='en')
        self.stats['after_language_filter'] = len(all_docs)
        print()

        # Step 3: Strip HTML
        print("Step 3: Stripping HTML...")
        all_docs = self.strip_html(all_docs)
        self.stats['after_html_cleaning'] = len(all_docs)
        print()

        # Step 4: Deduplication
        print("Step 4: Deduplication with MinHash...")
        all_docs = self.deduplicate_minhash(all_docs)
        self.stats['after_deduplication'] = len(all_docs)
        print()

        # Step 5: Remove PII
        print("Step 5: Removing PII...")
        all_docs = self.remove_pii(all_docs)
        self.stats['after_pii_removal'] = len(all_docs)
        print()

        # Step 6: Remove repetitive n-grams
        print("Step 6: Removing repetitive n-grams...")
        all_docs = self.remove_repetitive_ngrams(all_docs, n=3, threshold=5)
        self.stats['after_ngram_removal'] = len(all_docs)
        print()

        # Calculate final stats
        self.stats['total_tokens_final'] = self.count_tokens(all_docs)
        if self.stats['total_tokens_original'] > 0:
            removed = self.stats['total_tokens_original'] - self.stats['total_tokens_final']
            self.stats['removal_percentage'] = (removed / self.stats['total_tokens_original']) * 100

        print("="*80)
        print("Cleaning Pipeline Complete")
        print("="*80)

        return all_docs, self.stats

    def save_corpus(self, documents: List[Dict[str, str]], output_file: str = "clean_corpus.txt"):
        """
        Save cleaned corpus to text file.

        Args:
            documents: List of cleaned documents
            output_file: Output filename
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, doc in enumerate(documents):
                f.write(doc['text'])
                if idx < len(documents) - 1:
                    f.write("\n\n")

        print(f"Corpus saved to: {output_file}")

    def save_stats(self, stats: Dict, output_file: str = "stats.md"):
        """
        Save statistics to markdown file.

        Args:
            stats: Statistics dictionary
            output_file: Output filename
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Data Cleaning Statistics\n\n")

            f.write("## Document Counts\n\n")
            f.write(f"- **Total documents loaded**: {stats['total_documents']}\n")
            f.write(f"- **After language filter**: {stats['after_language_filter']}\n")
            f.write(f"- **After HTML cleaning**: {stats['after_html_cleaning']}\n")
            f.write(f"- **After deduplication**: {stats['after_deduplication']}\n")
            f.write(f"- **After PII removal**: {stats['after_pii_removal']}\n")
            f.write(f"- **Final document count**: {stats['after_ngram_removal']}\n\n")

            f.write("## Token Counts\n\n")
            f.write(f"- **Original token count**: {stats['total_tokens_original']:,}\n")
            f.write(f"- **Final token count**: {stats['total_tokens_final']:,}\n")
            f.write(f"- **Tokens removed**: {stats['total_tokens_original'] - stats['total_tokens_final']:,}\n")
            f.write(f"- **Removal percentage**: {stats['removal_percentage']:.2f}%\n\n")

            f.write("## Pipeline Steps\n\n")
            f.write("1. **Load data** from arXiv, OCR, and transcripts\n")
            f.write("2. **Language detection** - Keep English documents only\n")
            f.write("3. **HTML stripping** - Remove HTML tags and decode entities\n")
            f.write(f"4. **Deduplication** - MinHash LSH (threshold: {self.similarity_threshold})\n")
            f.write("5. **PII removal** - Mask emails, phone numbers, credit cards, SSNs\n")
            f.write("6. **N-gram removal** - Remove repetitive 3-grams\n")

        print(f"Statistics saved to: {output_file}")


def main():
    """Main execution function."""

    # Initialize cleaner with similarity threshold of 0.7
    cleaner = DataCleaner(similarity_threshold=0.7)

    # Run the complete pipeline
    cleaned_docs, stats = cleaner.process_all()

    # Save results
    print()
    print("Saving results...")
    cleaner.save_corpus(cleaned_docs, output_file="clean_corpus.txt")
    cleaner.save_stats(stats, output_file="stats.md")

    print()
    print("="*80)
    print("All Done!")
    print("="*80)
    print(f"Final corpus: {len(cleaned_docs)} documents, {stats['total_tokens_final']:,} tokens")
    print(f"Removed {stats['removal_percentage']:.2f}% of original content")


if __name__ == "__main__":
    main()
