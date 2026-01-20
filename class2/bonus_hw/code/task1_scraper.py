"""
arXiv Paper Abstract Scraper
Fetches latest papers from arXiv subcategories, scrapes abstracts, and saves as JSON.
"""

import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import json
import time
import os
from datetime import datetime
from typing import List, Dict, Optional
import trafilatura
import pytesseract
from PIL import Image
from pdf2image import convert_from_path


class ArxivScraper:
    """Scraper for arXiv papers with abstract extraction."""

    def __init__(self, category: str = "cs.CL"):
        """
        Initialize the scraper.

        Args:
            category: arXiv category (e.g., 'cs.CL', 'cs.AI', 'cs.LG')
        """
        self.category = category
        self.base_api_url = "http://export.arxiv.org/api/query"
        self.base_abs_url = "https://arxiv.org/abs/"
        self.base_pdf_url = "https://arxiv.org/pdf/"
        self.pdf_dir = "arxiv_pdf"
        os.makedirs(self.pdf_dir, exist_ok=True)

    def fetch_papers(self, max_results: int = 200) -> List[Dict[str, str]]:
        """
        Fetch latest papers from arXiv API.

        Args:
            max_results: Maximum number of papers to fetch (default: 200)

        Returns:
            List of dictionaries containing paper metadata
        """
        print(f"Fetching {max_results} papers from category: {self.category}")

        # Build query parameters
        params = {
            'search_query': f'cat:{self.category}',
            'start': 0,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }

        query_url = f"{self.base_api_url}?{urllib.parse.urlencode(params)}"

        try:
            with urllib.request.urlopen(query_url) as response:
                xml_data = response.read().decode('utf-8')
        except Exception as e:
            print(f"Error fetching data from arXiv API: {e}")
            return []

        # Parse XML response
        papers = self._parse_arxiv_xml(xml_data)
        print(f"Successfully fetched {len(papers)} papers")

        return papers

    def _parse_arxiv_xml(self, xml_data: str) -> List[Dict[str, str]]:
        """Parse arXiv API XML response."""
        papers = []

        try:
            root = ET.fromstring(xml_data)

            # Define namespace
            ns = {'atom': 'http://www.w3.org/2005/Atom',
                  'arxiv': 'http://arxiv.org/schemas/atom'}

            for entry in root.findall('atom:entry', ns):
                # Extract arxiv ID from the id field
                id_elem = entry.find('atom:id', ns)
                if id_elem is not None:
                    arxiv_id = id_elem.text.split('/abs/')[-1]
                else:
                    continue

                # Extract title
                title_elem = entry.find('atom:title', ns)
                title = title_elem.text.strip().replace('\n', ' ') if title_elem is not None else ""

                # Extract authors
                authors = []
                for author in entry.findall('atom:author', ns):
                    name_elem = author.find('atom:name', ns)
                    if name_elem is not None:
                        authors.append(name_elem.text)

                # Extract published date
                published_elem = entry.find('atom:published', ns)
                date = published_elem.text[:10] if published_elem is not None else ""

                # Extract summary (abstract from API)
                summary_elem = entry.find('atom:summary', ns)
                api_abstract = summary_elem.text.strip().replace('\n', ' ') if summary_elem is not None else ""

                paper = {
                    'url': f"{self.base_abs_url}{arxiv_id}",
                    'arxiv_id': arxiv_id,
                    'title': title,
                    'authors': authors,
                    'date': date,
                    'api_abstract': api_abstract,
                    'scraped_abstract': None,
                    'ocr_abstract': None
                }

                papers.append(paper)

        except Exception as e:
            print(f"Error parsing XML: {e}")

        return papers

    def scrape_abstract_with_trafilatura(self, url: str) -> Optional[str]:
        """
        Scrape abstract from arXiv /abs/ page using Trafilatura.

        Args:
            url: URL of the arXiv abstract page

        Returns:
            Cleaned abstract text or None if extraction fails
        """
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = trafilatura.extract(downloaded, include_comments=False,
                                          include_tables=False)
                return text
            return None
        except Exception as e:
            print(f"Error scraping {url} with Trafilatura: {e}")
            return None

    def download_pdf(self, arxiv_id: str) -> Optional[str]:
        """
        Download PDF for a given arXiv paper.

        Args:
            arxiv_id: arXiv paper ID

        Returns:
            Path to downloaded PDF or None if download fails
        """
        pdf_url = f"{self.base_pdf_url}{arxiv_id}.pdf"
        pdf_path = os.path.join(self.pdf_dir, f"{arxiv_id}.pdf")

        # Skip if already downloaded
        if os.path.exists(pdf_path):
            return pdf_path

        try:
            urllib.request.urlretrieve(pdf_url, pdf_path)
            return pdf_path
        except Exception as e:
            print(f"Error downloading PDF for {arxiv_id}: {e}")
            return None

    def extract_text_from_pdf_with_ocr(self, pdf_path: str) -> Optional[str]:
        """
        Extract text from PDF using Tesseract OCR.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            OCR-extracted text or None if extraction fails
        """
        try:
            # Convert PDF to images (only first 3 pages to get abstract)
            images = convert_from_path(pdf_path, first_page=1, last_page=3)

            # Extract text from each page
            full_text = []
            for i, image in enumerate(images):
                text = pytesseract.image_to_string(image)
                full_text.append(text)

            ocr_text = "\n".join(full_text)

            # Try to extract just the abstract section
            if "Abstract" in ocr_text:
                abstract_start = ocr_text.find("Abstract")
                # Try to find where abstract ends (usually at "Introduction" or similar)
                end_markers = ["Introduction", "1 Introduction", "1. Introduction"]
                abstract_end = len(ocr_text)
                for marker in end_markers:
                    pos = ocr_text.find(marker, abstract_start)
                    if pos != -1:
                        abstract_end = pos
                        break
                ocr_text = ocr_text[abstract_start:abstract_end]

            return ocr_text.strip()

        except Exception as e:
            print(f"Error extracting OCR from PDF {pdf_path}: {e}")
            return None

    def scrape_all_papers(self, papers: List[Dict[str, str]],
                         use_trafilatura: bool = True,
                         use_ocr: bool = True,
                         delay: float = 1.0) -> List[Dict[str, str]]:
        """
        Scrape abstracts for all papers.

        Args:
            papers: List of paper metadata from API
            use_trafilatura: Whether to use Trafilatura scraping
            use_ocr: Whether to use OCR extraction from PDFs
            delay: Delay between requests in seconds

        Returns:
            List of papers with scraped abstracts
        """
        total = len(papers)

        for idx, paper in enumerate(papers, 1):
            print(f"Processing paper {idx}/{total}: {paper['arxiv_id']}")

            # Scrape with Trafilatura
            if use_trafilatura:
                scraped = self.scrape_abstract_with_trafilatura(paper['url'])
                paper['scraped_abstract'] = scraped
                time.sleep(delay)

            # Extract with OCR from PDF
            if use_ocr:
                # Download PDF
                pdf_path = self.download_pdf(paper['arxiv_id'])
                if pdf_path:
                    ocr_text = self.extract_text_from_pdf_with_ocr(pdf_path)
                    paper['ocr_abstract'] = ocr_text
                time.sleep(delay)

        return papers

    def save_to_json(self, papers: List[Dict[str, str]], filename: str = None):
        """
        Save papers to JSON file.

        Args:
            papers: List of paper data
            filename: Output filename (default: arxiv_{category}_{timestamp}.json)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"arxiv_{self.category.replace('.', '_')}_{timestamp}.json"

        # Prepare data with final abstract selection
        output_data = []
        for paper in papers:
            # Choose best available abstract
            abstract = (paper.get('scraped_abstract') or
                       paper.get('api_abstract') or
                       paper.get('ocr_abstract') or
                       "")

            output_data.append({
                'url': paper['url'],
                'title': paper['title'],
                'abstract': abstract,
                'authors': paper['authors'],
                'date': paper['date']
            })

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(output_data)} papers to {filename}")
        return filename


def main():
    """Main execution function."""
    # Example usage: scrape cs.CL (Computation and Language) papers
    category = "cs.CL"
    max_papers = 200

    scraper = ArxivScraper(category=category)

    # Fetch papers from API
    papers = scraper.fetch_papers(max_results=max_papers)

    if not papers:
        print("No papers fetched. Exiting.")
        return

    # Scrape abstracts (you can disable OCR if not needed, as it's slower)
    papers = scraper.scrape_all_papers(
        papers,
        use_trafilatura=True,
        use_ocr=True,  # Set to True to enable OCR (slower)
        delay=1.0
    )

    # Save to JSON
    output_file = scraper.save_to_json(papers)
    print(f"Complete! Data saved to: {output_file}")


if __name__ == "__main__":
    main()
