"""
Batch OCR for arXiv PDFs
Converts PDFs to text using Tesseract OCR, preserving layout.
"""

import os
import json
from pdf2image import convert_from_path
import pytesseract
from typing import Optional, Dict
import time


class PDFBatchOCR:
    """Batch OCR processor for arXiv PDFs."""

    def __init__(self, pdf_dir: str = "arxiv_pdf", output_dir: str = "pdf_ocr"):
        """
        Initialize the OCR processor.

        Args:
            pdf_dir: Directory containing PDF files
            output_dir: Directory to save OCR text files
        """
        self.pdf_dir = pdf_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def ocr_pdf_to_text(self, pdf_path: str, preserve_layout: bool = True) -> Optional[str]:
        """
        Extract text from PDF using Tesseract OCR.

        Args:
            pdf_path: Path to PDF file
            preserve_layout: Whether to preserve layout structure

        Returns:
            Extracted text or None if extraction fails
        """
        try:
            print(f"Processing: {os.path.basename(pdf_path)}")

            # Convert PDF to images
            images = convert_from_path(pdf_path, dpi=300)

            # Extract text from each page
            all_text = []

            for page_num, image in enumerate(images, 1):
                print(f"  OCR page {page_num}/{len(images)}")

                # Use different PSM modes for layout preservation
                if preserve_layout:
                    # PSM 6: Assume uniform block of text
                    custom_config = r'--oem 3 --psm 6'
                else:
                    # PSM 3: Fully automatic page segmentation
                    custom_config = r'--oem 3 --psm 3'

                text = pytesseract.image_to_string(image, config=custom_config)

                # Add page separator
                page_text = f"\n{'='*80}\n"
                page_text += f"PAGE {page_num}\n"
                page_text += f"{'='*80}\n\n"
                page_text += text

                all_text.append(page_text)

            full_text = "\n".join(all_text)
            print(f"  Completed: {len(images)} pages processed")

            return full_text

        except Exception as e:
            print(f"  Error processing {pdf_path}: {e}")
            return None

    def process_single_pdf(self, pdf_filename: str, preserve_layout: bool = True) -> Optional[str]:
        """
        Process a single PDF file.

        Args:
            pdf_filename: Name of PDF file
            preserve_layout: Whether to preserve layout

        Returns:
            Path to output text file or None if failed
        """
        pdf_path = os.path.join(self.pdf_dir, pdf_filename)

        if not os.path.exists(pdf_path):
            print(f"PDF not found: {pdf_path}")
            return None

        # Extract text
        text = self.ocr_pdf_to_text(pdf_path, preserve_layout=preserve_layout)

        if text:
            # Save to text file
            base_name = os.path.splitext(pdf_filename)[0]
            output_path = os.path.join(self.output_dir, f"{base_name}.txt")

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)

            print(f"  Saved to: {output_path}")
            return output_path

        return None

    def process_all_pdfs(self, preserve_layout: bool = True, delay: float = 0.5) -> Dict[str, str]:
        """
        Process all PDFs in the input directory.

        Args:
            preserve_layout: Whether to preserve layout
            delay: Delay between processing files (seconds)

        Returns:
            Dictionary mapping PDF filenames to output text file paths
        """
        if not os.path.exists(self.pdf_dir):
            print(f"PDF directory not found: {self.pdf_dir}")
            return {}

        pdf_files = [f for f in os.listdir(self.pdf_dir) if f.endswith('.pdf')]

        if not pdf_files:
            print(f"No PDF files found in {self.pdf_dir}")
            return {}

        print(f"Found {len(pdf_files)} PDF files to process\n")

        results = {}

        for idx, pdf_file in enumerate(pdf_files, 1):
            print(f"[{idx}/{len(pdf_files)}] Processing {pdf_file}")

            output_path = self.process_single_pdf(pdf_file, preserve_layout=preserve_layout)

            if output_path:
                results[pdf_file] = output_path

            # Add delay between files
            if idx < len(pdf_files):
                time.sleep(delay)

            print()

        return results

    def generate_summary(self, results: Dict[str, str]) -> str:
        """
        Generate a summary of OCR processing results.

        Args:
            results: Dictionary of processing results

        Returns:
            Path to summary JSON file
        """
        summary = {
            'total_processed': len(results),
            'output_directory': self.output_dir,
            'files': []
        }

        for pdf_file, txt_file in results.items():
            # Get file sizes
            pdf_path = os.path.join(self.pdf_dir, pdf_file)
            pdf_size = os.path.getsize(pdf_path)
            txt_size = os.path.getsize(txt_file)

            # Count lines in text file
            with open(txt_file, 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f)

            summary['files'].append({
                'pdf_file': pdf_file,
                'txt_file': os.path.basename(txt_file),
                'pdf_size_kb': round(pdf_size / 1024, 2),
                'txt_size_kb': round(txt_size / 1024, 2),
                'line_count': line_count
            })

        # Save summary
        summary_path = os.path.join(self.output_dir, 'ocr_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"Summary saved to: {summary_path}")
        return summary_path


def main():
    """Main execution function."""

    # Initialize OCR processor
    ocr = PDFBatchOCR(pdf_dir="arxiv_pdf", output_dir="pdf_ocr")

    # Process all PDFs with layout preservation
    print("="*80)
    print("Starting Batch OCR Processing")
    print("="*80)
    print()

    results = ocr.process_all_pdfs(preserve_layout=True, delay=0.5)

    # Generate summary
    print("="*80)
    print("Generating Summary")
    print("="*80)
    print()

    summary_path = ocr.generate_summary(results)

    # Print final statistics
    print()
    print("="*80)
    print("Processing Complete")
    print("="*80)
    print(f"Total files processed: {len(results)}")
    print(f"Output directory: {ocr.output_dir}")
    print(f"Summary file: {summary_path}")


if __name__ == "__main__":
    main()
