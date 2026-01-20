# Data Cleaning Statistics

## Document Counts

- **Total documents loaded**: 410
- **After language filter**: 410
- **After HTML cleaning**: 410
- **After deduplication**: 404
- **After PII removal**: 404
- **Final document count**: 404

## Token Counts

- **Original token count**: 2,345,230
- **Final token count**: 2,017,219
- **Tokens removed**: 328,011
- **Removal percentage**: 13.99%

## Pipeline Steps

1. **Load data** from arXiv, OCR, and transcripts
2. **Language detection** - Keep English documents only
3. **HTML stripping** - Remove HTML tags and decode entities
4. **Deduplication** - MinHash LSH (threshold: 0.7)
5. **PII removal** - Mask emails, phone numbers, credit cards, SSNs
6. **N-gram removal** - Remove repetitive 3-grams
