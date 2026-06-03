# MinerU Extraction Quality Notes: chang-2023

## Run
- Extractor: MinerU 2.7.6 (via do_parse Python API)
- Venv: envseis-paper-tracker/Prototype/rockfall/.venv (mineru 2.7.6, torch 2.10.0)
- Script: parse_mineru.py (backend=pipeline, method=auto, formula_enable=True, table_enable=True)
- Started: 2026-05-23T13:31:55Z
- Ended:   2026-05-23T13:34:05Z
- Elapsed: 130.9s
- Exit code: 0
- stdout: mineru.stdout.log
- stderr: mineru.stderr.log

## Raw Output Inventory
- Markdown files: 1 (auto/chang-2023.md, 63,294 chars)
- Image files:    20 (auto/images/*.jpg)
- JSON files:     2 (chang-2023_content_list.json, chang-2023_middle.json)
- CSV files:      0
- HTML files:     0

## Extraction Quality
- Text output: yes
- Figures extracted: yes — 20 images (hash-named .jpg)
- Tables extracted: no CSV/HTML produced — table content likely inlined in Markdown
- OCR applied: yes (OCR-det + OCR-rec ran on all 13 pages, 1640 text regions)
- Equation preservation: partial — LaTeX math rendered (e.g. $M=38,989F_L^{-2.03}$), but some formulas garbled
- Reading order: mostly correct; multi-column intro text merged correctly

## Spot Checks
- Title found: yes — "Field experiments: How well can seismic monitoring assess rock mass falling?"
- Abstract found: yes — present but with OCR noise (some character substitutions)
- Section headings: 27 headings detected (all major sections present including 2.1, 2.2, 3.1, 3.2, 3.3, 4.1–4.6, 5)
- Figure captions: readable — Fig. 1 caption fully present; image refs use hash filenames
- Table structure: Table 1 (PGV data) appears to be inline text, no CSV output
- References: present at end of document

## Known OCR Issues
- Abstract has garbled text in several sentences (character substitutions from OCR)
  - e.g. "Rocalrackn nmataion are ciil or eeciveazar reon" (should be "Rockfall tracking and mass estimation are critical for effective hazard response")
- Some section heading numbering garbled: "# . Introduction" (should be "1. Introduction"), "# 4.4Station" (missing space)
- Smoke query q1 NOT found verbatim — abstract OCR noise breaks exact-match
- Equations mostly preserved as LaTeX inline ($...$) but some garbled
