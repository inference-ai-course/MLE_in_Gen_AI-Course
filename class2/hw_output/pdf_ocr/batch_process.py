"""batch_process: full pipeline from PDF to sidecar, for one or many papers.

Steps per paper (automatically sequenced):
  1. MinerU extraction  — only if outputs/mineru_raw/{doc_id}/auto/ is missing
  2. Sidecar generation — manifest + VLM repair + tables + Gemini alt-text

The two steps use different Python environments, handled transparently via
subprocess: MinerU uses .venv_mineru/bin/python, sidecar uses sys.executable.

Usage:
    conda run -n seismo python3 batch_process.py                       # auto-discover all PDFs in test_paper/
    conda run -n seismo python3 batch_process.py chang-2023 deparis-2008
    conda run -n seismo python3 batch_process.py --pull-model chang-2023 deparis-2008
    conda run -n seismo python3 batch_process.py --skip-vlm deparis-2008
    conda run -n seismo python3 batch_process.py --skip-vlm --skip-alt-text deparis-2008
    conda run -n seismo python3 batch_process.py --pdf-dir papers/     # auto-discover in papers/
    conda run -n seismo python3 batch_process.py --pdf-dir papers/ doc1 doc2

Note: auto-discovery scans only the top-level of --pdf-dir (not recursive).

Output:
    outputs/batch_runs/batch-{timestamp}.jsonl  — one record per paper + summary
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent
MINERU_PYTHON = ROOT / ".venv_mineru/bin/python"


def _mineru_ready(doc_id: str) -> bool:
    """True if MinerU has already extracted this paper (both required output files present)."""
    base = ROOT / f"outputs/mineru_raw/{doc_id}/auto"
    return (
        (base / f"{doc_id}_content_list.json").exists()
        and (base / f"{doc_id}.md").exists()
    )


def _clean_mineru_output(doc_id: str) -> None:
    """Remove stale MinerU output dir before a re-run to prevent merged/dirty results."""
    import shutil
    auto_dir = ROOT / f"outputs/mineru_raw/{doc_id}/auto"
    if auto_dir.exists():
        shutil.rmtree(auto_dir)
        print(f"[{doc_id}] Cleaned stale MinerU output dir before re-extraction")


def _find_pdf(doc_id: str, pdf_dir: Path) -> Path | None:
    for ext in (".pdf", ".PDF"):
        p = pdf_dir / f"{doc_id}{ext}"
        if p.exists():
            return p
    return None


def _read_manifest_stats(doc_id: str) -> dict:
    manifest_path = ROOT / f"papers_corpus/{doc_id}/manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text()).get("stats", {})


def _count_figure_errors(doc_id: str) -> int:
    alt_path = ROOT / f"papers_corpus/{doc_id}/figure_alt_text.md"
    if not alt_path.exists():
        return 0
    return alt_path.read_text().count("[Gemini generation failed:")


def _run_mineru(
    doc_id: str, pdf_path: Path, verbose: bool
) -> tuple[subprocess.CompletedProcess, str]:
    """Run MinerU with MPS-first, CPU-fallback strategy.

    Returns (result, device_used). On SIGSEGV (exit -11), cleans the stale
    output dir and retries with MINERU_DEVICE_MODE=cpu injected into the env.
    """
    cmd = [str(MINERU_PYTHON), "parse_mineru.py", doc_id, str(pdf_path)]

    # Always force MPS explicitly so the first attempt is guaranteed MPS,
    # regardless of whatever MINERU_DEVICE_MODE the caller's shell may have set.
    mps_env = {**os.environ, "MINERU_DEVICE_MODE": "mps"}
    result = subprocess.run(cmd, cwd=ROOT, capture_output=not verbose, text=True, env=mps_env)
    if not verbose and result.stdout:
        print(result.stdout, end="")

    if result.returncode == -11:  # SIGSEGV — MPS GPU crash on large PDF
        print(f"[{doc_id}] MinerU: SIGSEGV (MPS crash), retrying with CPU...")
        _clean_mineru_output(doc_id)
        cpu_env = {**os.environ, "MINERU_DEVICE_MODE": "cpu"}
        result = subprocess.run(
            cmd, cwd=ROOT, capture_output=not verbose, text=True, env=cpu_env
        )
        if not verbose and result.stdout:
            print(result.stdout, end="")
        return result, "cpu"

    return result, "mps"


def run_batch(
    doc_ids: list[str],
    sidecar_extra_args: list[str],
    pdf_dir: Path,
    verbose: bool = True,
) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = ROOT / "outputs/batch_runs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"batch-{timestamp}.jsonl"

    print(f"=== Batch run: {len(doc_ids)} papers ===")
    print(f"Log: {log_path.relative_to(ROOT)}\n")

    records: list[dict] = []
    total_errors = 0

    with log_path.open("w") as log_fh:
        for doc_id in doc_ids:
            t0 = time.perf_counter()
            mineru_elapsed = 0.0
            mineru_status = "skipped"
            mineru_device: str | None = None

            # ── Step 1: MinerU extraction (only if needed) ──────────────────
            if _mineru_ready(doc_id):
                print(f"[{doc_id}] MinerU: already extracted, skipping")
            else:
                pdf_path = _find_pdf(doc_id, pdf_dir)
                if pdf_path is None:
                    print(f"[{doc_id}] SKIP — PDF not found in {_display_path(pdf_dir)}")
                    rec = {
                        "doc_id": doc_id, "status": "skipped",
                        "reason": f"pdf_not_found_in_{pdf_dir}",
                    }
                    records.append(rec)
                    log_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    log_fh.flush()
                    continue

                _clean_mineru_output(doc_id)
                print(f"[{doc_id}] MinerU: extracting {pdf_path.name}...")
                t_mineru = time.perf_counter()
                mineru_result, mineru_device = _run_mineru(doc_id, pdf_path, verbose)
                mineru_elapsed = round(time.perf_counter() - t_mineru, 1)

                if mineru_result.returncode != 0:
                    print(f"[{doc_id}] ✗ MinerU failed ({mineru_elapsed}s, device={mineru_device})")
                    rec = {
                        "doc_id": doc_id, "status": "failed",
                        "stage": "mineru", "exit_code": mineru_result.returncode,
                        "elapsed_s": round(time.perf_counter() - t0, 1),
                        "mineru_elapsed_s": mineru_elapsed,
                        "mineru_device": mineru_device,
                    }
                    records.append(rec)
                    log_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    log_fh.flush()
                    total_errors += 1
                    continue

                mineru_status = "run"
                print(f"[{doc_id}] MinerU: done ({mineru_elapsed}s, device={mineru_device})")

            # ── Step 2: Sidecar generation ───────────────────────────────────
            print(f"[{doc_id}] Sidecar: starting...")
            cmd = [sys.executable, "processors/build_sidecar.py", doc_id] + sidecar_extra_args
            result = subprocess.run(cmd, cwd=ROOT, capture_output=not verbose, text=True)

            if not verbose and result.stdout:
                print(result.stdout, end="")
            if not verbose and result.stderr:
                print(result.stderr, end="", file=sys.stderr)

            elapsed = round(time.perf_counter() - t0, 1)
            sidecar_elapsed = round(elapsed - mineru_elapsed, 1)
            stats = _read_manifest_stats(doc_id)
            fig_errors = _count_figure_errors(doc_id) if result.returncode == 0 else 0
            status = "success" if result.returncode == 0 else "failed"
            total_errors += fig_errors + (1 if result.returncode != 0 else 0)

            record = {
                "doc_id": doc_id,
                "status": status,
                "exit_code": result.returncode,
                "elapsed_s": elapsed,
                "mineru_elapsed_s": mineru_elapsed,
                "mineru_status": mineru_status,
                "mineru_device": mineru_device if mineru_status == "run" else None,
                "sidecar_elapsed_s": sidecar_elapsed,
                "figures_total": stats.get("figures_total"),
                "images_extracted_total": stats.get("images_extracted_total"),
                "duplicate_figure_ids": stats.get("duplicate_figure_ids"),
                "ambiguous_figures": stats.get("ambiguous_figures"),
                "tables_total": stats.get("tables_total"),
                "figure_alt_text_errors": fig_errors,
            }
            records.append(record)
            log_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_fh.flush()

            label = "✓" if status == "success" else "✗"
            err_note = f" ({fig_errors} fig errors)" if fig_errors else ""
            mineru_note = f" (MinerU {mineru_elapsed}s)" if mineru_status == "run" else ""
            print(f"[{doc_id}] {label} {elapsed}s{mineru_note} — "
                  f"{stats.get('figures_total', '?')} figures, "
                  f"{stats.get('tables_total', '?')} tables{err_note}\n")

        summary = {
            "type": "summary",
            "timestamp": timestamp,
            "doc_ids": doc_ids,
            "total_papers": len(doc_ids),
            "succeeded": sum(1 for r in records if r.get("status") == "success"),
            "failed": sum(1 for r in records if r.get("status") == "failed"),
            "skipped": sum(1 for r in records if r.get("status") == "skipped"),
            "total_errors": total_errors,
            "sidecar_args": sidecar_extra_args,
        }
        records.append(summary)
        log_fh.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"=== Done: {summary['succeeded']}/{len(doc_ids)} succeeded, "
          f"{summary['failed']} failed, {summary['skipped']} skipped ===")
    print(f"Log: {log_path.relative_to(ROOT)}")


def _display_path(p: Path) -> str:
    """Return a short path for display: relative to ROOT if inside, else absolute."""
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def _resolve_pdf_dir(pdf_dir_arg: str) -> Path:
    p = Path(pdf_dir_arg).expanduser()
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists() or not p.is_dir():
        print(f"Error: PDF directory not found: {p}")
        sys.exit(1)
    return p


def _discover_doc_ids(pdf_dir: Path) -> list[str]:
    """Scan pdf_dir (non-recursive) for *.pdf files; derive doc_id from stem."""
    seen: dict[str, Path] = {}
    for f in sorted(pdf_dir.iterdir()):
        if f.is_file() and f.suffix.lower() == ".pdf":
            stem = f.stem
            if stem in seen:
                print(f"Warning: duplicate doc_id '{stem}' from {seen[stem].name} "
                      f"and {f.name} — skipping {f.name}")
            else:
                seen[stem] = f
    return list(seen.keys())


def _build_sidecar_extra_args(args: argparse.Namespace) -> list[str]:
    extra: list[str] = []
    if args.pull_model:
        extra.append("--pull-model")
    if args.skip_vlm:
        extra.append("--skip-vlm")
    if args.skip_alt_text:
        extra.append("--skip-alt-text")
    if args.model:
        extra += ["--model", args.model]
    if args.vlm_model:
        extra += ["--vlm-model", args.vlm_model]
    return extra


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full PDF→sidecar pipeline for one or many papers."
    )
    parser.add_argument("doc_ids", nargs="*", default=[],
                        help="doc_ids to process; if omitted, auto-discovers PDFs in --pdf-dir")
    parser.add_argument("--pdf-dir", default="test_paper",
                        help="directory to look for {doc_id}.pdf (default: test_paper/)")
    parser.add_argument("--pull-model", action="store_true",
                        help="auto-pull Ollama VLM model if not present")
    parser.add_argument("--skip-vlm", action="store_true",
                        help="skip Ollama VLM repair step")
    parser.add_argument("--skip-alt-text", action="store_true",
                        help="skip Gemini figure alt-text generation")
    parser.add_argument("--model", default=None, help="Gemini model override")
    parser.add_argument("--vlm-model", default=None, help="Ollama VLM model override")
    args = parser.parse_args()

    pdf_dir = _resolve_pdf_dir(args.pdf_dir)

    if args.doc_ids:
        doc_ids = args.doc_ids
    else:
        doc_ids = _discover_doc_ids(pdf_dir)
        if not doc_ids:
            print(f"No PDFs found in {_display_path(pdf_dir)}")
            sys.exit(1)
        print(f"Auto-discovered {len(doc_ids)} PDF(s) in "
              f"{_display_path(pdf_dir)}: {', '.join(doc_ids)}\n")

    run_batch(doc_ids, _build_sidecar_extra_args(args), pdf_dir=pdf_dir)
