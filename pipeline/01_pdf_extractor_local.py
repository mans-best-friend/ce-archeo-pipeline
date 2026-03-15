"""
ARCHAEOLOGICAL SITE PREDICTION PIPELINE
Step 1: PDF Extraction & LLM-based Entity Extraction
LOCAL VERSION — uses LM Studio running on localhost

Dependencies:
    pip install pymupdf requests tqdm

Usage:
    python 01_pdf_extractor_local.py --pdf_dir /path/to/your/pdfs --output extracted_sites.json

Requirements:
    - LM Studio running with a model loaded
    - Local server started in LM Studio (port 1234 by default)
"""

import os
import json
import argparse
import time
from pathlib import Path
import fitz  # PyMuPDF
import requests
from tqdm import tqdm


# ── CONFIG ────────────────────────────────────────────────────────────────────

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"

BASE        = "/Users/sudet/Documents/Study"
DEFAULT_PDF_DIR = BASE + "/pdfs_renamed"
DEFAULT_OUTPUT  = BASE + "/data/extracted_sites.json"

# These settings work well for Mistral 7B Instruct on M1 16GB
# Lower max_tokens if you get memory warnings
LM_STUDIO_PARAMS = {
    "temperature": 0.1,       # Low temperature = more consistent JSON output
    "max_tokens": 2048,
    "stream": False,
}

CHUNK_SIZE_CHARS = 4000       # Smaller than Claude version — local models have shorter context
OVERLAP_CHARS = 300
MAX_CHUNKS_PER_PDF = 40


# ── PROMPTS ───────────────────────────────────────────────────────────────────

# Mistral Instruct works better with a slightly more explicit prompt than Claude
SYSTEM_PROMPT = """You are an archaeological data extraction assistant. 
You read academic texts in any language and extract structured data about archaeological sites.
You ALWAYS respond with valid JSON only. No explanation. No markdown. No preamble. Just JSON."""

EXTRACTION_PROMPT_TEMPLATE = """Extract all archaeological site mentions from the text below.

Respond ONLY with this exact JSON structure, nothing else:

{{
  "sites": [
    {{
      "site_name": "name of site or null",
      "location_description": "town, region, river or landmark mentioned",
      "country": "country name or null",
      "coordinates_raw": "any coordinates mentioned verbatim or null",
      "site_type": "one of: settlement, burial, hoard, ritual, lithic_scatter, survey_area, unknown",
      "period": "archaeological period e.g. Early Bronze Age, Neolithic, La Tene",
      "culture": "archaeological culture e.g. Unetice, Linear Pottery or null",
      "finds": ["list", "of", "find", "types"],
      "elevation_masl": null,
      "proximity_to_water": "e.g. 200m from Morava river or null",
      "soil_type": "soil type if mentioned or null",
      "certainty": "one of: confirmed, probable, mentioned_only",
      "source_language": "language of the source text",
      "notes": "any extra useful context or null"
    }}
  ]
}}

Rules:
- If no sites are mentioned, return {{"sites": []}}
- Never invent data. Use null for missing fields.
- Detect and handle text in any language including Czech, Slovak, Polish, Hungarian, Romanian, German, Slovenian, Croatian.

Text:
---
{text}
---"""


# ── PDF EXTRACTION ────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: Path) -> str:
    doc = fitz.open(str(pdf_path))
    pages = []
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            pages.append(text)
    doc.close()
    return "\n\n".join(pages)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_CHARS, overlap: int = OVERLAP_CHARS):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            newline_pos = text.rfind("\n\n", start, end)
            if newline_pos > start + chunk_size // 2:
                end = newline_pos
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


# ── LLM CALL ─────────────────────────────────────────────────────────────────

def extract_sites_from_chunk(text_chunk: str, retries: int = 3) -> list:
    """Send chunk to LM Studio local server, return list of site dicts."""
    
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(text=text_chunk)
    
    payload = {
        **LM_STUDIO_PARAMS,
        "messages": [
            {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + prompt}
        ]
    }
    
    for attempt in range(retries):
        try:
            response = requests.post(
                LM_STUDIO_URL,
                json=payload,
                timeout=120   # Local inference can be slow — give it 2 minutes
            )
            response.raise_for_status()
            
            raw = response.json()["choices"][0]["message"]["content"].strip()
            
            # Strip markdown fences if model adds them
            if "```" in raw:
                parts = raw.split("```")
                for part in parts:
                    if "{" in part:
                        raw = part.strip()
                        if raw.startswith("json"):
                            raw = raw[4:].strip()
                        break
            
            # Find JSON object in response even if there's surrounding text
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                raw = raw[start:end]
            
            data = json.loads(raw)
            return data.get("sites", [])
        
        except requests.exceptions.ConnectionError:
            print(f"\n  ❌ Cannot connect to LM Studio.")
            print(f"     Make sure the local server is running in LM Studio (port 1234).")
            return []
        except json.JSONDecodeError as e:
            print(f"\n  ⚠ JSON parse error (attempt {attempt+1}): {e}")
            print(f"     Raw response: {raw[:200]}...")
            time.sleep(2)
        except Exception as e:
            print(f"\n  ⚠ Error (attempt {attempt+1}): {e}")
            time.sleep(3)
    
    return []


# ── DEDUPLICATION ─────────────────────────────────────────────────────────────

def deduplicate_sites(sites: list) -> list:
    seen = {}
    for site in sites:
        key = (
            (site.get("site_name") or "").lower().strip(),
            (site.get("location_description") or "").lower().strip()[:80]
        )
        if key == ("", ""):
            continue
        if key not in seen:
            seen[key] = site
        else:
            existing = seen[key]
            for field, value in site.items():
                if value and not existing.get(field):
                    existing[field] = value
    return list(seen.values())


# ── CONNECTION TEST ───────────────────────────────────────────────────────────

def test_lm_studio_connection():
    """Quick test to verify LM Studio server is reachable before processing."""
    try:
        response = requests.post(
            LM_STUDIO_URL,
            json={
                "messages": [{"role": "user", "content": "Reply with: {\"ok\": true}"}],
                "max_tokens": 20,
                "temperature": 0.1,
                "stream": False
            },
            timeout=30
        )
        response.raise_for_status()
        print("✅ LM Studio connection OK")
        return True
    except Exception as e:
        print(f"❌ LM Studio connection failed: {e}")
        print("   → Open LM Studio → Local Server tab → Start Server")
        return False


# ── MAIN ──────────────────────────────────────────────────────────────────────

def process_pdf_folder(pdf_dir: str, output_path: str):
    
    # Test connection first
    if not test_lm_studio_connection():
        return
    
    pdf_folder = Path(pdf_dir)
    pdf_files = sorted(pdf_folder.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDFs to process\n")
    
    all_sites = []
    processing_log = []
    
    # Load existing progress if resuming
    if Path(output_path).exists():
        with open(output_path, encoding="utf-8") as f:
            existing = json.load(f)
            all_sites = existing.get("sites", [])
            processing_log = existing.get("processing_log", [])
            already_done = {entry["file"] for entry in processing_log}
            print(f"Resuming — {len(already_done)} PDFs already processed\n")
    else:
        already_done = set()
    
    def save_progress():
        deduped = deduplicate_sites(all_sites)
        output = {
            "metadata": {
                "total_pdfs": len(pdf_files),
                "total_sites_raw": len(all_sites),
                "total_sites_deduped": len(deduped),
                "model": "local/mistral-7b-instruct via LM Studio",
            },
            "processing_log": processing_log,
            "sites": deduped
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
    
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        print(f"\n📄 {pdf_path.name}")
        
        # Skip if already processed in a previous run
        if pdf_path.name in already_done:
            print(f"  ✓ Already processed — skipping")
            continue
        
        try:
            text = extract_text_from_pdf(pdf_path)
        except Exception as e:
            print(f"  ❌ Failed to extract text: {e}")
            processing_log.append({"file": pdf_path.name, "status": "extraction_failed", "error": str(e)})
            continue
        
        if len(text.strip()) < 200:
            print(f"  ⚠ Very short text ({len(text)} chars) — likely a scanned PDF, needs OCR")
            processing_log.append({"file": pdf_path.name, "status": "likely_scanned", "chars": len(text)})
            save_progress()
            continue
        
        chunks = chunk_text(text)[:MAX_CHUNKS_PER_PDF]
        print(f"  → {len(text):,} chars → {len(chunks)} chunks")
        
        file_sites = []
        for i, chunk in enumerate(chunks):
            print(f"  chunk {i+1}/{len(chunks)}...", end="\r")
            sites = extract_sites_from_chunk(chunk)
            for site in sites:
                site["source_pdf"] = pdf_path.name
                site["chunk_index"] = i
            file_sites.extend(sites)
            # Small delay to avoid overwhelming local server
            time.sleep(0.2)
        
        print(f"  ✓ Extracted {len(file_sites)} site mentions        ")
        all_sites.extend(file_sites)
        processing_log.append({
            "file": pdf_path.name,
            "status": "ok",
            "sites_found": len(file_sites)
        })
        save_progress()  # Save after every PDF
    
    print(f"\n📊 Total raw site mentions: {len(all_sites)}")
    deduped = deduplicate_sites(all_sites)
    print(f"📊 After deduplication: {len(deduped)}")
    
    output = {
        "metadata": {
            "total_pdfs": len(pdf_files),
            "total_sites_raw": len(all_sites),
            "total_sites_deduped": len(deduped),
            "model": "local/mistral-7b-instruct via LM Studio",
        },
        "processing_log": processing_log,
        "sites": deduped
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Saved to {output_path}")
    print(f"   Next step: python 02_geocoder.py --input {output_path} --output geocoded_sites.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", default=DEFAULT_PDF_DIR, help="Folder containing PDF research papers")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSON file path")
    args = parser.parse_args()
    process_pdf_folder(args.pdf_dir, args.output)
