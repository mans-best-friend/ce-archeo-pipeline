"""
PDF AUTO-RENAMER
Hardcoded for: /Users/sudet/Documents/Study

Reads from:   pdfs_test/      (put your test copies here)
Writes to:    pdfs_renamed/   (renamed copies land here)
Log saved to: data/rename_log.json
Originals in: pdfs_original/  (NEVER touched)

Usage:
    python3 rename_pdfs.py --dry_run   # preview only
    python3 rename_pdfs.py             # actually copy and rename
"""

import re
import json
import time
import shutil
import argparse
import requests
from pathlib import Path
import fitz  # PyMuPDF

# ── PATHS ─────────────────────────────────────────────────────────────────────

BASE       = Path("/Users/sudet/Documents/Study")
SOURCE_DIR = BASE / "pdfs_original"
DEST_DIR   = BASE / "pdfs_renamed"
LOG_PATH   = BASE / "data" / "rename_log.json"

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MAX_CHARS     = 2000

# ── PROMPT ────────────────────────────────────────────────────────────────────

PROMPT = """Extract bibliographic info from this academic paper text.
Respond ONLY with JSON, nothing else:

{{
  "title": "full title of the paper",
  "first_author_surname": "surname of first author only",
  "year": "publication year as 4 digits or null"
}}

If you cannot determine a field with confidence, use null.

Text:
---
{text}
---"""

# ── HELPERS ───────────────────────────────────────────────────────────────────

def sanitize(text: str, max_len: int = 40) -> str:
    if not text:
        return "Unknown"
    clean = re.sub(r'[^\w\s-]', '', text)
    camel = ''.join(w.capitalize() for w in clean.split()[:6])
    return camel[:max_len]

def extract_first_pages(pdf_path: Path) -> str:
    try:
        doc = fitz.open(str(pdf_path))
        pages = []
        for i, page in enumerate(doc):
            if i >= 2:
                break
            t = page.get_text("text").strip()
            if t:
                pages.append(t)
        doc.close()
        return "\n\n".join(pages)[:MAX_CHARS]
    except Exception as e:
        print(f"    ⚠ Could not read: {e}")
        return ""

def get_info(text: str) -> dict:
    try:
        r = requests.post(LM_STUDIO_URL, json={
            "messages": [{"role": "user", "content": PROMPT.format(text=text)}],
            "temperature": 0.1, "max_tokens": 200, "stream": False
        }, timeout=60)
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"].strip()
        s, e = raw.find("{"), raw.rfind("}") + 1
        if s >= 0 and e > s:
            return json.loads(raw[s:e])
    except Exception as ex:
        print(f"    ⚠ {ex}")
    return {}

def build_name(info: dict, stem: str) -> str:
    author = sanitize(info.get("first_author_surname", ""), 20) or "UnknownAuthor"
    year   = info.get("year") or "XXXX"
    title  = sanitize(info.get("title", ""), 50) or sanitize(stem, 40)
    return f"{author}_{year}_{title}"

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main(dry_run: bool):
    # Check LM Studio
    try:
        requests.post(LM_STUDIO_URL, json={
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5, "stream": False
        }, timeout=10)
        print("✅ LM Studio OK")
    except:
        print("❌ LM Studio not reachable — start the local server first")
        return

    pdfs = sorted(SOURCE_DIR.rglob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {SOURCE_DIR}")
        print(f"Copy some test PDFs there first")
        return

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nFound {len(pdfs)} PDFs in {SOURCE_DIR}")
    print("DRY RUN\n" if dry_run else f"LIVE — copies → {DEST_DIR}\n")

    log = []
    skipped = done = 0

    for pdf in pdfs:
        print(f"📄 {pdf.name}")
        text = extract_first_pages(pdf)
        if not text:
            print("   ⚠ No text — skipping")
            skipped += 1
            continue

        info = get_info(text)
        if not info:
            print("   ⚠ No info extracted — skipping")
            skipped += 1
            continue

        stem    = build_name(info, pdf.stem)
        newname = stem + ".pdf"
        newpath = DEST_DIR / newname

        # Handle duplicates
        i = 1
        while newpath.exists():
            newname = f"{stem}_{i}.pdf"
            newpath = DEST_DIR / newname
            i += 1

        print(f"   → {newname}")
        print(f"      {info.get('first_author_surname','?')}, {info.get('year','?')}: {str(info.get('title','?'))[:70]}")

        log.append({"original": pdf.name, "new": newname, **info})

        if not dry_run:
            shutil.copy2(pdf, newpath)
            done += 1

        time.sleep(0.3)

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print(f"\n{'─'*55}")
    if dry_run:
        print(f"DRY RUN done — would process {len(log)}, skip {skipped}")
        print("Run without --dry_run to apply")
    else:
        print(f"Done — copied & renamed: {done}, skipped: {skipped}")
    print(f"Log: {LOG_PATH}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    main(dry_run=args.dry_run)
