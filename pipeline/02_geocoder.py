"""
ARCHAEOLOGICAL SITE PREDICTION PIPELINE
Step 2: Geocoding — resolve location descriptions to lat/lon

Dependencies:
    pip install geopy tqdm

Usage:
    python 02_geocoder.py --input extracted_sites.json --output geocoded_sites.json

This script:
    1. Takes sites that already have coordinates_raw and parses them
    2. For sites with only text descriptions, uses Nominatim (free, OSM-based) to geocode
    3. Flags low-confidence geocodes for manual review
    4. Outputs enriched JSON ready for QGIS or the predictive model
"""

import json
import re
import time
import argparse
from pathlib import Path
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from tqdm import tqdm


# ── CONFIG ────────────────────────────────────────────────────────────────────

# Nominatim requires a descriptive user agent
NOMINATIM_USER_AGENT = "ce_archeo_predictor/1.0 (sudet_archaeology_research)"

BASE           = "/Users/sudet/Documents/Study"
DEFAULT_INPUT  = BASE + "/data/extracted_sites.json"
DEFAULT_OUTPUT = BASE + "/data/geocoded_sites.json"

# Bounding box for Central Europe — reject geocodes outside this area
# Expand if your research area extends further
CE_BBOX = {
    "lat_min": 44.0,
    "lat_max": 55.0,
    "lon_min": 12.0,
    "lon_max": 28.0
}

# Seconds between Nominatim requests (required by usage policy: max 1/sec)
NOMINATIM_DELAY = 1.1


# ── COORDINATE PARSERS ────────────────────────────────────────────────────────

def parse_decimal_degrees(text: str):
    """Try to parse decimal degree coordinates like '48.2345 N, 17.1234 E'"""
    pattern = r'(\d{1,3}\.\d+)\s*°?\s*([NS])[,\s]+(\d{1,3}\.\d+)\s*°?\s*([EW])'
    m = re.search(pattern, text, re.IGNORECASE)
    if m:
        lat = float(m.group(1)) * (-1 if m.group(2).upper() == 'S' else 1)
        lon = float(m.group(3)) * (-1 if m.group(4).upper() == 'W' else 1)
        return lat, lon
    
    # Try bare decimal: "48.2345, 17.1234"
    pattern2 = r'(-?\d{1,3}\.\d{3,})[,\s]+(-?\d{1,3}\.\d{3,})'
    m2 = re.search(pattern2, text)
    if m2:
        lat, lon = float(m2.group(1)), float(m2.group(2))
        if 90 >= abs(lat) and 180 >= abs(lon):
            return lat, lon
    
    return None, None


def parse_dms_coordinates(text: str):
    """Try to parse degrees-minutes-seconds like 48°30'15\"N 17°45'20\"E"""
    pattern = r'(\d{1,3})°\s*(\d{1,2})[\'′]\s*(\d{1,2}(?:\.\d+)?)[\"″]?\s*([NS])[,\s]+(\d{1,3})°\s*(\d{1,2})[\'′]\s*(\d{1,2}(?:\.\d+)?)[\"″]?\s*([EW])'
    m = re.search(pattern, text, re.IGNORECASE)
    if m:
        lat = int(m.group(1)) + int(m.group(2))/60 + float(m.group(3))/3600
        lon = int(m.group(5)) + int(m.group(6))/60 + float(m.group(7))/3600
        if m.group(4).upper() == 'S': lat = -lat
        if m.group(8).upper() == 'W': lon = -lon
        return lat, lon
    return None, None


def try_parse_raw_coordinates(raw):
    """Try multiple coordinate formats on a raw string."""
    if not raw:
        return None, None
    # Handle case where raw is a list
    if isinstance(raw, list):
        raw = ' '.join(str(x) for x in raw if x)
    if not isinstance(raw, str):
        raw = str(raw)
    for parser in [parse_decimal_degrees, parse_dms_coordinates]:
        lat, lon = parser(raw)
        if lat is not None:
            return lat, lon
    return None, None


# ── GEOCODING ────────────────────────────────────────────────────────────────

def in_central_europe(lat, lon):
    """Check if coordinates fall within our study area."""
    return (CE_BBOX["lat_min"] <= lat <= CE_BBOX["lat_max"] and
            CE_BBOX["lon_min"] <= lon <= CE_BBOX["lon_max"])


def build_geocoding_query(site: dict) -> list:
    """
    Build a prioritized list of geocoding queries for a site.
    Returns queries from most to least specific.
    """
    queries = []
    name = site.get("site_name", "")
    location = site.get("location_description", "")
    country = site.get("country", "")
    
    if name and location:
        queries.append(f"{name}, {location}, {country}".strip(", "))
    if location:
        queries.append(f"{location}, {country}".strip(", "))
    if name:
        queries.append(f"{name}, {country}".strip(", "))
    
    return [q for q in queries if q.strip()]


def geocode_site(geolocator, site: dict) -> tuple:
    """
    Attempt to geocode a site. Returns (lat, lon, confidence, method).
    confidence: 'high' (parsed from text), 'medium' (nominatim specific), 
                'low' (nominatim broad), 'failed'
    """
    # First try: parse coordinates directly from raw text
    raw = site.get("coordinates_raw", "")
    if raw:
        lat, lon = try_parse_raw_coordinates(raw)
        if lat is not None and in_central_europe(lat, lon):
            return lat, lon, "high", "parsed_from_text"
    
    # Second try: Nominatim geocoding
    queries = build_geocoding_query(site)
    for query in queries:
        try:
            result = geolocator.geocode(
                query,
                exactly_one=True,
                timeout=10,
                viewbox=[
                    (CE_BBOX["lon_min"], CE_BBOX["lat_max"]),
                    (CE_BBOX["lon_max"], CE_BBOX["lat_min"])
                ],
                bounded=False  # Don't restrict — just prefer CE results
            )
            time.sleep(NOMINATIM_DELAY)
            
            if result:
                lat, lon = result.latitude, result.longitude
                if in_central_europe(lat, lon):
                    # Assess confidence based on result type
                    raw_data = result.raw
                    osm_type = raw_data.get("type", "")
                    confidence = "medium" if osm_type in ["city", "town", "village", "hamlet", "archaeological_site"] else "low"
                    return lat, lon, confidence, f"nominatim:{query}"
        
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            print(f"    ⚠ Geocoder error for '{query}': {e}")
            time.sleep(5)
    
    return None, None, "failed", "no_result"


# ── MAIN ──────────────────────────────────────────────────────────────────────

def geocode_all_sites(input_path: str, output_path: str):
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)
    
    sites = data["sites"]
    print(f"Geocoding {len(sites)} sites...")

    # Resume: load already geocoded results if output exists
    already_done = {}
    if Path(output_path).exists():
        with open(output_path, encoding="utf-8") as f:
            existing = json.load(f)
        for s in existing.get("sites", []):
            if s.get("geocode_confidence") and s.get("geocode_confidence") != "failed":
                key = (s.get("site_name"), s.get("location_description"))
                already_done[key] = s
        print(f"Resuming — {len(already_done)} sites already geocoded\n")

    geolocator = Nominatim(user_agent=NOMINATIM_USER_AGENT)
    results = {"high": 0, "medium": 0, "low": 0, "failed": 0}

    def save_progress():
        data["geocoding_summary"] = results
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    for site in tqdm(sites, desc="Geocoding"):
        key = (site.get("site_name"), site.get("location_description"))
        if key in already_done:
            cached = already_done[key]
            site["latitude"] = cached.get("latitude")
            site["longitude"] = cached.get("longitude")
            site["geocode_confidence"] = cached.get("geocode_confidence")
            site["geocode_method"] = cached.get("geocode_method")
            results[site["geocode_confidence"]] += 1
            continue

        lat, lon, confidence, method = geocode_site(geolocator, site)
        site["latitude"] = lat
        site["longitude"] = lon
        site["geocode_confidence"] = confidence
        site["geocode_method"] = method
        results[confidence] += 1

        # Save every 100 sites
        if sum(results.values()) % 100 == 0:
            save_progress()
    
    print(f"\n📊 Geocoding results:")
    print(f"   High confidence (parsed coords):  {results['high']}")
    print(f"   Medium confidence (Nominatim):    {results['medium']}")
    print(f"   Low confidence (Nominatim broad): {results['low']}")
    print(f"   Failed:                           {results['failed']}")
    print(f"   ⚠ Review 'low' and 'failed' entries manually before modeling")
    
    data["geocoding_summary"] = results
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Saved to {output_path}")
    print(f"   Next step: run 03_export_qgis.py to generate QGIS-ready files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    geocode_all_sites(args.input, args.output)
