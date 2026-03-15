"""
ARCHAEOLOGICAL SITE PREDICTION PIPELINE
Step 3: Export to QGIS-ready formats

Dependencies:
    pip install geopandas shapely

Usage:
    python 03_export_qgis.py --input geocoded_sites.json --output_dir ./qgis_output

Outputs:
    - known_sites.gpkg       — GeoPackage with all geocoded sites (open directly in QGIS)
    - known_sites.geojson    — GeoJSON alternative
    - sites_for_review.csv   — Low-confidence / failed geocodes for manual correction
    - summary_stats.json     — Quick statistics on your dataset
"""

import json
import argparse
from pathlib import Path
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

BASE           = "/Users/sudet/Documents/Study"
DEFAULT_INPUT  = BASE + "/data/geocoded_sites.json"
DEFAULT_OUTPUT = BASE + "/qgis"


# Fields to keep in the QGIS layer attribute table
# (Trim this to what's useful; QGIS handles long attribute tables fine but this keeps it readable)
ATTRIBUTE_FIELDS = [
    "site_name",
    "site_type",
    "period",
    "culture",
    "certainty",
    "geocode_confidence",
    "elevation_masl",
    "proximity_to_water",
    "soil_type",
    "source_pdf",
    "source_language",
    "finds_str",       # flattened from list
    "notes",
    "location_description",
    "country",
    "geocode_method",
]


def export_to_qgis(input_path: str, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)
    
    sites = data["sites"]
    print(f"Processing {len(sites)} sites for export...")
    
    # Separate geocoded vs failed
    geocoded = [s for s in sites if s.get("latitude") and s.get("longitude")]
    failed   = [s for s in sites if not (s.get("latitude") and s.get("longitude"))]
    low_conf = [s for s in geocoded if s.get("geocode_confidence") in ("low", "failed")]
    
    print(f"  ✓ Successfully geocoded: {len(geocoded)}")
    print(f"  ⚠ Failed geocoding:      {len(failed)}")
    print(f"  ⚠ Low confidence:        {len(low_conf)}")
    
    # ── Build GeoDataFrame ────────────────────────────────────────────────────
    rows = []
    for site in geocoded:
        row = {}
        for field in ATTRIBUTE_FIELDS:
            if field == "finds_str":
                finds = site.get("finds", [])
                row["finds_str"] = ", ".join(f for f in finds if f) if isinstance(finds, list) else str(finds or "")
            else:
                row[field] = site.get(field)
        row["latitude"]  = site["latitude"]
        row["longitude"] = site["longitude"]
        row["geometry"]  = Point(site["longitude"], site["latitude"])
        rows.append(row)
    
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    
    # ── Export GeoPackage (preferred for QGIS) ────────────────────────────────
    gpkg_path = output_dir / "known_sites.gpkg"
    gdf.to_file(gpkg_path, driver="GPKG", layer="known_sites")
    print(f"\n✅ GeoPackage: {gpkg_path}")
    
    # ── Export GeoJSON ────────────────────────────────────────────────────────
    geojson_path = output_dir / "known_sites.geojson"
    gdf.to_file(geojson_path, driver="GeoJSON")
    print(f"✅ GeoJSON:    {geojson_path}")
    
    # ── Export review CSV ─────────────────────────────────────────────────────
    review_sites = failed + low_conf
    if review_sites:
        review_df = pd.DataFrame(review_sites)
        review_path = output_dir / "sites_for_review.csv"
        review_df.to_csv(review_path, index=False, encoding="utf-8")
        print(f"⚠ Review CSV: {review_path}  ({len(review_sites)} sites need manual geocoding)")
    
    # ── Summary stats ─────────────────────────────────────────────────────────
    stats = {
        "total_sites": len(sites),
        "geocoded": len(geocoded),
        "failed_geocoding": len(failed),
        "by_type": gdf["site_type"].value_counts().to_dict() if "site_type" in gdf.columns else {},
        "by_period": gdf["period"].value_counts().head(20).to_dict() if "period" in gdf.columns else {},
        "by_culture": gdf["culture"].value_counts().head(20).to_dict() if "culture" in gdf.columns else {},
        "by_confidence": gdf["geocode_confidence"].value_counts().to_dict() if "geocode_confidence" in gdf.columns else {},
        "by_country": gdf["country"].value_counts().to_dict() if "country" in gdf.columns else {},
    }
    
    stats_path = output_dir / "summary_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"✅ Stats:      {stats_path}")
    
    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 QGIS QUICK START
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 1. Open QGIS
 2. Drag known_sites.gpkg onto the map canvas
 3. Style by 'period' or 'site_type' for quick overview
 4. Use 'geocode_confidence' field to filter:
      high   = parsed from text coordinates  ← trust these
      medium = Nominatim town/village match  ← usually good
      low    = Nominatim broad match         ← verify manually
 5. Manually fix sites_for_review.csv and re-import

 Next step: run 04_predict.py to generate probability heatmap raster
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    export_to_qgis(args.input, args.output_dir)
