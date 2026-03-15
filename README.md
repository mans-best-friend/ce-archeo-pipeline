# CE Archaeological Site Prediction Pipeline

An open-source pipeline for predicting undiscovered prehistoric archaeological sites in Central Europe using machine learning and freely available environmental data.

Built as a portfolio project for a Czech archaeology programme application (entrance exams June 2026). Runs entirely on a personal computer using free tools — no cloud, no subscriptions.

---

## What it does

1. **Extracts** site data from a multilingual corpus of archaeological PDFs (Czech, Slovak, German, Polish, Hungarian, Romanian) using a local LLM (Mistral 7B via LM Studio)
2. **Geocodes** extracted site names using Nominatim/OpenStreetMap
3. **Exports** to GeoPackage for QGIS visualisation
4. **Trains** a Random Forest classifier on known site locations using real environmental data
5. **Predicts** site probability across the study area and outputs a GeoTIFF probability heatmap

## Current results

- **281 papers** processed from a multilingual CE archaeological corpus
- **3,011 geocoded sites** across Czech Republic, Slovakia, Austria, Poland, Germany
- **Spatial block cross-validation AUC: 0.756 ± 0.136** (5 geographic folds, west→east)
- Five environmental features: SRTM elevation, slope, aspect, OSM water distance, BPEJ soil favourability

## Study area

Czech Republic and Slovakia (lon 12–22°E, lat 47–52°N), with the predictive model focused on Czech territory.

## Environmental data sources

| Layer | Source | Resolution |
|-------|--------|------------|
| Elevation, slope, aspect | SRTM via CGIAR | 90m |
| Distance to water | OpenStreetMap via Geofabrik | — |
| Soil favourability | BPEJ (SPÚ ČR) | 1:5000 |

## Pipeline scripts

```
python3 pipeline/rename_pdfs.py             # batch rename PDFs using LLM
python3 pipeline/01_pdf_extractor_local.py  # extract site data from PDFs
python3 pipeline/02_geocoder.py             # geocode extracted sites
python3 pipeline/03_export_qgis.py          # export to GeoPackage
python3 pipeline/04_predict.py              # train model + generate heatmap
```

All scripts are resume-safe — interrupted runs continue from the last saved point.

## Dependencies

```bash
pip install geopandas numpy scikit-learn scipy osmnx pymupdf geopy
```

LM Studio with Mistral 7B Instruct Q4_K_M required for PDF extraction steps.
QGIS 3.34+ required for raster export (uses built-in GDAL Python bindings).

## Methodology notes

- **Presence-only model** — background points used as pseudo-absences
- **Spatial block cross-validation** — study area divided into 5 longitudinal strips to prevent spatial autocorrelation from inflating AUC
- **No parameter tuning toward known outcomes** — model performance on known archaeological distributions is documented, not optimised for

## Known limitations

- 90m raster resolution underestimates potential in narrow river valley corridors (e.g. Labe/Elbe near Litoměřice)
- BPEJ soil layer covers Czech and Slovak agricultural land only — incomplete coverage reduces feature importance
- Modern environmental layers are imperfect proxies for prehistoric landscape conditions
- ~49% geocoding failure rate due to vague place descriptions in source literature

## Roadmap

- [ ] Proper spatial block CV using geographic clustering rather than longitudinal strips
- [ ] LiDAR derivatives from ČÚZK DMR5G for Czech territory
- [ ] Per-period models (Neolithic, Eneolithic, Bronze Age separately)
- [ ] OCR pipeline for 185 scanned PDFs
- [ ] AMČR database integration for independent validation

## Project presentation

Bilingual HTML presentation (EN/CS) available in `docs/`.

---

*Developed as an independent research project, 2025–2026.*
*All data sources are freely available. Pipeline runs on Apple M1 16GB, macOS Ventura, Python 3.13.*
