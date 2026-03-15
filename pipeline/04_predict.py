"""
ARCHAEOLOGICAL SITE PREDICTION PIPELINE
Step 4: Predictive Spatial Model → probability heatmap

Dependencies:
    pip install geopandas numpy scikit-learn scipy

Usage:
    python3 04_predict.py

What this does:
    1. Loads known sites from known_sites.gpkg
    2. Generates synthetic environmental features (replace with real rasters later)
    3. Trains a Random Forest classifier (site vs background)
    4. Predicts probability across the full study area
    5. Saves probability_heatmap.npy for QGIS reconstruction

IMPORTANT:
    Output is a RELATIVE PROBABILITY SURFACE, not a ground truth map.
    High probability = environmentally similar to known sites.
    Always interpret with field knowledge and domain expertise.
"""

import argparse
import warnings
from pathlib import Path
import numpy as np
import geopandas as gpd
from scipy.ndimage import gaussian_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import json

BASE           = "/Users/sudet/Documents/Study"
DEFAULT_SITES  = BASE + "/qgis/known_sites.gpkg"
DEFAULT_OUTPUT = BASE + "/qgis"

warnings.filterwarnings("ignore")

# ── STUDY AREA ────────────────────────────────────────────────────────────────
STUDY_BBOX = {
    "lon_min": 12.0,
    "lon_max": 22.0,
    "lat_min": 47.0,
    "lat_max": 52.0
}

OUTPUT_RESOLUTION = 0.01
N_BACKGROUND = 1000
FEATURE_NAMES = ["elevation", "slope", "dist_to_water_km", "aspect_northness", "soil_favourability"]


# ── REAL ENVIRONMENTAL FEATURES FROM SRTM ────────────────────────────────────

SRTM_DIR = BASE + "/qgis/"

def load_srtm():
    """Load pre-derived SRTM + OSM water + BPEJ soil arrays."""
    elev_array   = np.load(SRTM_DIR + "srtm_elev.npy")
    slope_array  = np.load(SRTM_DIR + "srtm_slope.npy")
    aspect_array = np.load(SRTM_DIR + "srtm_aspect.npy")
    water_array  = np.load(SRTM_DIR + "srtm_water.npy")
    soil_array   = np.load(SRTM_DIR + "bpej_soil.npy")
    gt = tuple(np.load(SRTM_DIR + "srtm_gt.npy").tolist())
    print(f"  Elevation range: {elev_array.min():.0f} - {elev_array.max():.0f}m, shape: {elev_array.shape}")
    print(f"  Water distance range: {water_array.min():.1f} - {water_array.max():.1f}km, shape: {water_array.shape}")
    print(f"  Soil favourability range: {soil_array.min():.2f} - {soil_array.max():.2f}, shape: {soil_array.shape}")
    return elev_array, slope_array, aspect_array, water_array, soil_array, gt

def sample_raster(array, gt, lat, lon):
    """Sample a raster array at a given lat/lon."""
    col = int((lon - gt[0]) / gt[1])
    row = int((lat - gt[3]) / gt[5])
    if 0 <= row < array.shape[0] and 0 <= col < array.shape[1]:
        return float(array[row, col])
    return 0.0

def sample_grid(arr, lat_flat, lon_flat):
    """Generic vectorized sampler for 1000x500 layers (water, soil)."""
    lon_min, lat_max = 12.0, 52.0
    pixel_w = (22.0 - 12.0) / arr.shape[1]
    pixel_h = (52.0 - 47.0) / arr.shape[0]
    cols = np.clip(((lon_flat - lon_min) / pixel_w).astype(int), 0, arr.shape[1]-1)
    rows = np.clip(((lat_max - lat_flat) / pixel_h).astype(int), 0, arr.shape[0]-1)
    return arr[rows, cols]

def sample_grid_point(arr, lat, lon):
    """Single point sampler for 1000x500 layers."""
    lon_min, lat_max = 12.0, 52.0
    pixel_w = (22.0 - 12.0) / arr.shape[1]
    pixel_h = (52.0 - 47.0) / arr.shape[0]
    col = int(np.clip((lon - lon_min) / pixel_w, 0, arr.shape[1]-1))
    row = int(np.clip((lat_max - lat) / pixel_h, 0, arr.shape[0]-1))
    return float(arr[row, col])

def get_env_features_point(lat, lon, elev_arr, slope_arr, aspect_arr, water_arr, soil_arr, gt):
    elev   = sample_raster(elev_arr, gt, lat, lon)
    slope  = sample_raster(slope_arr, gt, lat, lon)
    aspect = sample_raster(aspect_arr, gt, lat, lon)
    dist_w = sample_grid_point(water_arr, lat, lon)
    soil   = sample_grid_point(soil_arr, lat, lon)
    return [elev, slope, dist_w, aspect, soil]

def get_env_features_grid(lat_flat, lon_flat, elev_arr, slope_arr, aspect_arr, water_arr, soil_arr, gt):
    """Vectorized raster sampling for the full prediction grid."""
    cols = ((lon_flat - gt[0]) / gt[1]).astype(int)
    rows = ((lat_flat - gt[3]) / gt[5]).astype(int)
    h, w = elev_arr.shape
    rows = np.clip(rows, 0, h-1)
    cols = np.clip(cols, 0, w-1)
    elev   = elev_arr[rows, cols]
    slope  = slope_arr[rows, cols]
    aspect = aspect_arr[rows, cols]
    dist_w = sample_grid(water_arr, lat_flat, lon_flat)
    soil   = sample_grid(soil_arr, lat_flat, lon_flat)
    return np.column_stack([elev, slope, dist_w, aspect, soil])


# ── MODEL ─────────────────────────────────────────────────────────────────────

def train_model(sites_gdf):
    print("Loading SRTM elevation data...")
    elev_arr, slope_arr, aspect_arr, water_arr, soil_arr, gt = load_srtm()
    print(f"  Elevation range: {elev_arr.min():.0f} - {elev_arr.max():.0f}m")

    print("Extracting features at site locations...")
    good_sites = sites_gdf[sites_gdf["geocode_confidence"].isin(["high", "medium"])].copy()
    print(f"  Using {len(good_sites)} high/medium confidence sites for training")

    if len(good_sites) < 10:
        print("  ⚠ Very few sites — model will be unreliable.")

    presence_features = [get_env_features_point(r.geometry.y, r.geometry.x,
                         elev_arr, slope_arr, aspect_arr, water_arr, soil_arr, gt)
                         for _, r in good_sites.iterrows()]

    print(f"Sampling {N_BACKGROUND} background points...")
    bg_lats = np.random.uniform(STUDY_BBOX["lat_min"], STUDY_BBOX["lat_max"], N_BACKGROUND)
    bg_lons = np.random.uniform(STUDY_BBOX["lon_min"], STUDY_BBOX["lon_max"], N_BACKGROUND)
    background_features = get_env_features_grid(bg_lats, bg_lons,
                          elev_arr, slope_arr, aspect_arr, water_arr, soil_arr, gt).tolist()

    X = np.array(presence_features + background_features)
    y = np.array([1]*len(presence_features) + [0]*len(background_features))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Training Random Forest classifier...")
    model = RandomForestClassifier(n_estimators=200, max_depth=6,
                                   min_samples_leaf=5, random_state=42, n_jobs=-1)

    # ── SPATIAL BLOCK CROSS-VALIDATION ───────────────────────────────────────
    # Divide study area into a 5x1 latitudinal strip grid (W→E across CE).
    # Each strip becomes one fold — ensuring test points are geographically
    # separated from training points, preventing spatial autocorrelation
    # from inflating AUC scores.
    print("  Running spatial block cross-validation (5 geographic blocks)...")
    N_BLOCKS = 5
    presence_lons = np.array([r.geometry.x for _, r in good_sites.iterrows()])

    lon_edges = np.linspace(STUDY_BBOX["lon_min"], STUDY_BBOX["lon_max"], N_BLOCKS + 1)
    block_labels = np.digitize(presence_lons, lon_edges[1:-1])  # 0..N_BLOCKS-1

    bg_block_labels = np.digitize(bg_lons, lon_edges[1:-1])

    all_labels = np.concatenate([block_labels, bg_block_labels])

    spatial_cv_scores = []
    for block in range(N_BLOCKS):
        test_mask  = all_labels == block
        train_mask = ~test_mask
        if test_mask.sum() < 2:
            continue
        X_tr, X_te = X_scaled[train_mask], X_scaled[test_mask]
        y_tr, y_te = y[train_mask], y[test_mask]
        if len(np.unique(y_te)) < 2:
            continue  # skip block with no positive examples
        m = RandomForestClassifier(n_estimators=200, max_depth=6,
                                   min_samples_leaf=5, random_state=42, n_jobs=-1)
        m.fit(X_tr, y_tr)
        from sklearn.metrics import roc_auc_score
        probs = m.predict_proba(X_te)[:, 1]
        spatial_cv_scores.append(roc_auc_score(y_te, probs))
        print(f"    Block {block+1}: AUC {spatial_cv_scores[-1]:.3f} "
              f"({test_mask.sum()} test points, {y_te.sum()} sites)")

    spatial_cv_scores = np.array(spatial_cv_scores)
    print(f"  Spatial CV AUC: {spatial_cv_scores.mean():.3f} ± {spatial_cv_scores.std():.3f}")
    print(f"  (vs random CV which would likely overestimate due to spatial autocorrelation)")

    if spatial_cv_scores.mean() < 0.65:
        print("  ⚠ Low spatial AUC — model may not generalise across geographic regions.")

    model.fit(X_scaled, y)
    importances = dict(zip(FEATURE_NAMES, model.feature_importances_))
    print(f"  Feature importances: {json.dumps({k: round(v,3) for k,v in sorted(importances.items(), key=lambda x: -x[1])})}")

    return model, scaler, elev_arr, slope_arr, aspect_arr, water_arr, soil_arr, gt


# ── HEATMAP ───────────────────────────────────────────────────────────────────

def generate_heatmap(model, scaler, elev_arr, slope_arr, aspect_arr, water_arr, soil_arr, gt, output_dir):
    output_dir = Path(output_dir)
    print("Generating probability raster...")

    lons = np.arange(STUDY_BBOX["lon_min"], STUDY_BBOX["lon_max"], OUTPUT_RESOLUTION)
    lats = np.arange(STUDY_BBOX["lat_max"], STUDY_BBOX["lat_min"], -OUTPUT_RESOLUTION)
    n_rows, n_cols = len(lats), len(lons)
    print(f"  Raster size: {n_cols} x {n_rows} pixels")

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    X_grid = get_env_features_grid(lat_grid.flatten(), lon_grid.flatten(),
                                   elev_arr, slope_arr, aspect_arr, water_arr, soil_arr, gt)
    X_scaled = scaler.transform(X_grid)

    print("  Running predictions...")
    BATCH = 50000
    probs = np.zeros(len(X_scaled), dtype=np.float32)
    for i in range(0, len(X_scaled), BATCH):
        probs[i:i+BATCH] = model.predict_proba(X_scaled[i:i+BATCH])[:, 1]
        print(f"  {min(100, int(i/len(X_scaled)*100))}%...", end="\r")
    print("  100% ✓         ")

    prob_grid = probs.reshape(n_rows, n_cols)
    sigma = max(1, int(2.0 / (OUTPUT_RESOLUTION * 111)))
    prob_grid = gaussian_filter(prob_grid, sigma=sigma)
    print(f"  Probability range: {prob_grid.min():.3f} - {prob_grid.max():.3f}")

    npy_path = output_dir / "probability_heatmap.npy"
    np.save(npy_path, prob_grid)

    meta = {
        "lon_min": STUDY_BBOX["lon_min"], "lon_max": STUDY_BBOX["lon_max"],
        "lat_min": STUDY_BBOX["lat_min"], "lat_max": STUDY_BBOX["lat_max"],
        "n_rows": n_rows, "n_cols": n_cols
    }
    with open(output_dir / "heatmap_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Saved: {npy_path}")
    print(f"""
Now run this in QGIS Python Editor (Plugins → Python Console → Show Editor):

import numpy as np, json
from osgeo import gdal, osr

with open('{output_dir}/heatmap_meta.json') as f:
    m = json.load(f)
grid = np.load('{npy_path}')
pixel_w = (m['lon_max'] - m['lon_min']) / m['n_cols']
pixel_h = (m['lat_max'] - m['lat_min']) / m['n_rows']
driver = gdal.GetDriverByName('GTiff')
ds = driver.Create('{output_dir}/probability_heatmap.tif', m['n_cols'], m['n_rows'], 1, gdal.GDT_Float32, ['COMPRESS=LZW'])
ds.SetGeoTransform([m['lon_min'], pixel_w, 0, m['lat_max'], 0, -pixel_h])
srs = osr.SpatialReference()
srs.ImportFromEPSG(4326)
ds.SetProjection(srs.ExportToWkt())
ds.GetRasterBand(1).WriteArray(grid)
ds.GetRasterBand(1).SetNoDataValue(-9999)
ds.FlushCache()
ds = None
print(f'Done — range: {{grid.min():.3f}} - {{grid.max():.3f}}')
""")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run_prediction(sites_gpkg, output_dir):
    sites_gdf = gpd.read_file(sites_gpkg)
    model, scaler, elev_arr, slope_arr, aspect_arr, water_arr, soil_arr, gt = train_model(sites_gdf)
    generate_heatmap(model, scaler, elev_arr, slope_arr, aspect_arr, water_arr, soil_arr, gt, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", default=DEFAULT_SITES)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_prediction(args.sites, args.output_dir)
