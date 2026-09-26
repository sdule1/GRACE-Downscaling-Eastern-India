import xarray as xr
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import rioxarray
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

# --- CONFIGURATION & FONT ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5

folder = '/content/drive/MyDrive/CSR_GRACE_RL0603/Masked'
grace_path = os.path.join(folder, 'GRACE_StudyArea_Subset_ML_Filled_2003_23.nc')
ppt_path = os.path.join(folder, 'TerraClimate_Precip_2003_2023.nc')
temp_path = os.path.join(folder, 'TerraClimate_Temp_2003_2023.nc')
soil_path = os.path.join(folder, 'TerraClimate_Soil_2003_2023.nc')
aet_path = os.path.join(folder, 'TerraClimate_AET_2003_2023.nc')
q_path = os.path.join(folder, 'TerraClimate_Runoff_2003_2023.nc')
pdsi_path = os.path.join(folder, 'TerraClimate_PDSI_2003_2023.nc')
dem_path = os.path.join(folder, 'SRTM_DEM_1km_Resampled.tif')
slope_path = os.path.join(folder, 'Slope.tif')
aspect_path = os.path.join(folder, 'Aspect.tif')
shapefile_path = os.path.join(folder, 'StudyArea.shp')

# Helper functions
def load_and_fix_coords(path, name):
    da = rioxarray.open_rasterio(path).squeeze().drop_vars('band', errors='ignore')
    if 'x' in da.coords and 'y' in da.coords:
        da = da.rename({'x': 'lon', 'y': 'lat'})
    da.name = name
    return da

def clip_to_roi(da, shp_path):
    if not os.path.exists(shp_path): return da
    try:
        gdf = gpd.read_file(shp_path)
        if da.rio.crs is None: da.rio.write_crs("EPSG:4326", inplace=True)
        if gdf.crs != da.rio.crs: gdf = gdf.to_crs(da.rio.crs)
        if 'lon' in da.dims and 'lat' in da.dims:
            da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
        return da.rio.clip(gdf.geometry, gdf.crs, drop=False, invert=False)
    except: return da

print("Loading datasets for SPATIAL BLOCK model comparison...")
ds_grace = xr.open_dataset(grace_path).sel(time=slice('2003-01-01', '2023-12-31')).resample(time='MS').mean()
ds_ppt = xr.open_dataset(ppt_path).sel(time=slice('2003-01-01', '2023-12-31')).resample(time='MS').mean()
ds_temp = xr.open_dataset(temp_path).sel(time=slice('2003-01-01', '2023-12-31')).resample(time='MS').mean()
ds_soil = xr.open_dataset(soil_path).sel(time=slice('2003-01-01', '2023-12-31')).resample(time='MS').mean()
ds_aet = xr.open_dataset(aet_path).sel(time=slice('2003-01-01', '2023-12-31')).resample(time='MS').mean()
ds_q = xr.open_dataset(q_path).sel(time=slice('2003-01-01', '2023-12-31')).resample(time='MS').mean()
ds_pdsi = xr.open_dataset(pdsi_path).sel(time=slice('2003-01-01', '2023-12-31')).resample(time='MS').mean()

common_times = np.intersect1d(ds_grace.time, ds_ppt.time)
ds_grace, ds_ppt, ds_temp, ds_soil, ds_aet, ds_q, ds_pdsi = [
    ds.sel(time=common_times) for ds in [ds_grace, ds_ppt, ds_temp, ds_soil, ds_aet, ds_q, ds_pdsi]
]

da_dem = clip_to_roi(load_and_fix_coords(dem_path, 'DEM'), shapefile_path)
da_slope = clip_to_roi(load_and_fix_coords(slope_path, 'SLOPE'), shapefile_path)
da_aspect = clip_to_roi(load_and_fix_coords(aspect_path, 'ASPECT'), shapefile_path)

da_dem_coarse, da_slope_coarse, da_aspect_coarse = [
    da.interp_like(ds_grace, method='linear') for da in [da_dem, da_slope, da_aspect]
]
ds_ppt_coarse = ds_ppt['ppt'].interp_like(ds_grace, method='linear')
ds_temp_coarse = ds_temp['tmax'].interp_like(ds_grace, method='linear')
ds_soil_coarse = ds_soil['soil'].interp_like(ds_grace, method='linear')
ds_aet_coarse = ds_aet['aet'].interp_like(ds_grace, method='linear')
ds_q_coarse = ds_q['q'].interp_like(ds_grace, method='linear')
ds_pdsi_coarse = ds_pdsi['PDSI'].interp_like(ds_grace, method='linear')

# --- EXTRACTING SPATIAL GRIDS FOR BLOCKING ---
lons = ds_grace.lon.values
lats = ds_grace.lat.values
lon_grid, lat_grid = np.meshgrid(lons, lats)
num_times = len(ds_grace.time)
n_pixels_spatial = len(lats) * len(lons)

lon_flat = np.tile(lon_grid.flatten(), num_times)
lat_flat = np.tile(lat_grid.flatten(), num_times)
time_sin_flat = np.repeat(np.sin(2 * np.pi * ds_grace.time.dt.month / 12).values, n_pixels_spatial)
time_cos_flat = np.repeat(np.cos(2 * np.pi * ds_grace.time.dt.month / 12).values, n_pixels_spatial)

print("Constructing training DataFrame...")
df_train = pd.DataFrame({
    'LAT': lat_flat,
    'LON': lon_flat,
    'TWS': ds_grace['lwe_thickness'].values.flatten(),
    'PPT': ds_ppt_coarse.values.flatten(),
    'TEMP': ds_temp_coarse.values.flatten(),
    'SOIL': ds_soil_coarse.values.flatten(),
    'AET': ds_aet_coarse.values.flatten(),
    'Q': ds_q_coarse.values.flatten(),
    'PDSI': ds_pdsi_coarse.values.flatten(),
    'SIN_M': time_sin_flat,
    'COS_M': time_cos_flat,
    'DEM': np.tile(da_dem_coarse.values.flatten(), num_times),
    'SLOPE': np.tile(da_slope_coarse.values.flatten(), num_times),
    'ASPECT': np.tile(da_aspect_coarse.values.flatten(), num_times)
}).dropna()

# --- 1. SETUP SPATIAL BLOCK SPLIT ---
print("Creating ~1 Degree Spatial Blocks...")
df_train['BLOCK'] = np.floor(df_train['LAT']).astype(int).astype(str) + "_" + np.floor(df_train['LON']).astype(int).astype(str)
unique_blocks = df_train['BLOCK'].unique()
print(f"Total Unique Spatial Blocks Identified: {len(unique_blocks)}")

np.random.seed(42)
test_blocks = np.random.choice(unique_blocks, size=int(len(unique_blocks) * 0.20), replace=False)

train_mask = ~df_train['BLOCK'].isin(test_blocks)
test_mask = df_train['BLOCK'].isin(test_blocks)

features = ['PPT', 'TEMP', 'SOIL', 'AET', 'Q', 'PDSI', 'SIN_M', 'COS_M', 'DEM', 'SLOPE', 'ASPECT']
target = 'TWS'
X_train, y_train = df_train[train_mask][features], df_train[train_mask][target]
X_test, y_test = df_train[test_mask][features], df_train[test_mask][target]
print(f"Spatial Holdout Split -> Training Samples: {len(X_train)} | Testing Samples: {len(X_test)}")

# --- 2. DEFINE TUNED MODELS ---
models = {
    "MLR": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_split=5, n_jobs=-1, random_state=42),
    "XGBoost": xgb.XGBRegressor(n_estimators=200, learning_rate=0.03, max_depth=7, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, tree_method='hist', random_state=42)
}

# --- 3. TRAINING & EVALUATION LOOP ---
results = []
for name, model in models.items():
    print(f"Analyzing {name}...")
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    r2_train = r2_score(y_train, train_pred)
    r2_test = r2_score(y_test, test_pred)
    gap = r2_train - r2_test

    rmse_test = sqrt(mean_squared_error(y_test, test_pred))
    nse_test = 1 - (np.sum((y_test - test_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

    results.append({
        "Model": name,
        "Train R2": r2_train,
        "Test R2": r2_test,
        "Test RMSE": rmse_test,
        "Test NSE": nse_test,
        "Gap": gap
    })

df_results = pd.DataFrame(results)
print("\n--- SPATIAL BLOCK BENCHMARK RESULTS ---")
print(df_results.to_string(index=False))

# --- 4. PLOTTING ---
fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
x = np.arange(len(df_results))
width = 0.35

color_train = '#B0B0B0'
color_test = '#2C7BB6'
rects1 = ax.bar(x - width/2, df_results["Train R2"], width, label='Training Score', color=color_train, edgecolor='black', linewidth=1)
rects2 = ax.bar(x + width/2, df_results["Test R2"], width, label='Testing Score (Unseen Locations)', color=color_test, edgecolor='black', linewidth=1)

ax.set_ylabel('R-Squared ($R^2$) Score', fontsize=14, fontweight='bold', labelpad=10)
ax.set_title('Model Generalization on Spatial Block Cross-Validation', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(df_results["Model"], fontsize=14)
ax.tick_params(axis='y', labelsize=12)
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.3)
max_val = max(df_results["Train R2"].max(), df_results["Test R2"].max())
ax.set_ylim(0.0, max_val * 1.30)
ax.legend(loc='upper left', fontsize=12, frameon=True, shadow=True)

def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')

add_labels(rects1)
add_labels(rects2)

for i in range(len(df_results)):
    gap_val = df_results.loc[i, "Gap"]
    rmse_val = df_results.loc[i, "Test RMSE"]
    nse_val = df_results.loc[i, "Test NSE"]
    gap_color = '#D7191C' if gap_val > 0.15 else '#1A9641'
    
    label_text = f"Gap: {gap_val:.3f}\nRMSE: {rmse_val:.2f}\nNSE: {nse_val:.2f}"
    y_pos = max(df_results.loc[i, "Train R2"], df_results.loc[i, "Test R2"]) + 0.08
    ax.text(i, y_pos, label_text, ha='center', va='bottom',
            fontsize=12, fontweight='bold', color=gap_color,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

plt.tight_layout()
save_path = os.path.join(folder, 'Spatial_Holdout_Comparison.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"High-Res Plot Saved to: {save_path}")
plt.show()