import xarray as xr
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import rioxarray
import gc
import shutil
import pymannkendall as mk
from math import sqrt
from scipy.stats import pearsonr, t
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    import shap
except ImportError:
    print("WARNING: 'shap' library not found. Run '!pip install shap'")

try:
    import geopandas as gpd
except ImportError:
    print("ERROR: Geopandas not found. Run '!pip install geopandas'")
    raise

# --- 1. CONFIGURATION ---
folder = '/content/drive/MyDrive/CSR_GRACE_RL0603/Masked'
chunk_folder = os.path.join(folder, 'temp_yearly_chunks')

# Input Files
grace_path = os.path.join(folder, 'GRACE_StudyArea_Subset_ML_Filled_2003_23.nc')
ppt_path = os.path.join(folder, 'TerraClimate_Precip_2003_2023.nc')
temp_path = os.path.join(folder, 'TerraClimate_Temp_2003_2023.nc')
soil_path = os.path.join(folder, 'TerraClimate_Soil_2003_2023.nc')
aet_path = os.path.join(folder, 'TerraClimate_AET_2003_2023.nc')
q_path = os.path.join(folder, 'TerraClimate_Runoff_2003_2023.nc')
pdsi_path = os.path.join(folder, 'TerraClimate_PDSI_2003_2023.nc')

# Terrain & Shapefile
dem_path = os.path.join(folder, 'SRTM_DEM_1km_Resampled.tif')
slope_path = os.path.join(folder, 'Slope.tif')
aspect_path = os.path.join(folder, 'Aspect.tif')
shapefile_path = os.path.join(folder, 'StudyArea.shp')

# Outputs
output_path_final_twsa = os.path.join(folder, 'GRACE_Downscaled_TWSA_1km_RF_FINAL.nc')
output_path_ml_only = os.path.join(folder, 'GRACE_Downscaled_TWSA_1km_ML_ONLY_RF.nc')
output_path_trends = os.path.join(folder, 'GRACE_Downscaled_TWSA_1km_MK_Trends.nc')
metrics_path = os.path.join(folder, 'Model_Metrics_RF_Final.txt')
importance_path = os.path.join(folder, 'Feature_Importance_Standard_RF.png')
shap_summary_path = os.path.join(folder, 'SHAP_Summary_Beeswarm_RF.png')
vif_path = os.path.join(folder, 'VIF_Diagnostics.csv')
corr_path = os.path.join(folder, 'Predictor_Correlation_Matrix.png')

if os.path.exists(chunk_folder): shutil.rmtree(chunk_folder)
os.makedirs(chunk_folder)

# --- HELPER FUNCTIONS ---
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
    except Exception as e:
        print(f"Clipping failed: {e}")
        return da

def effective_n_significance(obs, pred):
    r, _ = pearsonr(obs, pred)
    r1_obs = pd.Series(obs).autocorr(lag=1)
    r1_pred = pd.Series(pred).autocorr(lag=1)
    r1 = max(r1_obs, r1_pred) if not np.isnan(r1_obs) and not np.isnan(r1_pred) else 0
    n = len(obs)
    n_eff = n * ((1 - r1) / (1 + r1)) if r1 != -1 else n
    t_stat = r * np.sqrt((n_eff - 2) / (1 - r**2))
    p_val = 2 * (1 - t.cdf(abs(t_stat), df=n_eff - 2))
    return r, n_eff, p_val

@np.vectorize(signature='(n)->(),()')
def calc_mk_trend_vectorized(ts):
    valid = ts[~np.isnan(ts)]
    if len(valid) < 10: return np.nan, np.nan
    try:
        res = mk.hamed_rao_modification_test(valid)
        return res.slope, res.p
    except:
        return np.nan, np.nan

# --- 2. LOAD & ALIGN ---
print("Loading datasets...")
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

print("Loading Terrain...")
da_dem = clip_to_roi(load_and_fix_coords(dem_path, 'DEM'), shapefile_path)
da_slope = clip_to_roi(load_and_fix_coords(slope_path, 'SLOPE'), shapefile_path)
da_aspect = clip_to_roi(load_and_fix_coords(aspect_path, 'ASPECT'), shapefile_path)

# --- 3. TRAINING PREP ---
print("Preparing Training Data...")
da_dem_coarse, da_slope_coarse, da_aspect_coarse = [
    da.interp_like(ds_grace, method='linear') for da in [da_dem, da_slope, da_aspect]
]
ds_ppt_coarse = ds_ppt['ppt'].interp_like(ds_grace, method='linear')
ds_temp_coarse = ds_temp['tmax'].interp_like(ds_grace, method='linear')
ds_soil_coarse = ds_soil['soil'].interp_like(ds_grace, method='linear')
ds_aet_coarse = ds_aet['aet'].interp_like(ds_grace, method='linear')
ds_q_coarse = ds_q['q'].interp_like(ds_grace, method='linear')
ds_pdsi_coarse = ds_pdsi['PDSI'].interp_like(ds_grace, method='linear')

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

# --- SPATIAL BLOCK CROSS-VALIDATION SPLIT ---
print("\n--- Splitting Data: Spatial Block Cross-Validation (Train 80% | Test 20% Blocks) ---")
df_train['BLOCK'] = np.floor(df_train['LAT']).astype(int).astype(str) + "_" + np.floor(df_train['LON']).astype(int).astype(str)
unique_blocks = df_train['BLOCK'].unique()

np.random.seed(42)
test_blocks = np.random.choice(unique_blocks, size=int(len(unique_blocks) * 0.20), replace=False)
train_mask = ~df_train['BLOCK'].isin(test_blocks)
test_mask = df_train['BLOCK'].isin(test_blocks)

features = ['PPT', 'TEMP', 'SOIL', 'AET', 'Q', 'PDSI', 'SIN_M', 'COS_M', 'DEM', 'SLOPE', 'ASPECT']
target = 'TWS'

X_train, y_train = df_train[train_mask][features], df_train[train_mask][target]
X_test, y_test = df_train[test_mask][features], df_train[test_mask][target]
print(f"Total Spatial Blocks: {len(unique_blocks)} | Withheld for Testing: {len(test_blocks)}")
print(f"Training Samples: {len(X_train)} | Testing Samples: {len(X_test)}")

# ==========================================
# --- MULTICOLLINEARITY DIAGNOSTICS ---
# ==========================================
print("\n--- Running Multicollinearity Diagnostics ---")
plt.figure(figsize=(10, 8))
corr = X_train.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title("Predictor Correlation Matrix")
plt.tight_layout()
plt.savefig(corr_path, dpi=300)
plt.close()
print(f"Correlation Matrix saved to: {corr_path}")

vif_data = pd.DataFrame()
vif_data["Feature"] = X_train.columns
X_vif = np.column_stack([np.ones(X_train.shape[0]), X_train.values])
vif_data["VIF"] = [variance_inflation_factor(X_vif, i) for i in range(1, X_vif.shape[1])]
vif_data.to_csv(vif_path, index=False)
print(f"VIF Diagnostics saved to: {vif_path}")

# ==========================================
# --- 4. MODEL EVALUATION & EXPLAINABLE AI ---
print("\n--- Calculating Model Evaluation Metrics ---")
model_val = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_split=5, n_jobs=-1, random_state=42)
model_val.fit(X_train, y_train)

y_pred = model_val.predict(X_test)
rmse = sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
nse = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
r_val, n_eff, p_adj = effective_n_significance(y_test.values, y_pred)

print(f"RF Test RMSE: {rmse:.4f} cm | RF Test R2: {r2:.4f} | RF Test NSE: {nse:.4f}")
print(f"Test Set Correlation (r): {r_val:.4f} | Effective N: {n_eff:.1f} | Adj p-value: {p_adj:.4e}")

with open(metrics_path, "w") as f:
    f.write("--- Random Forest Evaluation (Spatial Block Holdout 20%) ---\n")
    f.write(f"RMSE: {rmse:.4f} cm\nMAE:  {mae:.4f} cm\nR2:   {r2:.4f}\nNSE:  {nse:.4f}\n")
    f.write(f"Effective N Correction -> r: {r_val:.4f}, N_eff: {n_eff:.1f}, p-value: {p_adj:.4e}\n")

print("\nRetraining Full RF Model...")
model = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_split=5, n_jobs=-1, random_state=42)
model.fit(df_train[features], df_train[target])

print("\n--- Running SHAP Analysis ---")
try:
    X_shap_sample = X_train.sample(n=min(5000, len(X_train)), random_state=42)
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    shap_values = explainer.shap_values(X_shap_sample, check_additivity=False, approximate=True)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_shap_sample, show=False)
    plt.title("SHAP Summary (Impact on Groundwater Storage)", fontsize=14)
    plt.tight_layout()
    plt.savefig(shap_summary_path, dpi=300, bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"SHAP Analysis Failed: {e}")

del df_train, ds_ppt_coarse, ds_temp_coarse, ds_soil_coarse, ds_aet_coarse, ds_q_coarse, ds_pdsi_coarse
del X_train, X_test, y_train, y_test
gc.collect()

# --- 5. BATCH PREDICTION & MASS CONSERVATION ---
print("\nStarting Yearly Batch Prediction...")
dem_flat = da_dem.values.flatten()
valid_idx = ~np.isnan(dem_flat)
n_valid = np.sum(valid_idx)
target_template = da_dem

yearly_maps, yearly_times, current_year = [], [], None

for t_idx, t in enumerate(ds_grace.time):
    date_ts = pd.to_datetime(t.values)
    year, month = date_ts.year, date_ts.month
    if year != current_year and current_year is not None:
        da_year = xr.DataArray(np.array(yearly_maps), coords={'time': pd.to_datetime(yearly_times), 'lat': da_dem.lat, 'lon': da_dem.lon}, dims=('time', 'lat', 'lon'), name='lwe_thickness')
        da_year.to_netcdf(os.path.join(chunk_folder, f"grace_1km_{current_year}.nc"), encoding={'lwe_thickness': {'zlib': True, 'complevel': 5}})
        yearly_maps, yearly_times = [], []
        gc.collect()
    if current_year != year: current_year = year

    sin_valid = np.full(n_valid, np.sin(2 * np.pi * month / 12))
    cos_valid = np.full(n_valid, np.cos(2 * np.pi * month / 12))

    ppt_1km = ds_ppt['ppt'].sel(time=t, method='nearest').interp_like(target_template).values.flatten()[valid_idx]
    temp_1km = ds_temp['tmax'].sel(time=t, method='nearest').interp_like(target_template).values.flatten()[valid_idx]
    soil_1km = ds_soil['soil'].sel(time=t, method='nearest').interp_like(target_template).values.flatten()[valid_idx]
    aet_1km = ds_aet['aet'].sel(time=t, method='nearest').interp_like(target_template).values.flatten()[valid_idx]
    q_1km = ds_q['q'].sel(time=t, method='nearest').interp_like(target_template).values.flatten()[valid_idx]
    pdsi_1km = ds_pdsi['PDSI'].sel(time=t, method='nearest').interp_like(target_template).values.flatten()[valid_idx]

    X_input = pd.DataFrame(np.column_stack((ppt_1km, temp_1km, soil_1km, aet_1km, q_1km, pdsi_1km, sin_valid, cos_valid, dem_flat[valid_idx], da_slope.values.flatten()[valid_idx], da_aspect.values.flatten()[valid_idx])), columns=features)
    y_pred = model.predict(X_input)

    full_map = np.full(dem_flat.shape, np.nan)
    full_map[valid_idx] = y_pred
    yearly_maps.append(full_map.reshape(da_dem.shape))
    yearly_times.append(date_ts)

if yearly_maps:
    da_year = xr.DataArray(np.array(yearly_maps), coords={'time': pd.to_datetime(yearly_times), 'lat': da_dem.lat, 'lon': da_dem.lon}, dims=('time', 'lat', 'lon'), name='lwe_thickness')
    da_year.to_netcdf(os.path.join(chunk_folder, f"grace_1km_{current_year}.nc"), encoding={'lwe_thickness': {'zlib': True, 'complevel': 5}})

print("Applying Mass Conservation to TWSA...")
ds_all = xr.open_mfdataset(os.path.join(chunk_folder, "*.nc"), combine='by_coords')
da_pred_all = ds_all['lwe_thickness']
da_pred_all.rio.write_crs("EPSG:4326", inplace=True)
da_pred_all.to_netcdf(output_path_ml_only)

ds_grace_aligned = ds_grace.sel(time=da_pred_all.time).compute()
pred_coarse = da_pred_all.interp_like(ds_grace_aligned, method='linear')
residual = ds_grace_aligned['lwe_thickness'] - pred_coarse

da_final_twsa = da_pred_all + residual.interp_like(da_pred_all, method='linear')
da_final_twsa.name = 'TWSA'
da_final_twsa.to_netcdf(output_path_final_twsa, encoding={'TWSA': {'zlib': True, 'complevel': 5}})

if os.path.exists(chunk_folder): shutil.rmtree(chunk_folder)

# --- 6. PRE-WHITENED MANN-KENDALL TREND CALCULATION ---
print("\n--- Calculating Robust Pre-Whitened Mann-Kendall Trends (This may take a few minutes) ---")
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    slope_arr, pval_arr = xr.apply_ufunc(
        calc_mk_trend_vectorized,
        da_final_twsa,
        input_core_dims=[['time']],
        output_core_dims=[[], []],
        dask='allowed'
    )

ds_trends = xr.Dataset({
    'trend_slope': slope_arr,
    'p_value': pval_arr
})
ds_trends.rio.write_crs("EPSG:4326", inplace=True)
ds_trends.to_netcdf(output_path_trends)
print(f"SUCCESS! Robust Trend Maps saved to: {output_path_trends}")