import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import rioxarray

try:
    import geopandas as gpd
except:
    pass

# --- CONFIGURATION ---
folder = '/content/drive/MyDrive/CSR_GRACE_RL0603/Masked'
grace_path = os.path.join(folder, 'GRACE_StudyArea_Subset_ML_Filled_2003_23.nc')
ml_only_path = os.path.join(folder, 'GRACE_Downscaled_TWSA_1km_ML_ONLY_RF.nc')
shapefile_path = os.path.join(folder, 'StudyArea.shp')

output_fig = os.path.join(folder, 'Fig8_Model_Physical_Error_RF.png')
output_tif = os.path.join(folder, 'GRACE_ML_Error_1km_RF.tif')

print("Loading Data...")
ds_grace = xr.open_dataset(grace_path)
ds_ml = xr.open_dataset(ml_only_path)

common_time = np.intersect1d(ds_grace.time, ds_ml.time)
ds_grace = ds_grace.sel(time=common_time)
ds_ml = ds_ml.sel(time=common_time)

if 'lwe_thickness_predicted' in ds_ml:
    ds_ml = ds_ml.rename({'lwe_thickness_predicted': 'lwe_thickness'})

# --- CALCULATION ---
print("Calculating ML Residuals (Pre-Correction Error)...")
pred_coarse = ds_ml['lwe_thickness'].interp_like(ds_grace['lwe_thickness'], method='linear')
abs_error = np.abs(ds_grace['lwe_thickness'] - pred_coarse)
mae_map = abs_error.mean(dim='time')

mae_1km = mae_map.interp_like(ds_ml['lwe_thickness'], method='linear')
mae_1km.name = 'mae_cm'

# --- SAVE ---
try:
    mae_1km.rio.write_crs("EPSG:4326", inplace=True)
    mae_1km.rio.to_raster(output_tif, compress='LZW')
    print(f"Saved GeoTIFF: {output_tif}")
except:
    pass

# --- VISUALIZATION ---
print("Generating Figure...")
plt.figure(figsize=(14, 12))

if os.path.exists(shapefile_path):
    try:
        gdf = gpd.read_file(shapefile_path)
        if gdf.crs != "EPSG:4326": gdf = gdf.to_crs("EPSG:4326")
        mae_1km = mae_1km.rio.clip(gdf.geometry, gdf.crs, drop=True)
    except:
        pass

im = mae_1km.plot(
    cmap='inferno',
    vmin=0, vmax=8,
    add_colorbar=False
)

cbar = plt.colorbar(im, shrink=0.8)
cbar.set_label('Model Discrepancy / MAE (cm)', fontsize=18, fontweight='bold')
cbar.ax.tick_params(labelsize=16)

plt.title('Model Physical Error Map\n(Where Climate/Terrain fails to explain GRACE)', fontsize=24, fontweight='bold', pad=20)
plt.xlabel('Longitude', fontsize=18, fontweight='bold')
plt.ylabel('Latitude', fontsize=18, fontweight='bold')

plt.tick_params(axis='both', which='major', labelsize=16, direction='in', length=6, width=1.5)
ax = plt.gca() 
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(2)
    spine.set_color('black')

if os.path.exists(shapefile_path):
    try:
        gdf.boundary.plot(ax=ax, color='black', linewidth=1, zorder=10)
    except: pass

plt.tight_layout()
plt.savefig(output_fig, dpi=300)
print(f"SUCCESS! Figure saved to: {output_fig}")
print(f"Mean Error: {mae_1km.mean().values:.2f} cm")
plt.show()