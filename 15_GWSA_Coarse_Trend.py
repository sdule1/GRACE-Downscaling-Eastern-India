import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import rioxarray

try:
    import geopandas as gpd
except ImportError:
    gpd = None

# --- CONFIGURATION ---
folder = '/content/drive/MyDrive/CSR_GRACE_RL0603/Masked'
input_nc = os.path.join(folder, 'GRACE_Coarse_GWSA_0.25deg_2003_2023_MASKED.nc')
shapefile_path = os.path.join(folder, 'StudyArea.shp') 

output_tif = os.path.join(folder, 'Coarse_GW_Trend_Slope_0.25deg_2003_2023_MASKED.tif')
output_png = os.path.join(folder, 'Fig12_Coarse_GW_Trend_Map_MASKED.png')
plt.rcParams['font.family'] = 'sans-serif'

# --- LOAD DATA ---
print("Loading Coarse GRACE Data...")
if not os.path.exists(input_nc):
    print(f"ERROR: Input file not found: {input_nc}")
    exit()
try:
    ds = xr.open_dataset(input_nc)
    exclude_vars = ['crs', 'spatial_ref', 'time_bnds', 'lon_bnds', 'lat_bnds']
    var_name = [v for v in ds.data_vars if v not in exclude_vars][0]
    print(f"Variable found: {var_name}")
except Exception as e:
    print(f"Error loading NetCDF: {e}")
    exit()

# --- PREPARE TIME AXIS ---
print("Converting time to decimal years...")
def to_decimal_year(time_index):
    return time_index.year + (time_index.dayofyear - 1) / 365.25
decimal_years = [to_decimal_year(t) for t in pd.to_datetime(ds.time.values)]
ds = ds.assign_coords(year_frac=('time', decimal_years))
ds = ds.swap_dims({'time': 'year_frac'})

# --- CALCULATE TREND ---
print("Calculating Pixel-wise Linear Trends (cm/year)...")
trend_fit = ds[var_name].polyfit(dim='year_frac', deg=1, skipna=True)
slope_map = trend_fit.polyfit_coefficients.sel(degree=1)

# --- CALCULATE & PRINT STATISTICS ---
min_val = slope_map.min().item()
max_val = slope_map.max().item()
mean_val = slope_map.mean().item()

print("\n" + "="*40)
print("  COARSE GROUNDWATER TREND STATISTICS")
print("="*40)
print(f"Max Depletion Rate (Min): {min_val:.4f} cm/year")
print(f"Max Recharge Rate (Max):  +{max_val:.4f} cm/year")
print(f"Regional Mean Trend:      {mean_val:.4f} cm/year")
print("="*40 + "\n")

# --- SAVE GEOTIFF ---
print(f"Saving GeoTIFF to {output_tif}...")
if 'degree' in slope_map.coords: slope_map = slope_map.drop_vars('degree')
try:
    if slope_map.rio.crs is None:
        slope_map.rio.write_crs("EPSG:4326", inplace=True)
    if 'lon' in slope_map.dims and 'lat' in slope_map.dims:
        slope_map.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    slope_map.rio.to_raster(output_tif)
except Exception as e:
    print(f"Warning: GeoTIFF save failed ({e}). Proceeding to plot.")

# --- PLOTTING ---
print(f"Generating Plot to {output_png}...")
fig, ax = plt.subplots(figsize=(14, 12))
print(f"Dynamic Color Scale set to: {min_val:.3f} to {max_val:.3f} cm/yr")

im = slope_map.plot(
    ax=ax,
    cmap='RdYlBu',
    add_colorbar=False,
    vmin=min_val,
    vmax=max_val
)

if gpd is not None and os.path.exists(shapefile_path):
    try:
        gdf = gpd.read_file(shapefile_path)
        if gdf.crs != "EPSG:4326": gdf = gdf.to_crs("EPSG:4326")
        gdf.boundary.plot(ax=ax, color='black', linewidth=1, zorder=10)
    except: pass

for spine in ax.spines.values():
    spine.set_linewidth(2.5)
    spine.set_color('black')

ax.tick_params(axis='both', which='major', direction='in', length=8, width=2, labelsize=16)
ax.tick_params(axis='both', which='minor', direction='in', length=4, width=1.5)

cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.85, pad=0.03)
cbar.set_label('Coarse GWS Trend (cm/year)', fontsize=18, fontweight='bold', labelpad=16)
cbar.ax.tick_params(labelsize=16, direction='in', width=1.5)
cbar.outline.set_linewidth(1.5)

ticks = np.linspace(min_val, max_val, 5)
cbar.set_ticks(ticks)
cbar.ax.set_yticklabels([f"{x:.2f}" for x in ticks])

plt.title('Coarse-Resolution GWS Trends\n(2003–2023)', fontsize=26, fontweight='bold', pad=20)
plt.xlabel('Longitude', fontsize=18, fontweight='bold')
plt.ylabel('Latitude', fontsize=18, fontweight='bold')
plt.grid(False)

plt.tight_layout()
plt.savefig(output_png, dpi=600, bbox_inches='tight')
print("SUCCESS!")
plt.show()