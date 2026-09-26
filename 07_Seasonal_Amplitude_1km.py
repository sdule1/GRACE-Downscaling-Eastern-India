import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
import rioxarray

try:
    import geopandas as gpd
except ImportError:
    gpd = None

# --- CONFIGURATION ---
folder = '/content/drive/MyDrive/CSR_GRACE_RL0603/Masked'
gws_path = os.path.join(folder, 'GRACE_Downscaled_GWSA_1km_RF_FINAL.nc')
shapefile_path = os.path.join(folder, 'StudyArea.shp')

output_tif = os.path.join(folder, 'GWS_Seasonal_Amplitude_1km_RF.tif')
fig_amplitude = os.path.join(folder, 'Fig8_Seasonal_Amplitude_HighRes_RF.png')

FONT_FAMILY = 'serif'
FONT_NAME = 'Times New Roman'
plt.rcParams['font.family'] = FONT_FAMILY
plt.rcParams['font.serif'] = [FONT_NAME, 'Times', 'DejaVu Serif', 'serif']

print("Loading 1km GWSA Data...")
if not os.path.exists(gws_path):
    print("ERROR: GWSA input file not found.")
    exit()

da_gws = xr.open_dataset(gws_path)['GWSA']

# --- 1. SEASONAL AMPLITUDE ANALYSIS ---
print("Calculating Seasonal Amplitude...")
monthly_climatology = da_gws.groupby('time.month').mean('time')
amplitude = (monthly_climatology.max(dim='month') - monthly_climatology.min(dim='month')) / 2
amplitude.name = 'seasonal_amplitude_cm'

if amplitude.rio.crs is None:
    amplitude.rio.write_crs("EPSG:4326", inplace=True)

if gpd is not None and os.path.exists(shapefile_path):
    print("Clipping data to Study Area extent...")
    try:
        gdf = gpd.read_file(shapefile_path)
        if gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        amplitude = amplitude.rio.clip(gdf.geometry.values, gdf.crs, all_touched=True)
    except Exception as e:
        print(f"Warning: Clipping failed ({e}).")

# --- 2. SAVE AMPLITUDE AS GEOTIFF ---
print(f"Saving Amplitude GeoTIFF to: {output_tif}")
try:
    amplitude.rio.to_raster(output_tif, compress='LZW')
except Exception as e:
    print(f"Error saving GeoTIFF: {e}")

# --- 3. VISUALIZATION ---
print("Generating High-Res Map...")
fig, ax = plt.subplots(figsize=(14, 12))

min_amp = amplitude.min().item()
max_amp = amplitude.max().item()

im = amplitude.plot(
    ax=ax,
    cmap='turbo',
    add_colorbar=False,
    vmin=min_amp,
    vmax=max_amp
)

if gpd is not None and os.path.exists(shapefile_path):
    try:
        gdf.boundary.plot(ax=ax, color='black', linewidth=1, zorder=10)
    except Exception as e:
        pass

ax.set_title('Groundwater Seasonal Oscillation Strength\n(Recharge Potential)', fontsize=24, fontweight='bold', pad=20)
ax.set_xlabel('Longitude', fontsize=18, fontweight='bold')
ax.set_ylabel('Latitude', fontsize=18, fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=16, direction='in', length=8, width=2)

for spine in ax.spines.values():
    spine.set_linewidth(2)
    spine.set_color('black')

cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.85, pad=0.03)
cbar.set_label('GWS Seasonal Amplitude (cm)', fontsize=18, fontweight='bold', labelpad=15)
cbar.ax.tick_params(labelsize=16, length=6, width=1.5)
cbar.outline.set_linewidth(1.5)

ticks = np.linspace(min_amp, max_amp, 5)
cbar.set_ticks(ticks)
cbar.ax.set_yticklabels([f"{x:.2f}" for x in ticks])

plt.tight_layout()
plt.savefig(fig_amplitude, dpi=600, bbox_inches='tight')
print(f"SUCCESS! Figure saved to: {fig_amplitude}")
plt.show()