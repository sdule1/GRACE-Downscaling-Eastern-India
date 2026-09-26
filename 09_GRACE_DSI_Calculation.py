import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIGURATION ---
folder = '/content/drive/MyDrive/CSR_GRACE_RL0603/Masked'
gws_path = os.path.join(folder, 'GRACE_Downscaled_GWSA_1km_RF_FINAL_MASKED.nc')

dsi_path = os.path.join(folder, 'GRACE_DSI_GLDAS_Updated_1km_2003_2023.nc')
fig_dsi_map = os.path.join(folder, 'Fig6_Drought_Severity_Map_GLDAS.png')
fig_dsi_timeseries = os.path.join(folder, 'Fig7_Regional_Drought_Index_GLDAS.png')

FONT_FAMILY = 'serif'
FONT_NAME = 'Times New Roman'
plt.rcParams['font.family'] = FONT_FAMILY
plt.rcParams['font.serif'] = [FONT_NAME, 'Times', 'DejaVu Serif', 'serif']
plt.rcParams['axes.labelweight'] = 'bold'

print("Loading Groundwater Data...")
if not os.path.exists(gws_path):
    print(f"ERROR: GWS file not found at {gws_path}")
    exit()

ds_gws = xr.open_dataset(gws_path, decode_times=True)

data_vars = list(ds_gws.data_vars)
if 'spatial_ref' in data_vars:
    data_vars.remove('spatial_ref')

var_name = 'GWSA' if 'GWSA' in data_vars else data_vars[0]
print(f"Selected Variable: '{var_name}'")
da_gws = ds_gws[var_name]

if 'time' not in da_gws.dims:
    for dim in da_gws.dims:
        if dim in ['t', 'date', 'month']:
            da_gws = da_gws.rename({dim: 'time'})
            break

# --- 1. CALCULATE CLIMATOLOGY ---
print("Calculating monthly statistics (Mean & Std Dev)...")
monthly_mean = da_gws.groupby('time.month').mean('time')
monthly_std = da_gws.groupby('time.month').std('time')

# --- 2. CALCULATE GRACE-DSI ---
print("Calculating GRACE-DSI (Standardizing)...")
da_dsi = xr.apply_ufunc(
    lambda x, m, s: (x - m) / s,
    da_gws.groupby('time.month'),
    monthly_mean,
    monthly_std,
    dask='parallelized'
)
da_dsi.name = 'grace_dsi'

print(f"Saving DSI to: {dsi_path}")
da_dsi.encoding.pop('grid_mapping', None)
try:
    encoding = {'grace_dsi': {'zlib': True, 'complevel': 5, 'dtype': 'float32'}}
    da_dsi.to_netcdf(dsi_path, encoding=encoding)
except:
    da_dsi.to_netcdf(dsi_path)

# --- 3. VISUALIZATION: REGIONAL DROUGHT TIME SERIES ---
print("Generating High-Res Time Series Plot...")
regional_dsi = da_dsi.mean(dim=['lat', 'lon'])
fig, ax = plt.subplots(figsize=(10, 5))

ax.fill_between(regional_dsi.time, regional_dsi, 0, where=(regional_dsi < 0),
                color='#d62728', alpha=0.6, label='Drought (Negative)')
ax.fill_between(regional_dsi.time, regional_dsi, 0, where=(regional_dsi > 0),
                color='#1f77b4', alpha=0.4, label='Wet (Positive)')
ax.plot(regional_dsi.time, regional_dsi, color='black', linewidth=1.5)

ax.axhline(0, color='black', linewidth=1.2)
ax.axhline(-1.5, color='darkred', linestyle='--', linewidth=1.5, alpha=0.8, label='Severe Drought (-1.5)')

ax.set_ylabel("DSI (Std Dev)", fontsize=14)
ax.set_xlabel("Year", fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=13, direction='out', length=6, width=1.5)
ax.legend(loc='upper right', fontsize=13, frameon=True, framealpha=1.0, edgecolor='black')

for spine in ax.spines.values():
    spine.set_linewidth(1.5)
    spine.set_color('black')

ax.grid(True, linestyle='--', alpha=0.4, color='gray')
plt.tight_layout()
plt.savefig(fig_dsi_timeseries, dpi=600, bbox_inches='tight')
print(f"Time Series saved to: {fig_dsi_timeseries}")
plt.show()

# --- 4. VISUALIZATION: SPATIAL DROUGHT FREQUENCY ---
print("Generating High-Res Spatial Map...")
is_drought = da_dsi < -1.0
drought_freq = is_drought.sum(dim='time') / len(da_dsi.time) * 100

fig, ax = plt.subplots(figsize=(8, 7))
im = drought_freq.plot(
    ax=ax,
    cmap='YlOrRd',
    add_colorbar=False
)

ax.set_xlabel('Longitude', fontsize=14)
ax.set_ylabel('Latitude', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12, direction='out', length=6, width=1.5)

for spine in ax.spines.values():
    spine.set_linewidth(1.5)
    spine.set_color('black')
ax.grid(False)

cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.80, pad=0.04)
cbar.set_label('Drought Frequency (% of months)', fontsize=14, fontweight='bold', labelpad=12)
cbar.ax.tick_params(labelsize=12, length=5, width=1.2)
cbar.outline.set_linewidth(1.5)

plt.tight_layout()
plt.savefig(fig_dsi_map, dpi=600, bbox_inches='tight')
print(f"Map saved to: {fig_dsi_map}")
plt.show()