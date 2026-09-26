import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
folder = '/content/drive/MyDrive/CSR_GRACE_RL0603/Masked'

coarse_grace_path = os.path.join(folder, 'GRACE_StudyArea_Subset_ML_Filled_2003_23.nc')
soil_path = os.path.join(folder, 'TerraClimate_Soil_2003_2023.nc')
gldas_path = os.path.join(folder, 'GLDAS_Anomalies_Final_252.nc')

output_coarse_nc = os.path.join(folder, 'GRACE_Coarse_GWSA_0.25deg_2003_2023.nc')
fig_coarse_amp = os.path.join(folder, 'Fig_Coarse_Seasonal_Amplitude_RF.png')

FONT_FAMILY = 'serif'
FONT_NAME = 'Times New Roman'
plt.rcParams['font.family'] = FONT_FAMILY
plt.rcParams['font.serif'] = [FONT_NAME, 'Times', 'DejaVu Serif', 'serif']

print("Loading Datasets...")
ds_grace = xr.open_dataset(coarse_grace_path)
ds_soil = xr.open_dataset(soil_path).sel(time=slice('2003-01-01', '2023-12-31')).resample(time='MS').mean()
gldas_1km = xr.open_dataset(gldas_path)

common_time = np.intersect1d(ds_grace.time, ds_soil.time)
ds_grace = ds_grace.sel(time=common_time)
ds_soil = ds_soil.sel(time=common_time)

if 'y' in gldas_1km.dims:
    gldas_1km = gldas_1km.rename({'y': 'lat', 'x': 'lon'})

gldas_1km['time'] = ds_grace['time'].values
var_tws = [v for v in ds_grace.data_vars if v not in ['crs', 'spatial_ref']][0]

# --- 2. CALCULATE AND UPSCALE NON-GROUNDWATER COMPONENTS ---
print("Calculating Soil Anomaly and Upscaling all variables to 0.25 deg...")
soil_mean = ds_soil['soil'].sel(time=slice('2004-01-01', '2009-12-31')).mean(dim='time')
sma_native = (ds_soil['soil'] - soil_mean) / 10.0
sma_coarse = sma_native.interp_like(ds_grace[var_tws], method='linear')
cwa_coarse = gldas_1km['CWA'].interp_like(ds_grace[var_tws], method='linear')
swea_coarse = gldas_1km['SWEA'].interp_like(ds_grace[var_tws], method='linear')

# --- 3. ISOLATE COARSE GROUNDWATER ---
print("Applying RAM-safe physical subtraction (All units perfectly scaled)...")
raw_grace = ds_grace[var_tws].values
raw_sma = sma_coarse.values
raw_cwa = cwa_coarse.values
raw_swea = swea_coarse.values

raw_gws = raw_grace - (raw_sma + raw_cwa + raw_swea)

da_gws_coarse = xr.DataArray(
    raw_gws,
    coords=ds_grace[var_tws].coords,
    dims=ds_grace[var_tws].dims,
    name='GWSA'
)

print(f"Saving Coarse GWSA to {output_coarse_nc}...")
encoding = {'GWSA': {'zlib': True, 'complevel': 5, 'dtype': 'float32'}}
try:
    da_gws_coarse.rio.write_crs("EPSG:4326", inplace=True)
except: pass
da_gws_coarse.to_netcdf(output_coarse_nc, encoding=encoding)

# --- 4. COARSE SEASONAL AMPLITUDE ANALYSIS ---
print("Calculating Coarse Seasonal Amplitude...")
monthly_climatology = da_gws_coarse.groupby('time.month').mean('time')
amplitude = (monthly_climatology.max(dim='month') - monthly_climatology.min(dim='month')) / 2

# --- 5. VISUALIZATION ---
print("Generating Coarse Amplitude Map...")
fig, ax = plt.subplots(figsize=(14, 12))
im = amplitude.plot(ax=ax, cmap='YlGnBu', add_colorbar=False)
ax.set_title('Coarse GWS Seasonal Oscillation Strength\n(Recharge Potential)', fontsize=24, fontweight='bold', pad=20)
ax.set_xlabel('Longitude', fontsize=18, fontweight='bold')
ax.set_ylabel('Latitude', fontsize=18, fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=16, direction='in', length=8, width=2)

for spine in ax.spines.values():
    spine.set_linewidth(2)
    spine.set_color('black')

cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.85, pad=0.03)
cbar.set_label('GWS Amplitude (cm)', fontsize=18, fontweight='bold', labelpad=15)
cbar.ax.tick_params(labelsize=16, length=6, width=1.5)
cbar.outline.set_linewidth(1.5)

plt.tight_layout()
plt.savefig(fig_coarse_amp, dpi=300, bbox_inches='tight')
print(f"SUCCESS! Figure saved to: {fig_coarse_amp}")
plt.show()