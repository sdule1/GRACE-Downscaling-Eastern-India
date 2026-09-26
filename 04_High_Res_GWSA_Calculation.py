import xarray as xr
import rioxarray
import os

# --- CONFIGURATION ---
folder = '/content/drive/MyDrive/CSR_GRACE_RL0603/Masked'

# Inputs
twsa_path = os.path.join(folder, 'GRACE_Downscaled_TWSA_1km_RF_FINAL.nc')
soil_path = os.path.join(folder, 'TerraClimate_Soil_2003_2023.nc')
gldas_path = os.path.join(folder, 'GLDAS_Anomalies_Final_252.nc')

# Output
output_path = os.path.join(folder, 'GRACE_Downscaled_GWSA_1km_RF_FINAL.nc')

print("\n--- Correcting GWSA Units (UNMASKED) ---")

# 1. Load the data
print("Loading datasets from disk...")
da_final_twsa = xr.open_dataset(twsa_path)['TWSA'] # This is in CM
ds_soil = xr.open_dataset(soil_path).sel(time=slice('2003-01-01', '2023-12-31')).resample(time='MS').mean()
gldas_1km = xr.open_dataset(gldas_path)

# 2. Calculate SMA and apply CRITICAL UNIT CONVERSIONS
print("Calculating anomalies and applying unit corrections...")
baseline_soil = ds_soil['soil'].sel(time=slice('2004-01-01', '2009-12-31')).mean(dim='time')

# ONLY divide Soil Moisture by 10 to convert mm to cm!
sma_cm = (ds_soil['soil'] - baseline_soil) / 10.0

# Leave GLDAS as-is based on raw data inspection
cwa_cm = gldas_1km['CWA']
swea_cm = gldas_1km['SWEA']

# 3. Interpolate to 1km to match TWSA
print("Interpolating to 1km grid...")
sma_1km = sma_cm.interp_like(da_final_twsa, method='linear')
cwa_1km = cwa_cm.interp_like(da_final_twsa, method='linear')
swea_1km = swea_cm.interp_like(da_final_twsa, method='linear')

# 4. Synchronize spatial/temporal dimensions
if 'y' in cwa_1km.dims:
    cwa_1km = cwa_1km.rename({'y': 'lat', 'x': 'lon'})
    swea_1km = swea_1km.rename({'y': 'lat', 'x': 'lon'})

cwa_1km['time'] = da_final_twsa['time'].values
cwa_1km['lat'] = da_final_twsa['lat'].values
cwa_1km['lon'] = da_final_twsa['lon'].values

swea_1km['time'] = da_final_twsa['time'].values
swea_1km['lat'] = da_final_twsa['lat'].values
swea_1km['lon'] = da_final_twsa['lon'].values

sma_1km['time'] = da_final_twsa['time'].values
sma_1km['lat'] = da_final_twsa['lat'].values
sma_1km['lon'] = da_final_twsa['lon'].values

# 5. Final Subtraction: GWSA = TWSA - (SMA + CWA + SWEA)
print("Applying physical subtraction (All units now perfectly scaled)...")
gwsa_final = da_final_twsa - (sma_1km + cwa_1km + swea_1km)
gwsa_final.name = 'GWSA'

# Assign CRS
gwsa_final.rio.write_crs("EPSG:4326", inplace=True)

# 6. Save the final unmasked, corrected dataset
print("Saving corrected 1km Groundwater NetCDF...")
encoding = {'GWSA': {'zlib': True, 'complevel': 5, 'dtype': 'float32'}}
gwsa_final.to_netcdf(output_path, encoding=encoding)
print(f"SUCCESS! Physically accurate, unmasked GWSA saved to: {output_path}")