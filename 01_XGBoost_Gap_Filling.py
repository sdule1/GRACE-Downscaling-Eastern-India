import xarray as xr
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
folder = '/content/drive/MyDrive/CSR_GRACE_RL0603/Masked'
grace_path = os.path.join(folder, 'GRACE_StudyArea_Subset_2003_2023.nc')
ppt_path = os.path.join(folder, 'TerraClimate_Precip_2003_2023.nc')
temp_path = os.path.join(folder, 'TerraClimate_Temp_2003_2023.nc')

# --- TIME FIXER FUNCTION ---
def fix_time_index(ds, ds_name):
    """Ensures the dataset has a valid DatetimeIndex"""
    print(f"Checking time index for {ds_name}...")
    if isinstance(ds.indexes['time'], pd.DatetimeIndex):
        return ds
    try:
        ds = xr.decode_cf(ds)
        if isinstance(ds.indexes['time'], pd.DatetimeIndex): return ds
    except: pass
    try:
        ds['time'] = ds.indexes['time'].to_datetimeindex()
        return ds
    except: pass
    print(f" -> Forcing manual time fix for {ds_name}...")
    try:
        new_times = pd.to_datetime(ds.time.values, unit='D', origin='2002-01-01')
        ds['time'] = new_times
    except:
        start_date = '2002-04-01'
        new_times = pd.date_range(start=start_date, periods=len(ds.time), freq='MS')
        ds['time'] = new_times
    return ds

# 1. Load Datasets
print("Loading datasets...")
ds_grace = xr.open_dataset(grace_path)
ds_ppt = xr.open_dataset(ppt_path)
ds_temp = xr.open_dataset(temp_path)

# 2. FIX TIME BEFORE SLICING
ds_grace = fix_time_index(ds_grace, "GRACE")
ds_ppt = fix_time_index(ds_ppt, "Precipitation")
ds_temp = fix_time_index(ds_temp, "Temperature")

# 3. ALIGN TIME RANGES (2003-2023)
print("Aligning time ranges (2003-2023)...")
start_date = '2003-01-01'
end_date = '2023-12-31'
ds_grace = ds_grace.sel(time=slice(start_date, end_date))
ds_ppt = ds_ppt.sel(time=slice(start_date, end_date))
ds_temp = ds_temp.sel(time=slice(start_date, end_date))

# Ensure monthly frequency
ds_grace = ds_grace.resample(time='MS').mean()
ds_ppt = ds_ppt.resample(time='MS').mean()
ds_temp = ds_temp.resample(time='MS').mean()

# 4. HARMONIZE GRIDS
print("Interpolating predictors to GRACE grid...")
ds_ppt_coarse = ds_ppt.interp_like(ds_grace, method='linear')
ds_temp_coarse = ds_temp.interp_like(ds_grace, method='linear')

# 5. CONSTRUCT DATAFRAME
print("Structuring data for XGBoost...")
ppt_var = 'ppt' if 'ppt' in ds_ppt_coarse else list(ds_ppt_coarse.data_vars)[0]
temp_var = 'tmax' if 'tmax' in ds_temp_coarse else list(ds_temp_coarse.data_vars)[0]
df = ds_grace['lwe_thickness'].to_dataframe().reset_index()

df_ppt = ds_ppt_coarse[ppt_var].to_dataframe().reset_index()
df_temp = ds_temp_coarse[temp_var].to_dataframe().reset_index()
df[ppt_var] = df_ppt[ppt_var]
df[temp_var] = df_temp[temp_var]
print(f"Total rows: {len(df)}")

# 6. FEATURE ENGINEERING
print("Creating Lag features...")
df = df.sort_values(by=['lat', 'lon', 'time'])
df['ppt_1mo_ago'] = df.groupby(['lat', 'lon'])[ppt_var].shift(1)
df['ppt_2mo_ago'] = df.groupby(['lat', 'lon'])[ppt_var].shift(2)
df['ppt_3mo_ago'] = df.groupby(['lat', 'lon'])[ppt_var].shift(3)

# 7. TRAIN XGBOOST
features = [ppt_var, temp_var, 'ppt_1mo_ago', 'ppt_2mo_ago', 'ppt_3mo_ago']
target = 'lwe_thickness'
df_clean = df.dropna(subset=features)
print(f"Rows with valid predictors: {len(df_clean)}")

train_data = df_clean.dropna(subset=[target])
predict_data = df_clean[df_clean[target].isnull()]
print(f"Training on {len(train_data)} points. Filling {len(predict_data)} gaps.")

if len(train_data) == 0:
    print("CRITICAL ERROR: No training data found. Check variable names or overlap.")
else:
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, objective='reg:squarederror')
    model.fit(train_data[features], train_data[target])
    
    # 8. PREDICT GAPS
    if len(predict_data) > 0:
        print("Filling gaps...")
        predicted_values = model.predict(predict_data[features])
        df.loc[predict_data.index, target] = predicted_values
        print("Gaps filled.")
    else:
        print("No gaps found (Dataset might be complete).")
        
    # 9. SAVE
    ds_filled = df.set_index(['time', 'lat', 'lon']).to_xarray()
    output_path = os.path.join(folder, 'GRACE_StudyArea_Subset_ML_Filled_2003_23.nc')
    ds_filled.to_netcdf(output_path)
    print(f"SUCCESS! Saved to: {output_path}")
    
    # Plot
    plt.figure(figsize=(15, 5))
    ds_grace['lwe_thickness'].mean(dim=['lat','lon']).plot(label='Original', marker='o', markersize=3)
    ds_filled['lwe_thickness'].mean(dim=['lat','lon']).plot(label='Filled', linestyle='--', alpha=0.7)
    plt.title("Gap Filling Result (2003-2023)")
    plt.legend()
    plt.show()