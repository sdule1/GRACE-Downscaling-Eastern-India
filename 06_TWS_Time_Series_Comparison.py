import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
folder = '/content/drive/MyDrive/CSR_GRACE_RL0603/Masked'

orig_path = os.path.join(folder, 'GRACE_StudyArea_Subset_ML_Filled_2003_23_MASKED.nc')
ml_only_path = os.path.join(folder, 'GRACE_Downscaled_TWSA_1km_ML_ONLY_RF_MASKED.nc')
corrected_path = os.path.join(folder, 'GRACE_Downscaled_TWSA_1km_RF_FINAL_MASKED.nc')

fig_ts_path = os.path.join(folder, 'Fig_TimeSeries_Comparison_PubReady_RF.png')
fig_hist_path = os.path.join(folder, 'Fig_Spatial_Distribution_Histogram_RF.png')

FONT_FAMILY = 'serif'
FONT_NAME = 'Times New Roman'
plt.rcParams['font.family'] = FONT_FAMILY
plt.rcParams['font.serif'] = [FONT_NAME, 'Times', 'DejaVu Serif', 'serif']

print("Loading datasets...")
try:
    if not os.path.exists(orig_path): orig_path = orig_path.replace('_MASKED', '')
    if not os.path.exists(ml_only_path): ml_only_path = ml_only_path.replace('_MASKED', '')
    if not os.path.exists(corrected_path): corrected_path = corrected_path.replace('_MASKED', '')
    
    ds_orig = xr.open_dataset(orig_path)
    ds_ml = xr.open_dataset(ml_only_path)
    ds_corr = xr.open_dataset(corrected_path)
    
    var_orig = [v for v in ds_orig.data_vars if v not in ['crs', 'spatial_ref', 'time_bnds']][0]
    var_ml = [v for v in ds_ml.data_vars if v not in ['crs', 'spatial_ref', 'time_bnds']][0]
    var_corr = [v for v in ds_corr.data_vars if v not in ['crs', 'spatial_ref', 'time_bnds']][0]
    
    print("Aligning time series (2003-2023)...")
    time_slice = slice('2003-01-01', '2023-12-31')
    ds_orig = ds_orig.sel(time=time_slice).resample(time='MS').mean()
    ds_ml = ds_ml.sel(time=time_slice).resample(time='MS').mean()
    ds_corr = ds_corr.sel(time=time_slice).resample(time='MS').mean()
    
    # --- PART 1: TIME SERIES PLOT ---
    print("Generating Time Series Plot...")
    ts_orig = ds_orig[var_orig].mean(dim=['lat', 'lon'])
    ts_ml = ds_ml[var_ml].mean(dim=['lat', 'lon'])
    ts_corr = ds_corr[var_corr].mean(dim=['lat', 'lon'])
    
    plt.figure(figsize=(12, 7))
    ts_orig.plot(label='Original GRACE (Filled)', color='gray', linewidth=3.5, alpha=0.6, zorder=1)
    ts_ml.plot(label='ML Predicted (Pre-Correction)', color='blue', linestyle=':', linewidth=2, alpha=0.8, zorder=2)
    ts_corr.plot(label='Final Downscaled (Corrected)', color='red', linestyle='--', linewidth=2, alpha=1.0, zorder=3)
    
    plt.title("", fontsize=18, fontweight='bold', pad=15)
    plt.ylabel("TWS Anomaly (cm)", fontsize=15, fontweight='bold')
    plt.xlabel("Year", fontsize=15, fontweight='bold')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc='upper right', fontsize=15, frameon=True, framealpha=0.95, edgecolor='black')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(fig_ts_path, dpi=600, bbox_inches='tight')
    print(f"Saved Time Series to: {fig_ts_path}")
    plt.close()
    
    # --- PART 2: SPATIAL VARIABILITY HISTOGRAM ---
    print("Generating Histogram Plot...")
    data_orig = ds_orig[var_orig].values.flatten()
    data_downscaled = ds_corr[var_corr].values.flatten()
    
    data_orig = data_orig[~np.isnan(data_orig)]
    data_downscaled = data_downscaled[~np.isnan(data_downscaled)]
    
    plt.figure(figsize=(10, 6))
    combined_data = np.concatenate([data_orig, data_downscaled])
    bins = np.linspace(np.min(combined_data), np.max(combined_data), 60)
    
    plt.hist(data_downscaled, bins=bins, density=True, color='red', alpha=0.7,
             label='Downscaled (1km)', edgecolor='none', zorder=2)
    plt.hist(data_orig, bins=bins, density=True, color='blue', alpha=0.5,
             label='Original (0.25°)', edgecolor='none', zorder=1)
             
    plt.title("Distribution of TWS Values (Spatial Variability)", fontsize=18, fontweight='bold', pad=15)
    plt.ylabel("Frequency (Density)", fontsize=14, fontweight='bold')
    plt.xlabel("TWS (cm)", fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='-', alpha=0.3, color='lightgray')
    plt.legend(loc='upper right', fontsize=12, frameon=True, framealpha=0.95, edgecolor='gray')
    plt.tight_layout()
    plt.savefig(fig_hist_path, dpi=600, bbox_inches='tight')
    print(f"Saved Histogram to: {fig_hist_path}")
    plt.show()

except FileNotFoundError as e:
    print(f"\nERROR: Could not find dataset.\n{e}")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")