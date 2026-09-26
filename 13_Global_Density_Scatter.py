import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, gaussian_kde
import os

# --- CONFIGURATION ---
base_folder = '/content/drive/MyDrive/CSR_GRACE_RL0603/Masked'
well_folder = '/content/drive/MyDrive/CGWB_WL_2003_23'
grace_path = os.path.join(base_folder, 'GRACE_Downscaled_GWSA_1km_RF_FINAL_MASKED.nc')
wells_path = os.path.join(well_folder, 'Selected_Validation_Wells28_MinMaxYear19.xlsx')
output_fig = os.path.join(base_folder, 'Fig5_Global_Density_Scatter_RFModel.png')

FONT_FAMILY = 'serif'
FONT_NAME = 'Times New Roman'
plt.rcParams['font.family'] = FONT_FAMILY
plt.rcParams['font.serif'] = [FONT_NAME, 'Times', 'DejaVu Serif', 'serif']

# --- LOAD DATA ---
print("Loading data for Global Scatter Plot...")
df = pd.read_excel(wells_path)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['UID'] = df['LATITUDE'].round(3).astype(str) + "_" + df['LONGITUDE'].round(3).astype(str)

try:
    ds = xr.open_dataset(grace_path)
    var_name = [v for v in ds.data_vars if v not in ['crs', 'spatial_ref', 'time_bnds']][0]
except:
    print("Error loading GRACE file.")
    exit()

# --- COLLECT DATA ---
all_obs_z = []      
all_pred_z = []     
all_obs_raw = []    
print(f"Aggregating data points from {len(df['UID'].unique())} wells...")

unique_wells = df.groupby('UID')
for uid, group in unique_wells:
    try:
        lat = group.iloc[0]['LATITUDE']
        lon = group.iloc[0]['LONGITUDE']
        group = group.sort_values('Date')
        group = group[(group['Date'].dt.year >= 2003) & (group['Date'].dt.year <= 2023)]
        
        well_values = -1 * group['DTWL'].values
        well_dates = group['Date'].values
        if len(well_values) < 10: continue
        
        grace_values = []
        valid_well_values = []
        for w_date, w_val in zip(well_dates, well_values):
            if np.isnan(w_val): continue
            try:
                grace_val = ds[var_name].sel(lat=lat, lon=lon, method='nearest').sel(time=w_date, method='nearest').values
                if not np.isnan(grace_val):
                    grace_values.append(float(grace_val))
                    valid_well_values.append(w_val)
            except: continue
            
        if len(grace_values) < 10: continue
        
        ts_well = np.array(valid_well_values)
        ts_grace = np.array(grace_values)
        z_well = (ts_well - np.mean(ts_well)) / np.std(ts_well)
        z_grace = (ts_grace - np.mean(ts_grace)) / np.std(ts_grace)
        
        all_obs_z.extend(z_well)
        all_pred_z.extend(z_grace)
        all_obs_raw.extend(ts_well) 
    except: continue

# --- CALCULATE METRICS ---
x = np.array(all_obs_z)
y = np.array(all_pred_z)
print(f"Total Data Points: {len(x)}")

r_global, _ = pearsonr(x, y)
rmse_global_z = np.sqrt(np.mean((x - y)**2))
raw_obs_cm = np.array(all_obs_raw) * 100
std_obs_cm = np.nanstd(raw_obs_cm)
rmse_cm = rmse_global_z * std_obs_cm

print(f"\n--- Validation Metrics ---")
print(f"Global Pearson R: {r_global:.2f}")
print(f"Z-Score RMSE: {rmse_global_z:.2f}")
print(f"Std Dev of Raw Obs: {std_obs_cm:.2f} cm")
print(f"Back-Transformed RMSE: {rmse_cm:.2f} cm\n")

# --- CALCULATE DENSITY ---
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]

# --- PLOTTING ---
fig, ax = plt.subplots(figsize=(7.5, 6.5))
sc = ax.scatter(x, y, c=z, s=15, cmap='turbo', edgecolors='none', alpha=0.85)

ax.plot([-4, 4], [-4, 4], 'k--', linewidth=2, label='1:1 Line')
m, b = np.polyfit(x, y, 1)
ax.plot(x, m*x + b, 'r-', linewidth=2.5, label=f'Best Fit ($y={m:.2f}x + {b:.2f}$)')

ax.set_xlabel('Observed GWSA (Z-Score)', fontsize=14, fontweight='bold')
ax.set_ylabel('Predicted GWSA (Z-Score)', fontsize=14, fontweight='bold')
ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-3.5, 3.5)

ax.tick_params(axis='both', which='major', labelsize=13, width=1.5, length=4,
               direction='out', top=False, right=False)
ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.7)

for spine in ax.spines.values(): spine.set_linewidth(1.5)

stats_text = (f"Global Pearson $R$ = {r_global:.2f}\n"
              f"Z-Score RMSE = {rmse_global_z:.2f}\n"
              f"Physical RMSE = {rmse_cm:.2f} cm\n"
              f"Total Points (N) = {len(x)}")
ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
         fontsize=12, fontweight='bold', verticalalignment='top',
         bbox=dict(boxstyle='square,pad=0.6', facecolor='white', alpha=0.95,
                   edgecolor='black', linewidth=1.5))

legend = ax.legend(loc='lower right', fontsize=12, frameon=True, edgecolor='black')
legend.get_frame().set_linewidth(1.5)

cbar = plt.colorbar(sc, ax=ax, pad=0.02)
cbar.set_label('Point Density', fontsize=12, fontweight='bold')
cbar.ax.tick_params(labelsize=11, direction='out')
cbar.outline.set_linewidth(1.5)

plt.tight_layout()
plt.savefig(output_fig, dpi=600, bbox_inches='tight')
print(f"Saved Figure 5 to {output_fig}")
plt.show()