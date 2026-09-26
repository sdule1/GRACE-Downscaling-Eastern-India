import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import rioxarray

# --- CONFIGURATION ---
folder = '/content/drive/MyDrive/CSR_GRACE_RL0603/Masked'
nc_path = os.path.join(folder, 'GRACE_Downscaled_GWSA_1km_RF_FINAL_MASKED.nc')
well_csv = os.path.join(folder, 'Validation_Results_76Wells_RF.csv')

fig_path_combined = os.path.join(folder, 'Fig9_Combined_A4_600dpi.png')
path_map1 = os.path.join(folder, 'Fig9_Map_Alluvial_600dpi.png')
path_ts1 = os.path.join(folder, 'Fig9_TS_Alluvial_600dpi.png')
path_map2 = os.path.join(folder, 'Fig9_Map_HardRock_600dpi.png')
path_ts2 = os.path.join(folder, 'Fig9_TS_HardRock_600dpi.png')
tif_path_1 = os.path.join(folder, 'CaseStudy1_Alluvial_Zoom_GWS_RF.tif')
tif_path_2 = os.path.join(folder, 'CaseStudy2_HardRock_Zoom_GWS_RF.tif')

FONT_FAMILY = 'serif'
FONT_NAME = 'Times New Roman'
plt.rcParams['font.family'] = FONT_FAMILY
plt.rcParams['font.serif'] = [FONT_NAME, 'Times', 'DejaVu Serif', 'serif']

print("Loading Data...")
if not os.path.exists(nc_path):
    print(f"ERROR: NC file not found at {nc_path}")
    exit()
    
ds = xr.open_dataset(nc_path)
ds = ds.sortby('lat').sortby('lon')
var_name = [v for v in ds.data_vars if v not in ['crs', 'spatial_ref', 'time_bnds']][0]

if os.path.exists(well_csv):
    df_wells = pd.read_csv(well_csv)
    df_wells['Location'] = df_wells['Location'].astype(str)
else:
    print("ERROR: CSV file not found.")
    exit()

# --- DEFINE LOCATIONS ---
loc1_id = '25.508_84.625'
loc2_id = '22.82_87.009'

try:
    loc1 = df_wells[df_wells['Location'] == loc1_id].iloc[0]
    loc2 = df_wells[df_wells['Location'] == loc2_id].iloc[0]
except IndexError:
    print("Error: Specific wells not found. Using top 2 based on R.")
    df_sorted = df_wells.sort_values('R', ascending=False)
    loc1 = df_sorted.iloc[0]
    loc2 = df_sorted.iloc[1]
    
box_size = 0.2

# --- DATA EXTRACTION HELPER FUNCTIONS ---
def get_zoom_data(lat, lon, ds, var_name):
    subset = ds[var_name].sel(lat=slice(lat-box_size, lat+box_size), lon=slice(lon-box_size, lon+box_size))
    mean_map = subset.mean(dim='time')
    if mean_map.size == 0 or np.isnan(mean_map).all():
        return ds[var_name].sel(lat=lat, lon=lon, method='nearest').mean(dim='time')
    return mean_map

def get_timeseries(lat, lon, ds, var_name):
    ts = ds[var_name].sel(lat=lat, lon=lon, method='nearest')
    if len(ts) == 0 or np.isnan(ts).all():
        return None
    return (ts - ts.mean()) / ts.std()

def save_tiff(data, filepath):
    if data.ndim == 2:
        try:
            if data.rio.crs is None: data.rio.write_crs("EPSG:4326", inplace=True)
            if 'lon' in data.dims and 'lat' in data.dims:
                data.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
            data.rio.to_raster(filepath)
        except Exception as e:
            print(f"Error saving TIFF: {e}")

# --- PLOTTING HELPER FUNCTIONS ---
def render_map(ax, data, loc, panel_letter=""):
    vmin, vmax = data.min().item(), data.max().item()
    im = data.plot(ax=ax, cmap='RdBu', add_colorbar=False, vmin=vmin, vmax=vmax)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('GWS Anomaly (cm)', fontsize=10, fontweight='bold')
    ticks = np.linspace(vmin, vmax, 5)
    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels([f"{x:.1f}" for x in ticks])
    cbar.ax.tick_params(labelsize=9)
    
    ax.plot(loc.Lon, loc.Lat, 'k*', markersize=13, markeredgecolor='white', markeredgewidth=1.2, label='Well Location')
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')
    ax.tick_params(axis='both', labelsize=11, direction='out', length=4, width=1.2)
    ax.set_xlabel("Longitude", fontsize=12, fontweight='bold')
    ax.set_ylabel("Latitude", fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    
    if panel_letter:
        ax.annotate(panel_letter, xy=(0.04, 0.96), xycoords='axes fraction',
                    fontsize=12, fontweight='bold', va='top', ha='left',
                    bbox=dict(boxstyle='square,pad=0.2', fc='white', alpha=0.9, ec='black', lw=1.5))

def render_ts(ax, ts, loc, color, panel_letter=""):
    ax.plot(ts.time, ts, color=color, linewidth=2, label='Downscaled GWS')
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')
    ax.tick_params(axis='both', labelsize=11, direction='out', length=4, width=1.2)
    ax.set_ylabel("Z-Score", fontsize=12, fontweight='bold')
    ax.set_xlabel("Year", fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5, color='gray', linewidth=0.8)
    
    legend_text = f"Downscaled GWS\n(R = {loc.R:.2f})"
    ax.plot([], [], color=color, linewidth=2, label=legend_text) 
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[-1:], labels[-1:], loc='upper right', fontsize=11, framealpha=1, edgecolor='black')
    
    if panel_letter:
        ax.annotate(panel_letter, xy=(0.02, 0.96), xycoords='axes fraction',
                    fontsize=12, fontweight='bold', va='top', ha='left',
                    bbox=dict(boxstyle='square,pad=0.2', fc='white', alpha=0.9, ec='black', lw=1.5))

# --- EXTRACT ALL DATA ONCE ---
print("Extracting Subset Data...")
data_box1 = get_zoom_data(loc1.Lat, loc1.Lon, ds, var_name)
ts1 = get_timeseries(loc1.Lat, loc1.Lon, ds, var_name)
data_box2 = get_zoom_data(loc2.Lat, loc2.Lon, ds, var_name)
ts2 = get_timeseries(loc2.Lat, loc2.Lon, ds, var_name)

save_tiff(data_box1, tif_path_1)
save_tiff(data_box2, tif_path_2)

# ==========================================
# 1. GENERATE COMBINED A4 MANUSCRIPT FIGURE
# ==========================================
print("Generating Combined A4 Figure...")
fig = plt.figure(figsize=(8.27, 6.5))
gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 2.5], hspace=0.35, wspace=0.2)

ax_map1 = fig.add_subplot(gs[0, 0])
ax_ts1 = fig.add_subplot(gs[0, 1])
ax_map2 = fig.add_subplot(gs[1, 0])
ax_ts2 = fig.add_subplot(gs[1, 1])

render_map(ax_map1, data_box1, loc1, "(a)")
if ts1 is not None: render_ts(ax_ts1, ts1, loc1, '#1f77b4', "(b)")
render_map(ax_map2, data_box2, loc2, "(c)")
if ts2 is not None: render_ts(ax_ts2, ts2, loc2, '#d62728', "(d)")

fig.align_ylabels([ax_map1, ax_map2])
fig.align_ylabels([ax_ts1, ax_ts2])
plt.savefig(fig_path_combined, dpi=600, bbox_inches='tight')
plt.close(fig)

# ==========================================
# 2. GENERATE SEPARATE HIGH-RES PLOTS
# ==========================================
print("Generating Separate High-Res Figures...")
fig_m1, ax_m1 = plt.subplots(figsize=(4, 3.5))
render_map(ax_m1, data_box1, loc1)
plt.savefig(path_map1, dpi=600, bbox_inches='tight')
plt.close(fig_m1)

if ts1 is not None:
    fig_t1, ax_t1 = plt.subplots(figsize=(7, 4))
    render_ts(ax_t1, ts1, loc1, '#1f77b4')
    plt.savefig(path_ts1, dpi=600, bbox_inches='tight')
    plt.close(fig_t1)

fig_m2, ax_m2 = plt.subplots(figsize=(4, 3.5))
render_map(ax_m2, data_box2, loc2)
plt.savefig(path_map2, dpi=600, bbox_inches='tight')
plt.close(fig_m2)

if ts2 is not None:
    fig_t2, ax_t2 = plt.subplots(figsize=(7, 4))
    render_ts(ax_t2, ts2, loc2, '#d62728')
    plt.savefig(path_ts2, dpi=600, bbox_inches='tight')
    plt.close(fig_t2)

print("All outputs generated successfully!")