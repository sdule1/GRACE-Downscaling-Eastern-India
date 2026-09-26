import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import pearsonr

# --- 1. USER PLOT STYLING & CUSTOMIZATION BLOCK ---
PLOT_CONFIG = {
    'figsize': (11, 7),
    'dpi': 300,
    'font_family': 'serif',
    'font_name': 'Times New Roman',
    'title_size': 13,
    'title_weight': 'bold',              
    'title_pad': 12,
    'label_size': 16.5,
    'label_weight': 'bold',             
    'label_pad': 8,
    'tick_direction': 'out',            
    'tick_label_size': 16.5,
    'tick_label_weight': 'normal',      
    'tick_major_length': 6,
    'tick_major_width': 1.5,
    'tick_minor_length': 3.5,
    'tick_minor_width': 1.0,
    'enable_minor_ticks': False,
    'spine_width': 1.5,                 
    'spine_color': 'black',
    'grace_label': 'GRACE (Downscaled GWS)',
    'grace_color': '#D62728',           
    'grace_linestyle': '-',
    'grace_linewidth': 1.8,
    'grace_marker': 'o',
    'grace_markersize': 6,
    'grace_markeredgewidth': 1.3,
    'grace_markerfacecolor': '#D62728',
    'grace_markeredgecolor': 'black',
    'grace_alpha': 0.85,
    'well_label': 'In-situ Well',
    'well_color': '#1F77B4',            
    'well_linestyle': '--',
    'well_linewidth': 1.8,
    'well_marker': 's',                 
    'well_markersize': 6,
    'well_markeredgewidth': 1.3,
    'well_markerfacecolor': 'none',     
    'well_markeredgecolor': '#1F77B4',
    'well_alpha': 0.85,
    'legend_loc': 'upper right',
    'legend_fontsize': 12.5,
    'legend_fontweight': 'bold',        
    'legend_frameon': True,
    'legend_framealpha': 0.9,
    'legend_facecolor': 'white',
    'legend_edgecolor': 'black',
    'legend_box_linewidth': 1.2,        
    'grid_enable': True,
    'grid_linestyle': ':',
    'grid_linewidth': 0.8,
    'grid_alpha': 0.6,
}

plt.rcParams['font.family'] = PLOT_CONFIG['font_family']
plt.rcParams['font.serif'] = [PLOT_CONFIG['font_name'], 'Times', 'DejaVu Serif', 'serif']

# --- 2. FILE PATHS & DIRECTORIES ---
base_folder = '/content/drive/MyDrive/CSR_GRACE_RL0603/Masked'
well_folder = '/content/drive/MyDrive/CGWB_WL_2003_23'
grace_path = os.path.join(base_folder, 'GRACE_Downscaled_GWSA_1km_RF_FINAL_MASKED.nc')
wells_path = os.path.join(well_folder, 'Selected_Validation_Wells28_MinMaxYear19.xlsx')
output_fig_folder = os.path.join(base_folder, 'Validation_Plots_76_Wells_RF')
output_csv_path = os.path.join(base_folder, 'Validation_Results_76Wells_RF.csv')

if not os.path.exists(output_fig_folder):
    os.makedirs(output_fig_folder)

# --- 3. DATA LOADING ---
print("Loading Datasets...")
try:
    ds = xr.open_dataset(grace_path)
    if 'GWSA' in ds.data_vars:
        var_name = 'GWSA'
    else:
        var_name = None
        for v in ds.data_vars:
            if v not in ['crs', 'spatial_ref', 'time_bnds', 'grid_mapping']:
                var_name = v
                break
        if var_name is None:
            var_name = list(ds.data_vars)[0]
    print(f"Loaded GRACE: {os.path.basename(grace_path)} (Using Variable: {var_name})")
except FileNotFoundError:
    print(f"CRITICAL ERROR: GRACE file not found at {grace_path}")
    exit()

df = pd.read_excel(wells_path)
print(f"Loaded Excel with {len(df)} rows.")
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['UID'] = df['LATITUDE'].round(3).astype(str) + "_" + df['LONGITUDE'].round(3).astype(str)

# --- 4. VALIDATION LOOP ---
results = []
unique_wells = df.groupby('UID')
print(f"\nProcessing {len(unique_wells)} unique well locations...")
count = 0

for uid, group in unique_wells:
    count += 1
    try:
        meta = group.iloc[0]
        lat = meta['LATITUDE']
        lon = meta['LONGITUDE']
        location_name = str(meta.get('LOCATIONS', uid))
        state_name = str(meta.get('STATE', 'Unknown'))
        
        if 'STATE_UT' in meta:
            state_name = meta['STATE_UT']
        if state_name == 'Unknown':
            loc_upper = location_name.upper()
            if 'WEST BENGAL' in loc_upper: state_name = 'WEST BENGAL'
            elif 'BIHAR' in loc_upper: state_name = 'BIHAR'
            elif 'JHARKHAND' in loc_upper: state_name = 'JHARKHAND'
            
        group = group.sort_values('Date')
        group = group[(group['Date'].dt.year >= 2003) & (group['Date'].dt.year <= 2023)]
        well_values = -1 * group['DTWL'].values
        well_dates = group['Date'].values
        
        if len(well_values) < 10: continue
        
        grace_values = []
        valid_well_values = []
        valid_dates = []
        for w_date, w_val in zip(well_dates, well_values):
            if np.isnan(w_val): continue
            try:
                grace_val = ds[var_name].sel(lat=lat, lon=lon, method='nearest').sel(time=w_date, method='nearest').values
                if not np.isnan(grace_val):
                    grace_values.append(float(grace_val))
                    valid_well_values.append(w_val)
                    valid_dates.append(w_date)
            except Exception:
                continue
                
        if len(grace_values) < 10: continue
        
        ts_well = np.array(valid_well_values)
        ts_grace = np.array(grace_values)
        z_well = (ts_well - np.mean(ts_well)) / np.std(ts_well)
        z_grace = (ts_grace - np.mean(ts_grace)) / np.std(ts_grace)
        
        r_score, p_val = pearsonr(z_well, z_grace)
        rmse = np.sqrt(np.mean((z_well - z_grace)**2))
        
        results.append({
            'Location': uid,
            'Name': location_name,
            'STATE': state_name,
            'Lat': lat,
            'Lon': lon,
            'R': r_score,
            'RMSE': rmse,
            'Points': len(z_grace)
        })
        
        if count % 10 == 0:
            print(f"Processed {count} wells...")
            
        fig, ax = plt.subplots(figsize=PLOT_CONFIG['figsize'], dpi=PLOT_CONFIG['dpi'])
        dynamic_well_label = f"{PLOT_CONFIG['well_label']} (r = {r_score:.2f})"
        
        ax.plot(
            valid_dates, z_grace, color=PLOT_CONFIG['grace_color'],
            linestyle=PLOT_CONFIG['grace_linestyle'], linewidth=PLOT_CONFIG['grace_linewidth'],
            marker=PLOT_CONFIG['grace_marker'], markersize=PLOT_CONFIG['grace_markersize'],
            markeredgewidth=PLOT_CONFIG['grace_markeredgewidth'],
            markerfacecolor=PLOT_CONFIG['grace_markerfacecolor'],
            markeredgecolor=PLOT_CONFIG['grace_markeredgecolor'],
            alpha=PLOT_CONFIG['grace_alpha'], label=PLOT_CONFIG['grace_label']
        )
        ax.plot(
            valid_dates, z_well, color=PLOT_CONFIG['well_color'],
            linestyle=PLOT_CONFIG['well_linestyle'], linewidth=PLOT_CONFIG['well_linewidth'],
            marker=PLOT_CONFIG['well_marker'], markersize=PLOT_CONFIG['well_markersize'],
            markeredgewidth=PLOT_CONFIG['well_markeredgewidth'],
            markerfacecolor=PLOT_CONFIG['well_markerfacecolor'],
            markeredgecolor=PLOT_CONFIG['well_markeredgecolor'],
            alpha=PLOT_CONFIG['well_alpha'], label=dynamic_well_label
        )
        
        ax.set_title(f"Well Location: {location_name} ({state_name})", fontsize=PLOT_CONFIG['title_size'], fontweight=PLOT_CONFIG['title_weight'], pad=PLOT_CONFIG['title_pad'])
        ax.set_xlabel("Date", fontsize=PLOT_CONFIG['label_size'], fontweight=PLOT_CONFIG['label_weight'], labelpad=PLOT_CONFIG['label_pad'])
        ax.set_ylabel("Standardized Anomaly (Z-Score)", fontsize=PLOT_CONFIG['label_size'], fontweight=PLOT_CONFIG['label_weight'], labelpad=PLOT_CONFIG['label_pad'])
        
        ax.tick_params(axis='both', which='major', direction=PLOT_CONFIG['tick_direction'], length=PLOT_CONFIG['tick_major_length'], width=PLOT_CONFIG['tick_major_width'], labelsize=PLOT_CONFIG['tick_label_size'])
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight(PLOT_CONFIG['tick_label_weight'])
            
        for spine in ax.spines.values():
            spine.set_linewidth(PLOT_CONFIG['spine_width'])
            spine.set_color(PLOT_CONFIG['spine_color'])
            
        if PLOT_CONFIG['grid_enable']:
            ax.grid(True, which='major', linestyle=PLOT_CONFIG['grid_linestyle'], linewidth=PLOT_CONFIG['grid_linewidth'], alpha=PLOT_CONFIG['grid_alpha'])
            
        leg = ax.legend(loc=PLOT_CONFIG['legend_loc'], fontsize=PLOT_CONFIG['legend_fontsize'], frameon=PLOT_CONFIG['legend_frameon'], framealpha=PLOT_CONFIG['legend_framealpha'], facecolor=PLOT_CONFIG['legend_facecolor'], edgecolor=PLOT_CONFIG['legend_edgecolor'])
        if leg:
            leg.get_frame().set_linewidth(PLOT_CONFIG['legend_box_linewidth'])
            for text in leg.get_texts(): text.set_fontweight(PLOT_CONFIG['legend_fontweight'])
            
        plt.tight_layout()
        safe_name = str(uid).replace(".", "_")
        plt.savefig(os.path.join(output_fig_folder, f"Well_{safe_name}.png"), dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        plt.close(fig)
        
    except Exception as e:
        print(f"Error processing {uid}: {e}")

# --- 5. SUMMARY & EXPORT ---
if results:
    df_res = pd.DataFrame(results)
    df_res.to_csv(output_csv_path, index=False)
    print("\n" + "="*40)
    print(f"VALIDATION COMPLETED FOR {len(df_res)} WELLS")
    print(f"Results saved to: {output_csv_path}")
    print(f"Mean Correlation (R): {df_res['R'].mean():.3f}")
    if 'STATE' in df_res.columns:
        print("\nState-wise Mean R:")
        print(df_res.groupby('STATE')['R'].mean())
else:
    print("No valid comparisons generated.")