import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 1. FILE CONFIGURATION ---
folder = '/content/drive/MyDrive/CSR_GRACE_RL0603/Masked'
input_nc_path = os.path.join(folder, 'GRACE_Downscaled_GWSA_1km_RF_FINAL_MASKED.nc')
output_plot_path = os.path.join(folder, 'Hovmoller_Diagram_Publication_RF_GLDAS.png')

# --- 2. USER CUSTOMIZATION BLOCK ---
FONT_FAMILY = 'serif'
FONT_NAME = 'Times New Roman'
SIZE_AXIS_LABELS = 16
SIZE_TICK_LABELS = 14
SIZE_COLORBAR_LABEL = 16
SIZE_COLORBAR_TICKS = 14
SPINE_LINEWIDTH = 1.5 
TICK_LINEWIDTH = 1.5 
TICK_LENGTH_MAJOR = 4.0

print("Generating Hovmöller Diagram...")

# 3. Load Data
if not os.path.exists(input_nc_path):
    print(f"ERROR: File not found: {input_nc_path}")
else:
    ds = xr.open_dataset(input_nc_path)
    
    # 4. Calculate Zonal Mean (Average across Longitude)
    print("Calculating Zonal Mean (averaging over longitude)...")
    var_name = 'GWSA' if 'GWSA' in ds else 'lwe_thickness'
    hov_data = ds[var_name].mean(dim='lon')
    
    # 5. Setup Plot
    plt.rcParams['font.family'] = FONT_FAMILY
    plt.rcParams['font.serif'] = [FONT_NAME, 'Times', 'DejaVu Serif', 'serif']
    fig, ax = plt.subplots(figsize=(12, 6), dpi=600)
    
    # 6. Create the Contour Plot
    levels = np.linspace(-25, 25, 21)
    cplot = hov_data.plot.contourf(
        ax=ax,
        x='time',
        y='lat',
        levels=levels,
        cmap='RdBu', 
        extend='both',
        add_colorbar=False 
    )
    
    # 7. Apply Customizations
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LINEWIDTH)
        
    ax.tick_params(
        axis='both',
        which='major',
        labelsize=SIZE_TICK_LABELS,
        width=TICK_LINEWIDTH,
        length=TICK_LENGTH_MAJOR,
        direction='out'
    )
    
    # 8. Formatting Labels
    ax.set_ylabel(r'Latitude ($^\circ$N)', fontsize=SIZE_AXIS_LABELS, fontweight='bold')
    ax.set_xlabel('Year', fontsize=SIZE_AXIS_LABELS, fontweight='bold')
    ax.grid(color='black', linestyle=':', alpha=0.3)
    ax.set_axisbelow(True)
    
    # 9. Custom Colorbar
    cbar = plt.colorbar(cplot, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('Groundwater Anomaly (cm)', fontsize=SIZE_COLORBAR_LABEL, fontweight='bold')
    cbar.ax.tick_params(labelsize=SIZE_COLORBAR_TICKS, width=TICK_LINEWIDTH, length=TICK_LENGTH_MAJOR)
    cbar.outline.set_linewidth(SPINE_LINEWIDTH)
    
    # 10. Save & Show
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=600, bbox_inches='tight')
    print(f"Hovmöller Diagram saved to: {output_plot_path}")
    plt.show()