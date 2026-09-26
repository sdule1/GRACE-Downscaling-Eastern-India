import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIGURATION ---
results_file = '/content/drive/MyDrive/CSR_GRACE_RL0603/Masked/Validation_Results_76Wells_RF.csv'
output_fig = '/content/drive/MyDrive/CSR_GRACE_RL0603/Masked/Fig10_Validation_BoxPlot_Regional_RF.png'

FONT_FAMILY = 'serif'
FONT_NAME = 'Times New Roman'
plt.rcParams['font.family'] = FONT_FAMILY
plt.rcParams['font.serif'] = [FONT_NAME, 'Times', 'DejaVu Serif', 'serif']

if not os.path.exists(results_file):
    print(f"Error: File not found at {results_file}. Please check the path.")
else:
    df = pd.read_csv(results_file)
    plt.figure(figsize=(10, 6.5))
    
    palette = {"WEST BENGAL": "#1f77b4", "BIHAR": "#ff7f0e", "JHARKHAND": "#2ca02c", "Unknown": "gray"}
    
    # 1. BOX PLOT
    ax = sns.boxplot(x='STATE', y='R', data=df,
                     hue='STATE',
                     palette=palette,
                     width=0.5,
                     linewidth=2.5,
                     fliersize=0,
                     legend=False)
                     
    # 2. STRIP PLOT
    sns.stripplot(x='STATE', y='R', data=df,
                  color='black',
                  size=6,
                  alpha=0.6,
                  jitter=0.15)
                  
    # 3. STYLING
    plt.axhline(y=0.5, color='darkgreen', linestyle='--', linewidth=2, alpha=0.8, label='Strong (R=0.5)')
    plt.axhline(y=0.0, color='red', linestyle=':', linewidth=2, alpha=0.6, label='No Correlation')
    
    ax.set_ylabel('Pearson Correlation (R)', fontsize=17, fontweight='bold', labelpad=10)
    ax.set_xlabel('', fontsize=0) 
    
    ax.tick_params(axis='x', labelsize=14, length=4, direction='out', pad=10) 
    ax.tick_params(axis='y', labelsize=14, width=1.5, length=4, direction='out')
    ax.yaxis.grid(True, linestyle='--', alpha=0.6, linewidth=1)
    ax.set_axisbelow(True) 
    
    medians = df.groupby(['STATE'])['R'].median()
    counts = df['STATE'].value_counts()
    
    labels = [item.get_text() for item in ax.get_xticklabels()]
    for i, label in enumerate(labels):
        if label in medians:
            val = medians[label]
            n = counts[label]
            ax.text(i, val + 0.03, f'{val:.2f}',
                    ha='center', va='bottom', fontsize=14, fontweight='bold', color='black',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.5))
            ax.text(i, -0.10, f'(n={n})',
                    ha='center', va='top', fontsize=14, color='#333333',
                    transform=ax.get_xaxis_transform())
                    
    plt.ylim(-0.25, 1.05)
    plt.legend(loc='lower right', frameon=True, fontsize=14, edgecolor='black', borderpad=0.8)
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')
        
    plt.tight_layout()
    plt.savefig(output_fig, dpi=600, bbox_inches='tight')
    print(f"High-Impact Comparison Plot saved to: {output_fig}")
    plt.show()