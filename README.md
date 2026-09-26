# GRACE Downscaling Eastern India

This repository contains the Python code and analytical scripts used in the study: **"Unveiling Localized Groundwater Depletion Hotspots in Eastern India via Physically Consistent Machine Learning Downscaling of GRACE Data"**.

## Project Overview
This repository hosts the computational framework used to downscale GRACE Terrestrial Water Storage (TWS) anomalies from a native resolution of 0.25° to a 1 km hyper-resolution across Eastern India (2003–2023). By coupling machine learning with physical mass-conservation rules, this workflow isolates human-induced abstraction from natural hydro-climatic variability, revealing critical sub-grid groundwater dynamics.

## Contents
* **01_XGBoost_Gap_Filling.py**: Reconstructs inter-mission and instrumental temporal discontinuities using lagged hydro-climatic forcing.
* **02_Spatial_Block_Model_Comparison.py**: Benchmarks model generalizability using a strict spatial block holdout cross-validation strategy to prevent autocorrelation bias.
* **03_RF_Downscaling_Model_Full.py**: The primary Random Forest spatial downscaling architecture, integrating SHAP spatial driver attribution and radiometric mass-conservation correction.
* **04_High_Res_GWSA_Calculation.py**: Isolates 1 km Groundwater Storage Anomalies (GWSA) using physical terrestrial water balance subtractions (GLDAS and TerraClimate).
* **05 to 16_Analytical_and_Validation_Scripts.py**: Comprehensive plotting and validation scripts including:
  * In-situ validation against 76 CGWB observation wells
  * Pre-whitened Mann-Kendall spatiotemporal trend analysis
  * GRACE Drought Severity Index (GRACE-DSI) mapping
  * Physical uncertainty and anthropogenic hotspot isolation maps

## Dependencies
The analytical scripts require the following primary Python libraries:
* `xgboost`
* `scikit-learn`
* `shap`
* `rasterio`
* `geopandas`
* `xarray`
* `pymannkendall`
