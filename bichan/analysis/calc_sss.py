"""
This script calculates the following components from MPAS-O output across
multiple horizontal and vertical resolutions:

- sea surface salinity, a proxy for frontal strength

The outputs are saved as NetCDF files for each resolution/layer combo in:
`./sss_outputs/`

"""
# Packages 
import numpy as np
import xarray as xr
import os
import warnings
warnings.filterwarnings("ignore")

# Define resolutions and layer options
resolutions = [10000, 5000, 2000, 1000, 500]  # meters
layers = [50]

# Base path template
base_path_template = '/pscratch/sd/d/dylan617/bichan/mpaso/new_hres/{res_km}/'

# Store datasets in nested dictionary
datasets = {}

for res in resolutions:
    res_km = res / 1000
    if res_km >= 1.0:
        res_dir = f'{int(res_km)}km'
        res_key = f'{int(res_km)}km'
    else:
        res_dir = f'{res}m'
        res_key = f'{res}m'

    datasets[res_key] = {}
    rootdir = base_path_template.format(res_km=res_dir)

    for layer in layers:
        key = f'z{layer}'
        output_file = f'{rootdir}output_{res}m_{layer}_layers.nc'
        init_file = f'{rootdir}channel_{res}m_{layer}_layers_init.nc'

        if not all(os.path.exists(f) for f in [output_file, init_file, decay_file]):
            print(f"Skipping resolution {res_key}, layer {layer} due to missing files.")
            continue

        dso = xr.open_dataset(output_file).isel(Time=slice(1, None))
        dsg = xr.open_dataset(init_file)

        # Fix Time coordinate naming
        dso['Time'] = dso.xtime
        dsd['Time'] = dsd.xtime

        datasets[res_key][key] = {'output': dso, 'init': dsg, 'decay': dsd}

# Mapping from key to resolution in meters
res_km_to_dx = {
    '10km': 10000,
    '5km': 5000,
    '1km': 1000,
    '500m': 500,
}

# Output directory
output_dir = "./sss_outputs/"
os.makedirs(output_dir, exist_ok=True)

# Loop through all resolutions
for res_key, zdict in datasets.items():
    dx = res_km_to_dx[res_key]
    print(f"\nProcessing resolution: {res_key} ({dx} m)")

    for z_key, ds in zdict.items():
        print(f"  Layer: {z_key}")

        # Output file paths
        sss_path = os.path.join(output_dir, f"sss_{res_key}_{z_key}.npy")

        # Skip if already exists
        if os.path.exists(sss_path):
            print(f"    Skipping existing files")
            continue

        # Load datasets
        dsg = ds['init']
        dso = ds['output']

        sss = dso.salinity.isel(nVertLevels=0).mean('nCells')

        # Save results
        np.save(tot_path, sss.values)
        print(f"    Saved: {tot_path}")
