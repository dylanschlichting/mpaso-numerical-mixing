"""
This script calculates the following mixing components from MPAS-O output across
multiple horizontal and vertical resolutions:

- Mnum_salt = 1/V iiint chiSpurSalt dz dA 
- Mphy_salt - 1/V iiint 2*kappa_v * (ds/dz)**2

The outputs are saved as NetCDF files for each resolution/layer combo in:
`./mixing_outputs/`

"""
# Packages 
import numpy as np
import xarray as xr
import os
import warnings
warnings.filterwarnings("ignore")

# Interpolation function for physical mixing
def interp_mphy(ds):
    mphys_interp = 0.5 * (
        ds.chiPhyVerSalt.isel(nVertLevelsP1=slice(0, -1)) +
        ds.chiPhyVerSalt.isel(nVertLevelsP1=slice(1, None))
    )
    mphys_interp = mphys_interp.rename({'nVertLevelsP1': 'nVertLevels'})
    mphy_salt = mphys_interp.assign_coords(nVertLevels=ds.nVertLevels)
    ds['chiPhyVerSalt'] = mphy_salt

# Define resolutions and layer options
# resolutions = [10000, 9000, 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1000, 900, 800, 700, 600,
#               500, 400, 300, 200]  # meters
resolutions = [200]
layers = [50, 100]

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
        decay_file = f'{rootdir}analysis_members/discreteVarianceDecay_{res}m_{layer}_layers.0001-01-01.nc'

        if not all(os.path.exists(f) for f in [output_file, init_file, decay_file]):
            print(f"Skipping resolution {res_key}, layer {layer} due to missing files.")
            continue

        dso = xr.open_dataset(output_file).isel(Time=slice(1, None))
        dsg = xr.open_dataset(init_file)
        dsd = xr.open_dataset(decay_file)

        # Fix Time coordinate naming
        dso['Time'] = dso.xtime
        dsd['Time'] = dsd.xtime

        datasets[res_key][key] = {'output': dso, 'init': dsg, 'decay': dsd}

# Mapping from key to resolution in meters
# res_km_to_dx = {
#     '10km': 10000,
#     '9km': 9000,
#     '8km': 8000,
#     '7km': 7000,
#     '6km': 6000,
#     '5km': 5000,
#     '4km': 4000,
#     '3km': 3000,
#     '2km': 2000,
#     '1km': 1000,
#     '900m': 900,
#     '800m': 800,
#     '700m': 700,
#     '600m': 600,
#     '500m': 500,
#     '400m': 400,
#     '300m': 300,
#     '200m': 200,
# }
res_km_to_dx = {
    # '400m': 400,
    '300m': 300,
    '200m': 200,
}

# Output directory
output_dir = "./mixing_outputs/"
os.makedirs(output_dir, exist_ok=True)

# Loop through all resolutions
for res_key, zdict in datasets.items():
    dx = res_km_to_dx[res_key]
    print(f"\nProcessing resolution: {res_key} ({dx} m)")

    for z_key, ds in zdict.items():
        print(f"  Layer: {z_key}")

        # Output file paths
        num_path = os.path.join(output_dir, f"mnum_salt_vavg_{res_key}_{z_key}.npy")
        phy_path = os.path.join(output_dir, f"mphy_salt_vavg_{res_key}_{z_key}.npy")

        # Skip if already exists
        if os.path.exists(num_path) and os.path.exists(phy_path):
            print(f"    Skipping existing files")
            continue

        # Load datasets
        dsg = ds['init']
        dso = ds['output']
        dsd = ds['decay']

        # Interpolate physical mixing
        interp_mphy(dsd)

        # Create mask to exclude outer cells
        xcell = dsg.xCell.values
        ycell = dsg.yCell.values

        # Extract variables
        area_cell = dsg.areaCell#.isel(nCells=idx)
        layer_thickness = dso.layerThickness#.isel(nCells=idx)

        # Total volume
        V = (layer_thickness * area_cell).sum(dim=["nCells", "nVertLevels"])

        # Numerical mixing
        mnum_salt_dv = (1/V)*((
            dsd.chiSpurSaltBR08 *
            layer_thickness *
            area_cell
        ).sum(dim=['nVertLevels', 'nCells']))

        # Physical mixing
        mphy_salt_dv = (1/V)*((
            dsd.chiPhyVerSalt *
            layer_thickness *
            area_cell
        ).sum(dim=['nVertLevels', 'nCells']))

        # Save results
        np.save(num_path, mnum_salt_dv.values)
        np.save(phy_path, mphy_salt_dv.values)
        print(f"    Saved: {num_path}, {phy_path}")
