"""
This script calculates the following mixing components from MPAS-O output across
multiple horizontal and vertical resolutions using wildcards:

- Mnum_salt = 1/V \iiint chiSpurSalt dz dA 
- Mphy_salt = 1/V \iiint 2*kappa_v * (ds/dz)**2 dz dA

The outputs are saved as NumPy files for each resolution/layer combo in:
`./mixing_outputs/`
"""

import numpy as np
import xarray as xr
import os
import glob
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

# Config
resolutions = [100]
layers = [50]
save_every = 12  # Save results every 12 time steps

# Base path with wildcard pattern
base_path_template = '/global/cfs/projectdirs/m4572/dylan617/bichan/{res_km}/'

# Dataset container
datasets = {}

for res in resolutions:
    res_km = f"{int(res/1000)}km" if res >= 1000 else f"{res}m"
    datasets[res_km] = {}

    rootdir = base_path_template.format(res_km=res_km)

    for layer in layers:
        key = f'z{layer}'
        file_glob_output = os.path.join(rootdir, f"output_{res_km}_{layer}_layers_days_*.nc")
        file_glob_decay = os.path.join(rootdir, f"analysis_members/discreteVarianceDecay_{res_km}_{layer}_layers_days_*.nc")
        init_dir = f"/pscratch/sd/d/dylan617/bichan/mpaso/new_hres/{res_km}/"
        file_glob_init = os.path.join(init_dir, f"channel_{res_km}_{layer}_layers_init.nc")

        output_files = sorted(glob.glob(file_glob_output))
        decay_files = sorted(glob.glob(file_glob_decay))
        init_file = file_glob_init  # Usually only one init file

        if not output_files or not decay_files or not os.path.exists(init_file):
            print(f"Skipping {res_km}, layer {layer}: missing files.")
            continue

        dso = xr.open_mfdataset(output_files, combine='nested', concat_dim='Time',
                                chunks={"Time": save_every, "nCells": 500000})
        dsd = xr.open_mfdataset(decay_files, combine='nested', concat_dim='Time',
                                chunks={"Time": save_every, "nCells": 500000})
        dsg = xr.open_dataset(init_file)

        # Assign proper time coordinate
        if 'xtime' in dso:
            dso = dso.assign_coords(Time=dso.xtime)
        if 'xtime' in dsd:
            dsd = dsd.assign_coords(Time=dsd.xtime)

        # Drop duplicate times (if any)
        if not dso.Time.to_index().is_unique:
            dso = dso.sel(Time=~dso.Time.to_index().duplicated())
        if not dsd.Time.to_index().is_unique:
            dsd = dsd.sel(Time=~dsd.Time.to_index().duplicated())

        datasets[res_km][key] = {'output': dso, 'init': dsg, 'decay': dsd}

# Resolution mapping
res_km_to_dx = {'100m': 100}  # Extend if needed

# Output directory
output_dir = "./mixing_outputs/"
os.makedirs(output_dir, exist_ok=True)

# Processing loop
for res_key, zdict in datasets.items():
    dx = res_km_to_dx[res_key]
    print(f"\nProcessing resolution: {res_key} ({dx} m)")

    for z_key, ds in zdict.items():
        print(f"  Layer: {z_key}")

        dsg = ds['init']
        dso = ds['output']
        dsd = ds['decay']

        interp_mphy(dsd)  # Interpolate physical mixing

        area_cell = dsg.areaCell
        layer_thickness = dso.layerThickness
        nt = dsd.dims['Time']

        for t_start in range(0, nt, save_every):
            t_end = min(t_start + save_every, nt)

            # Align after slicing to avoid ValueError
            dsd_sub, dso_sub = xr.align(
                dsd.isel(Time=slice(t_start, t_end)),
                dso.isel(Time=slice(t_start, t_end)),
                join='inner'
            )

            # Re-slice layer thickness (based on aligned dso_sub)
            layer_thick_sub = dso_sub.layerThickness

            # Total volume
            V = (layer_thick_sub * area_cell).sum(dim=["nCells", "nVertLevels"])

            # Numerical mixing
            mnum_salt = (1 / V) * (
                dsd_sub.chiSpurSaltBR08 *
                layer_thick_sub *
                area_cell
            ).sum(dim=['nVertLevels', 'nCells'])

            # Physical mixing
            mphy_salt = (1 / V) * (
                dsd_sub.chiPhyVerSalt *
                layer_thick_sub *
                area_cell
            ).sum(dim=['nVertLevels', 'nCells'])

            # Save to file
            chunk_str = f"{t_start:04d}_{t_end:04d}"
            num_path = os.path.join(output_dir, f"mnum_salt_vavg_{res_key}_{z_key}_T{chunk_str}.npy")
            phy_path = os.path.join(output_dir, f"mphy_salt_vavg_{res_key}_{z_key}_T{chunk_str}.npy")

            np.save(num_path, mnum_salt)
            np.save(phy_path, mphy_salt)
            print(f"    Saved [{chunk_str}]: {num_path}, {phy_path}")
