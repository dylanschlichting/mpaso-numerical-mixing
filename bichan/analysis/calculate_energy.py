"""
This script calculates the following energy components from MPAS-O output across
multiple horizontal and vertical resolutions:

- **TKE**: Total Kinetic Energy
- **MKE**: Mean Kinetic Energy (zonal mean)
- **EKE**: Eddy Kinetic Energy (deviation from zonal mean)

The energies are volume-integrated across the domain using masked data,
binned meridionally, and normalized by the initial MKE.

Formulas used:

TKE = 1/V iiint 0.5 * (u² + v²) dz dA 

MKE = 1/V iiint 0.5 * (ubar² + vbar²) dz dA 

EKE = 1/V iiint 0.5 * ((u - ubar)² + (v - vbar)²) dz dA 

Where:
- u, v     = instantaneous zonal and meridional velocity components
- ubar, vbar    = zonal (y-bin) mean of u and v
- h        = layer thickness
- A        = horizontal cell area
- V        = Bulk Volume

The outputs are saved as NetCDF files for each resolution/layer combo in:
`./energy_outputs/`

"""
import os
import numpy as np
import xarray as xr

# === Configuration ===
# resolutions = [10000, 9000, 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1000]
# resolutions = [900, 800, 700, 600, 500, 400, 300]

# resolutions = []

# layers = [50, 100]
base_path_template = '/pscratch/sd/d/dylan617/bichan/mpaso/new_hres/{res_km}/'

output_dir = "energy_outputs"
os.makedirs(output_dir, exist_ok=True)

# === Processing Loop ===
for res in resolutions:
    dx = res  # resolution in meters

    # Format path and key
    if dx >= 1000:
        res_dir = f"{int(dx / 1000)}km"
        res_key = res_dir
    else:
        res_dir = f"{dx}m"
        res_key = res_dir

    for layer in layers:
        key = f'z{layer}'

        rootdir = base_path_template.format(res_km=res_dir)
        output_file = f'{rootdir}output_{dx}m_{layer}_layers.nc'
        init_file = f'{rootdir}channel_{dx}m_{layer}_layers_init.nc'

        if not all(os.path.exists(f) for f in [output_file, init_file]):
            print(f"Skipping {res_key}, {key} due to missing files.")
            continue

        try:
            # Load datasets
            dso = xr.open_dataset(output_file).isel(Time=slice(1, None))
            dsg = xr.open_dataset(init_file)
            dso['Time'] = dso.xtime  # Fix Time coordinate

            # === Apply spatial mask ===
            # xcell = dsg.xCell.values
            # ycell = dsg.yCell.values

            # mask = np.logical_and.reduce([
            #     xcell > 10000,
            #     xcell < 90000,
            #     ycell > 10000,
            #     ycell < 290000
            # ])
            # idx = np.where(mask)[0]

            # Subset datasets by mask (only on nCells dimension)
            dso_masked = dso#.isel(nCells=idx)
            dsg_masked = dsg#.isel(nCells=idx)

            # === Adaptive bin width ===
            dy = max(10 * dx, 5000)  # At least 5 km bins, or 10x resolution
            y_vals = dsg_masked["yCell"].data
            y_bins = np.arange(y_vals.min(), y_vals.max() + dy, dy)
            bin_indices = np.digitize(y_vals, y_bins)
            bin_labels = xr.DataArray(bin_indices, dims=["nCells"])

            # Assign bin labels
            dso_masked = dso_masked.assign_coords(yBin=bin_labels)
            dsg_masked = dsg_masked.assign_coords(yBin=bin_labels)

            # === Weighted zonal mean function ===
            def weighted_zonal_mean(var, area):
                weighted_sum = (var * area).groupby("yBin").sum("nCells")
                area_sum = area.groupby("yBin").sum("nCells")
                return weighted_sum / area_sum

            u_bar = weighted_zonal_mean(dso_masked["velocityZonal"], dsg_masked["areaCell"])
            v_bar = weighted_zonal_mean(dso_masked["velocityMeridional"], dsg_masked["areaCell"])

            # Broadcast zonal means back to grid
            u_bar_cell = u_bar.sel(yBin=bin_labels).transpose("Time", "nVertLevels", "nCells")
            v_bar_cell = v_bar.sel(yBin=bin_labels).transpose("Time", "nVertLevels", "nCells")

            area_cell = dsg_masked["areaCell"]
            layer_thickness = dso_masked["layerThickness"]

            # === Energy Calculations ===
            tke = 0.5 * (dso_masked["velocityZonal"]**2 + dso_masked["velocityMeridional"]**2) * layer_thickness * area_cell
            mke = 0.5 * (u_bar_cell**2 + v_bar_cell**2) * layer_thickness * area_cell
            eke = 0.5 * ((dso_masked["velocityZonal"] - u_bar_cell)**2 +
                         (dso_masked["velocityMeridional"] - v_bar_cell)**2) * layer_thickness * area_cell

            V = (layer_thickness*area_cell).sum(dim=["nCells", "nVertLevels"])
            
            tke_ts = (tke.sum(dim=["nCells", "nVertLevels"]))
            mke_ts = (mke.sum(dim=["nCells", "nVertLevels"]))
            eke_ts = (eke.sum(dim=["nCells", "nVertLevels"]))

            # Normalize by initial MKE at first time index
            norm_factor = (mke_ts.isel(Time=0))
            tke_ts = tke_ts / norm_factor
            mke_ts = mke_ts / norm_factor
            eke_ts = eke_ts / norm_factor

            # Compute with dask (optional, can also defer)
            # tke_ts = tke_ts.compute()
            # mke_ts = mke_ts.compute()
            # eke_ts = eke_ts.compute()

            # Save immediately for this resolution and layer
            tke_path = os.path.join(output_dir, f"{res_key}_{key}_tke_vavg.nc")
            mke_path = os.path.join(output_dir, f"{res_key}_{key}_mke_vavg.nc")
            eke_path = os.path.join(output_dir, f"{res_key}_{key}_eke_vavg.nc")

            tke_ts.to_netcdf(tke_path)
            mke_ts.to_netcdf(mke_path)
            eke_ts.to_netcdf(eke_path)

            print(f"Saved energy time series for {res_key}, {key}")

        except Exception as e:
            print(f"Failed to process {res_key}, {key}: {e}")
