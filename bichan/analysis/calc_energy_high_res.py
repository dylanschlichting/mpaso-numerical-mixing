import os
import glob
import numpy as np
import xarray as xr

# === Config ===
resolutions = [100]  # meters
layers = [50]
save_every = 3
output_dir = "energy_outputs"
os.makedirs(output_dir, exist_ok=True)

# === Paths ===
init_dir_template = "/pscratch/sd/d/dylan617/bichan/mpaso/new_hres/{res_km}/"
base_output_dir = "/global/cfs/projectdirs/m4572/dylan617/bichan/100m/"

# === Weighted Zonal Mean ===
def weighted_zonal_mean(var, area):
    return (var * area).groupby("yBin").sum("nCells") / area.groupby("yBin").sum("nCells")

# === Loop ===
for res in resolutions:
    res_km = f"{res}m"
    for layer in layers:
        z_key = f'z{layer}'
        rootdir = base_output_dir.format(res_km=res_km)
        init_dir = init_dir_template.format(res_km=res_km)

        # Match only output files (exclude init files)
        output_pattern = os.path.join(rootdir, f"output_{res_km}_{layer}_layers_days_*.nc")
        output_files = sorted(glob.glob(output_pattern))
        init_file = os.path.join(init_dir, f"channel_{res_km}_{layer}_layers_init.nc")

        if not output_files or not os.path.exists(init_file):
            print(f"Skipping {res_km}, z{layer} — missing files.")
            continue

        try:
            # Load with Dask and chunk to fit in memory
            dso = xr.open_mfdataset(output_files, combine='nested', concat_dim='Time',
                                    chunks={"Time": save_every, "nCells": 100000})
            dsg = xr.open_dataset(init_file)

            if 'xtime' in dso:
                dso['Time'] = dso.xtime
            dso = dso.drop_duplicates(dim='Time')

            # Bin yCell
            dx = res
            dy = max(10 * dx, 5000)
            y_vals = dsg["yCell"].values
            y_bins = np.arange(y_vals.min(), y_vals.max() + dy, dy)
            bin_indices = np.digitize(y_vals, y_bins)
            bin_labels = xr.DataArray(bin_indices, dims=["nCells"])

            dso = dso.assign_coords(yBin=bin_labels)
            dsg = dsg.assign_coords(yBin=bin_labels)

            area_cell = dsg["areaCell"]
            layer_thickness = dso["layerThickness"]

            # === Initial MKE for normalization ===
            u0 = dso["velocityZonal"].isel(Time=0)
            v0 = dso["velocityMeridional"].isel(Time=0)
            h0 = layer_thickness.isel(Time=0)

            ubar_init = weighted_zonal_mean(u0, area_cell).load()
            vbar_init = weighted_zonal_mean(v0, area_cell).load()

            ubar_init_cell = ubar_init.sel(yBin=bin_labels)
            vbar_init_cell = vbar_init.sel(yBin=bin_labels)

            mke_init = 0.5 * (ubar_init_cell**2 + vbar_init_cell**2) * h0 * area_cell
            norm_factor = mke_init.sum(dim=["nCells", "nVertLevels"]).compute().item()

            # === Loop over time chunks ===
            nt = dso.sizes["Time"]
            for t_start in range(0, nt, save_every):
                t_end = min(t_start + save_every, nt)
                chunk_str = f"T{t_start:04d}_{t_end:04d}"

                # Define output file paths
                tke_path = os.path.join(output_dir, f"{res_km}_{z_key}_tke_vavg_{chunk_str}.nc")
                mke_path = os.path.join(output_dir, f"{res_km}_{z_key}_mke_vavg_{chunk_str}.nc")
                eke_path = os.path.join(output_dir, f"{res_km}_{z_key}_eke_vavg_{chunk_str}.nc")

                # Skip this chunk if all outputs already exist
                if os.path.exists(tke_path) and os.path.exists(mke_path) and os.path.exists(eke_path):
                    print(f"⏭️ Skipping {chunk_str} — outputs already exist.")
                    continue

                # Slice the data
                ds_sub = dso.isel(Time=slice(t_start, t_end))
                u = ds_sub["velocityZonal"]
                v = ds_sub["velocityMeridional"]
                h = ds_sub["layerThickness"]

                # Zonal means
                ubar = weighted_zonal_mean(u, area_cell)
                vbar = weighted_zonal_mean(v, area_cell)

                ubar_cell = ubar.sel(yBin=bin_labels).transpose("Time", "nVertLevels", "nCells")
                vbar_cell = vbar.sel(yBin=bin_labels).transpose("Time", "nVertLevels", "nCells")

                # Energies
                tke = 0.5 * (u**2 + v**2) * h * area_cell
                mke = 0.5 * (ubar_cell**2 + vbar_cell**2) * h * area_cell
                eke = 0.5 * ((u - ubar_cell)**2 + (v - vbar_cell)**2) * h * area_cell

                tke_ts = tke.sum(dim=["nCells", "nVertLevels"]) / norm_factor
                mke_ts = mke.sum(dim=["nCells", "nVertLevels"]) / norm_factor
                eke_ts = eke.sum(dim=["nCells", "nVertLevels"]) / norm_factor

                # Save each file individually — compute here
                xr.Dataset({"tke": tke_ts}).to_netcdf(tke_path)
                xr.Dataset({"mke": mke_ts}).to_netcdf(mke_path)
                xr.Dataset({"eke": eke_ts}).to_netcdf(eke_path)

                print(f"✅ Saved: [{chunk_str}] → {res_km}_{z_key}")

        except Exception as e:
            print(f"❌ Failed for {res_km}, {z_key}: {e}")
