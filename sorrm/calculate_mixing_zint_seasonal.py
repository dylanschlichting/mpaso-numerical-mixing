"""
Calculates the depth-integrated physical 
and numerical mixing for the active tracers from a fully
coupled SORRM simulation over each season
"""

import numpy as np
import xarray as xr
import os
import glob

# initial condition file
path = '/global/cfs/cdirs/e3sm/inputdata/ocn/mpas-o/SOwISC12to30E3r3/mpaso.SOwISC12to30E3r3.20240829.nc'
dsm = xr.open_dataset(path)
# monthly averages
rundir = '/global/cfs/cdirs/m3780/dylan617/20260318.v3.SORRME3r3.CRYO2010.bluepulse-beta1-dvd.perlmutter/run/'
path = '20260318.v3.SORRME3r3.CRYO2010.bluepulse-beta1-dvd.perlmutter.mpaso.hist.am.timeSeriesStatsMonthly.*.nc'
ds = xr.open_mfdataset(glob.glob(rundir+path),
                       combine='nested',
                       concat_dim='Time')
ds = ds.rename({
    var: var.removeprefix("timeMonthly_avg_")
    for var in ds.data_vars
})
ds = ds.sortby(ds.xtime)
ds['Time'] = ds.xtime

import os

print("Loading mixing variables...")

mnum_temp = ds.chiSpurTracerBR08_chiSpurTempBR08
mnum_salt = ds.chiSpurTracerBR08_chiSpurSaltBR08

mphy_temp = ds.chiPhyVerTracer_chiPhyVerTemp
mphy_salt = ds.chiPhyVerTracer_chiPhyVerSalt

dz = ds.layerThickness

# Interpolate physical mixing from n+1 to n layers
mphy_temp = mphy_temp.rename({"nVertLevelsP1": "nVertLevels"})
mphy_salt = mphy_salt.rename({"nVertLevelsP1": "nVertLevels"})

mphy_temp = 0.5 * (
    mphy_temp.isel(nVertLevels=slice(None, -1))
    + mphy_temp.isel(nVertLevels=slice(1, None))
)

mphy_salt = 0.5 * (
    mphy_salt.isel(nVertLevels=slice(None, -1))
    + mphy_salt.isel(nVertLevels=slice(1, None))
)

print("Physical mixing interpolation complete.")

# Compute vertically integrated mixing
print("Computing vertically integrated numerical temperature mixing...")

mnum_temp_zint = (
    (mnum_temp * dz)
    .sum("nVertLevels", skipna=True)
    .groupby("Time.season")
    .mean("Time")
    .compute()
)

print("Numerical temperature mixing complete.")
print("Computing vertically integrated numerical salinity mixing...")

mnum_salt_zint = (
    (mnum_salt * dz)
    .sum("nVertLevels", skipna=True)
    .groupby("Time.season")
    .mean("Time")
    .compute()
)

print("Numerical salinity mixing complete.")
print("Computing vertically integrated physical temperature mixing...")

mphy_temp_zint = (
    (mphy_temp * dz)
    .sum("nVertLevels", skipna=True)
    .groupby("Time.season")
    .mean("Time")
    .compute()
)

print("Physical temperature mixing complete.")
print("Computing vertically integrated physical salinity mixing...")

mphy_salt_zint = (
    (mphy_salt * dz)
    .sum("nVertLevels", skipna=True)
    .groupby("Time.season")
    .mean("Time")
    .compute()
)

print("Physical salinity mixing complete.")

# Save each field individually to avoid memory errors on a login node
output_dir = "/pscratch/sd/d/dylan617"
os.makedirs(output_dir, exist_ok=True)

print("\nSaving vertically integrated mixing fields...")

mnum_temp_zint.to_netcdf(
    os.path.join(output_dir, "mnum_temp_vertically_integrated.nc"),
    format="NETCDF4"
)

print("Saved numerical temperature mixing.")

mnum_salt_zint.to_netcdf(
    os.path.join(output_dir, "mnum_salt_vertically_integrated.nc"),
    format="NETCDF4"
)

print("Saved numerical salinity mixing.")

mphy_temp_zint.to_netcdf(
    os.path.join(output_dir, "mphy_temp_vertically_integrated.nc"),
    format="NETCDF4"
)

print("Saved physical temperature mixing.")

mphy_salt_zint.to_netcdf(
    os.path.join(output_dir, "mphy_salt_vertically_integrated.nc"),
    format="NETCDF4"
)

print("Saved physical salinity mixing.")

print("\nAll files saved to:")
print(output_dir)