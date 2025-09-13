import numpy as np
import xarray as xr
import argparse

# functions for interpolate physical mixing, vertical diffusivity, and center vertices

def interp_mphy(dsd):
    # Step 1: Interpolate (average)
    mphys_interp = 0.5 * (
        dsd.chiPhyVerSalt.isel(nVertLevelsP1=slice(0, -1)) +
        dsd.chiPhyVerSalt.isel(nVertLevelsP1=slice(1, None))
    )
    # Step 2: Rename the dimension
    mphys_interp = mphys_interp.rename({'nVertLevelsP1': 'nVertLevels'})
    # Step 3: Assign new coordinates
    mphy_salt = mphys_interp.assign_coords(nVertLevels=dsd.nVertLevels)
    dsd['chiPhyVerSalt'] = mphy_salt

    # Step 1: Interpolate (average)
    mphyt_interp = 0.5 * (
        dsd.chiPhyVerTemp.isel(nVertLevelsP1=slice(0, -1)) +
        dsd.chiPhyVerTemp.isel(nVertLevelsP1=slice(1, None))
    )
    # Step 2: Rename the dimension
    mphyt_interp = mphyt_interp.rename({'nVertLevelsP1': 'nVertLevels'})
    # Step 3: Assign new coordinates
    mphy_temp = mphyt_interp.assign_coords(nVertLevels=dsd.nVertLevels)
    dsd['chiPhyVerTemp'] = mphy_temp

# interpolate the vertical diffusivity at the top of the cell onto cell centers (similar to interp_mphy function at the top of this notebook)
def interp_k_v(dso):
    # Step 1: Interpolate (average)
    k_v_interp = 0.5 * (
        dso.vertDiffTopOfCell.isel(nVertLevelsP1=slice(0, -1)) +
        dso.vertDiffTopOfCell.isel(nVertLevelsP1=slice(1, None))
    )
    # Step 2: Rename the dimension
    k_v_interp = k_v_interp.rename({'nVertLevelsP1': 'nVertLevels'})
    # Step 3: Assign new coordinates
    k_v_coords = k_v_interp.assign_coords(nVertLevels=dso.nVertLevels)
    dso['vertDiffTopOfCell'] = k_v_coords

def center_vertices(dsg, hres):
    # Clean up the indexing so the domain is centered for plotting

    # Original vertices (x,y) arrays
    xv = dsg.xVertex.values
    yv = dsg.yVertex.values

    # Compute the new vertices based on your filtering and offset logic:

    # 1. Extract verticesOnCell (assuming shape: [nCells, maxVerticesPerCell])
    voc = dsg.verticesOnCell.values

    # 2. Build verts as in your snippet (shape: [nCells, maxVerticesPerCell, 2])
    verts = np.dstack((xv[voc - 1], yv[voc - 1]))
    nverts = np.sum(voc != 0, axis=1)
    verts_list = [vert[:n] for vert, n in zip(verts, nverts)]

    # 3. Filter verts using np.ptp
    idx = [np.ptp(vert[:, 0]) < 50000 for vert in verts_list]
    verts_filtered = np.array(verts_list)[idx]

    # 4. Copy and apply offsets
    nuverts = verts_filtered.copy()
    m_dsx = hres
    nuverts[:, :, 0] += m_dsx / 2
    nuverts[:, :, 1] -= m_dsx

    # Now, update dsg's xVertex and yVertex arrays accordingly:
    # But note: verts are grouped by cell, so we need to flatten and assign properly.

    # Because verts_filtered is a filtered subset of verts_list,
    # you should update only those cells where idx is True.

    # Get indices of cells that passed filter
    filtered_cells = np.where(idx)[0]

    # Create copies of xVertex and yVertex to modify
    new_xVertex = xv.copy()
    new_yVertex = yv.copy()

    # Loop over filtered cells and update the corresponding vertex coords
    for cell_i, verts_cell in zip(filtered_cells, nuverts):
        nv = verts_cell.shape[0]  # number of vertices for this cell
        vertex_inds = voc[cell_i, :nv] - 1  # zero-based vertex indices for this cell
        new_xVertex[vertex_inds] = verts_cell[:, 0]
        new_yVertex[vertex_inds] = verts_cell[:, 1]

    # Assign back to dsg (if dsg is an xarray Dataset or DataArray)
    dsg['xVertex'].values = new_xVertex
    dsg['yVertex'].values = new_yVertex


# reformat restart files to eliminate duplicate time values(i.e., where restart files overlap)
def reformat_restart(res_choice):
    filepath = f'/pscratch/sd/k/kuyeda/bichan/mpaso/{res_choice}/'
    dso_all = xr.open_mfdataset(filepath + 'output_member_*.nc',combine='nested',join="outer",concat_dim='Time')[['xtime','Time','layerThickness']]
    dso_all = dso_all.sortby(dso_all.xtime) # Sort by xtime in ascending order
    unique_xtime, unique_indices = np.unique(dso_all.xtime.values, return_index=True)
    # Now, re-index the xarray object based on the unique indices
    dso_unique = dso_all.isel(Time=unique_indices)
    dso = dso_unique.isel(Time=slice(1,361))

    dsd_all = xr.open_mfdataset(filepath + f'analysis_members/{res_choice}_channel_discreteVarianceDecay_member_*.nc',
                                combine='nested',join="outer",concat_dim='Time')
    dsd_all = dsd_all.sortby(dsd_all.xtime) # Sort by xtime in ascending order
    unique_xtime, unique_indices = np.unique(dsd_all.xtime.values, return_index=True)
    # Now, re-index the xarray object based on the unique indiceg
    dsd_unique = dsd_all.isel(Time=unique_indices)
    dsd = dsd_unique.isel(Time=slice(0,361))

    dsg = xr.open_dataset(filepath + f'{res_choice}_channel_init.nc')[['xCell','yCell','xVertex','yVertex','nVertLevels','verticesOnCell','areaCell','nCells']]
    print('time series in one .nc file. dso, dsd, dsg files created.', flush=True)

    return dso,dsd,dsg

# create array of time slices
def time_slices(dso,dsd,num_files):
    dsos = []
    dsds = []
    total_time = len(dso.Time)
    slice_size = int(total_time / num_files)
    print(slice_size)
    for i in range(0,num_files):
        slice_lowerbound = i*slice_size
        slice_upperbound = (i+1) * slice_size
        # print(slice_lowerbound,slice_upperbound)
        dso_slice = dso.isel(Time=slice(slice_lowerbound, slice_upperbound))
        dsos.append(dso_slice)

        dsd_slice = dsd.isel(Time=slice(slice_lowerbound,slice_upperbound))
        dsds.append(dsd_slice)
    print('array of time slices created',flush=True)
    return dsos,dsds


# ## Create netcdf files that just contain volume-weighted numerical and physical mixing with a domain size of 280km
def mixing(dso,dsd,dsg,res_choice_meters,res_choice,slice_num):
    rootdir = '/pscratch/sd/k/kuyeda/bichan/mpaso/280km_domain/'

    interp_mphy(dsd) # Call interpolation of physical mixing
    center_vertices(dsg,res_choice_meters) # Call fixing of vertices, second arg is horiz res in meters
    ycell = dsg.yCell.values
    # Clip values away from walls
    idx = np.where(np.logical_and(ycell>10000, ycell<290000))

    # calculate volume-integrated mixing.
    mnum_salt_dv = (dsd.chiSpurSaltBR08.isel(nCells=slice(idx[0][0], idx[0][-1])) *dso.layerThickness.isel(nCells=slice(idx[0][0], idx[0][-1]))*dsg.areaCell.isel(nCells=slice(idx[0][0], idx[0][-1]))).sum(['nVertLevels', 'nCells']).load()
    mphys_salt_dv = (dsd.chiPhyVerSalt.isel(nCells=slice(idx[0][0], idx[0][-1])) *dso.layerThickness.isel(nCells=slice(idx[0][0], idx[0][-1]))*dsg.areaCell.isel(nCells=slice(idx[0][0], idx[0][-1]))).sum(['nVertLevels', 'nCells']).load()

    # convert to dataset
    mnum_ds = mnum_salt_dv.to_dataset(name = 'int_VchiSpurSaltBR08dV')
    mphys_ds = mphys_salt_dv.to_dataset(name = 'int_VchiPhysSaltBR08dV')
    print(f'slice {slice_num} converted to ds',flush=True)

    # merge with xtimes
    xtimes = dso.xtime.to_dataset(name='xtime')
    mnum_xtime_ds = xr.merge([mnum_ds,xtimes])
    mphys_xtime_ds = xr.merge([mphys_ds,xtimes])


    # convert to netcdf
    mnum_xtime_ds.to_netcdf(rootdir + f'mnums/{res_choice}_mnums_output_member_{slice_num}.nc', mode = 'w', format='NETCDF4')
    mphys_xtime_ds.to_netcdf(rootdir + f'mphys/{res_choice}_mphys_output_member_{slice_num}.nc', mode = 'w', format='NETCDF4')


    print(f'slice {slice_num} mixing saved as .nc',flush=True)


# require variables
if (__name__ =="__main__"):
    parser = argparse.ArgumentParser(description="Script that calculates volume-integrated mixing as a function of time")
    parser.add_argument("--RES_CHOICE",required=True,type=str)
    parser.add_argument("--RES_CHOICE_METERS",required=True,type=int)
    parser.add_argument("--NUM_FILES",required=True,type=int)

    args = parser.parse_args()

    dso,dsd,dsg = reformat_restart(args.RES_CHOICE)
    dsos,dsds = time_slices(dso,dsd,args.NUM_FILES)

    for i in range(1,len(dsos)):
        mixing(dsos[i],dsds[i],dsg,args.RES_CHOICE_METERS,args.RES_CHOICE,i)
