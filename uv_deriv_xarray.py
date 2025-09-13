
import time
import numpy as np
import xarray as xr
import argparse

from msh import load_mesh
from ops import operators

#-- A quick hack to inject the velocity gradient tensor into an
#-- MPAS-O output file

#-- Authors: Darren Engwirda

if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "--mesh-file", dest="mesh_file", type=str,
        required=True, help="Path to user mesh file.")
        
    parser.add_argument(
        "--flow-file", dest="flow_file", type=str,
        required=True, help="Path to user flow file.")
        
    args = parser.parse_args()
  
    # MPASO mesh as a python-ish data-structure  
    print("Loading input assets...")
    mesh = load_mesh(args.mesh_file)

    # spatial discretisation as sparse matrices 
    print("Forming coefficients...")
    mats = operators(mesh)

    # setup space in the output file for tensor
    ttic = time.time()
    print("Growing NetCDF files...")
    flow = xr.open_dataset(args.flow_file)

    ncel = int(flow.dims["nCells"])
    nlev = int(flow.dims["nVertLevels"])
    ndts = int(flow.dims["Time"])

    if ("Ux_cell" not in flow.variables.keys()):
        flow["Ux_cell"] = (
            ("Time", "nCells", "nVertLevels"), 
            np.empty((ndts, ncel, nlev), dtype=np.float32))
        flow["Ux_cell"].attrs["long_name"] = \
            "cartesian x-axis velocity, remapped to cells"
            
    if ("Vy_cell" not in flow.variables.keys()):
        flow["Vy_cell"] = (
            ("Time", "nCells", "nVertLevels"),
            np.empty((ndts, ncel, nlev), dtype=np.float32))
        flow["Vy_cell"].attrs["long_name"] = \
            "cartesian y-axis velocity, remapped to cells"
            
    if ("dUdx_cell" not in flow.variables.keys()):
        flow["dUdx_cell"] = (
            ("Time", "nCells", "nVertLevels"),
            np.empty((ndts, ncel, nlev), dtype=np.float32))
        flow["dUdx_cell"].attrs["long_name"] = \
            "cartesian dU/dx, remapped to cells"
            
    if ("dUdy_cell" not in flow.variables.keys()):
        flow["dUdy_cell"] = (
            ("Time", "nCells", "nVertLevels"),
            np.empty((ndts, ncel, nlev), dtype=np.float32))
        flow["dUdy_cell"].attrs["long_name"] = \
            "cartesian dU/dy, remapped to cells"
            
    if ("dVdx_cell" not in flow.variables.keys()):
        flow["dVdx_cell"] = (
            ("Time", "nCells", "nVertLevels"),
            np.empty((ndts, ncel, nlev), dtype=np.float32))
        flow["dVdx_cell"].attrs["long_name"] = \
            "cartesian dV/dx, remapped to cells"
            
    if ("dVdy_cell" not in flow.variables.keys()):
        flow["dVdy_cell"] = (
            ("Time", "nCells", "nVertLevels"),
            np.empty((ndts, ncel, nlev), dtype=np.float32))
        flow["dVdy_cell"].attrs["long_name"] = \
            "cartesian dV/dy, remapped to cells"

    ttoc = time.time()
    #print(ttoc - ttic)

    # loop through steps in file and LSQR reconstruct tensor from
    # unstructured velocities

    print("Computing vel tensor...")

    ttic = time.time()

    klev = 0  # only do the top layer

    for step in range(ndts):

        uu_edge = flow["normalVelocity"][step, :, klev]
        
        # remap edge normal vel. to cells
        Ux_cell = mats.cell_lsqr_xnrm * uu_edge
        Vy_cell = mats.cell_lsqr_ynrm * uu_edge
        
        flow["Ux_cell"][step, :, klev] = Ux_cell
        flow["Vy_cell"][step, :, klev] = Vy_cell
        
        # take normal gradient: d/dn(Ux)
        dUdn_edge = mats.edge_grad_norm * Ux_cell
        # and remap to cells
        dUdx_cell = mats.cell_lsqr_xnrm * dUdn_edge
        dUdy_cell = mats.cell_lsqr_ynrm * dUdn_edge
        
        flow["dUdx_cell"][step, :, klev] = dUdx_cell
        flow["dUdy_cell"][step, :, klev] = dUdy_cell
        
        # take normal gradient: d/dn(Vy)
        dVdn_edge = mats.edge_grad_norm * Vy_cell
        # and remap to cells
        dVdx_cell = mats.cell_lsqr_xnrm * dVdn_edge
        dVdy_cell = mats.cell_lsqr_ynrm * dVdn_edge
    
        flow["dVdx_cell"][step, :, klev] = dVdx_cell
        flow["dVdy_cell"][step, :, klev] = dVdy_cell
    
        """
        # it's also possible to remap elsewhere on the mesh...
        # remap edge normal vel. to duals
        Ux_dual = mats.dual_lsqr_xnrm * uu_edge
        Vy_dual = mats.dual_lsqr_ynrm * uu_edge
        
        flow["Ux_dual"][step, :, klev] = Ux_dual
        flow["Vy_dual"][step, :, klev] = Vy_dual
        """
    
    ttoc = time.time()
    #print(ttoc - ttic)

    print("Saving new flow file...")

    flow.to_netcdf(args.flow_file, format="NETCDF4")
    
