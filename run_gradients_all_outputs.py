import os
import subprocess
import argparse
import xarray as xr

print('script started')

filepath = os.getcwd() + '/'
cwd = os.path.basename(os.getcwd())

# make command line input of JOBID a variable to specify which files to look at
if (__name__ =="__main__"):
    parser = argparse.ArgumentParser(description="Script that calculates hgrads for multiple output members")
    
    parser.add_argument("--JOBID",required=True, type=int)
    args = parser.parse_args()

    JOBID = str(args.JOBID)

    # divide output file into multiple .nc files based on time
    def ind_to_days(index):
        return int(index/12)

    def smaller_time_intervals(JOBID):
        ds = xr.open_dataset(filepath + f'output_member_{JOBID}.nc')
        ds1 = ds.isel(Time=slice(0,100))
        ds2 = ds.isel(Time=slice(100,200))
        ds3 = ds.isel(Time=slice(200,300))
        ds4 = ds.isel(Time=slice(300,362))
        ds1.to_netcdf(filepath + f'output_member_{JOBID}_days{ind_to_days(0)}-{ind_to_days(100)}.nc', mode = 'w', format='NETCDF4')
        ds2.to_netcdf(filepath + f'output_member_{JOBID}_days{ind_to_days(100)}-{ind_to_days(200)}.nc', mode = 'w', format='NETCDF4')
        ds3.to_netcdf(filepath + f'output_member_{JOBID}_days{ind_to_days(200)}-{ind_to_days(300)}.nc', mode = 'w', format='NETCDF4')
        ds4.to_netcdf(filepath + f'output_member_{JOBID}_days{ind_to_days(300)}-{ind_to_days(362)}.nc', mode = 'w', format='NETCDF4')

    smaller_time_intervals(JOBID)
    print('original output divided by time')

    # run DE's add horizontal gradients script 
    for file in os.listdir('.'):
            if file.startswith('output_member') and JOBID in file and 'days' in file:
                print(f"file found {file}")
                result = subprocess.run(['python3','uv_deriv.py',f'--mesh-file={cwd}_channel_init.nc',f'--flow-file={file}'],capture_output=True,text=True)
                if result.returncode ==0:
                    print(f"{file} processed successfully")
                else:
                    print(f"error processing {file}")
                    print("STDERR:\n", result.stderr)
    
    # recombine files that now contain hgrads
    def recombine(JOBID):
        hgrad_variables = ['dUdx_cell','dUdy_cell','dVdx_cell','dVdy_cell','dSdx_cell','dSdy_cell']
        ds1 = xr.open_dataset(filepath + f'output_member_{JOBID}_days0-8.nc')[hgrad_variables]
        ds2 = xr.open_dataset(filepath + f'output_member_{JOBID}_days8-16.nc')[hgrad_variables]
        ds3 = xr.open_dataset(filepath + f'output_member_{JOBID}_days16-25.nc')[hgrad_variables]
        ds4 = xr.open_dataset(filepath + f'output_member_{JOBID}_days25-30.nc')[hgrad_variables]
    
        ds = xr.concat([ds1,ds2,ds3,ds4],dim='Time')
        ds.to_netcdf(filepath + f'output_member_{JOBID}_hgrads.nc')
    
    recombine(JOBID)
    print('hgrad dataset created')
