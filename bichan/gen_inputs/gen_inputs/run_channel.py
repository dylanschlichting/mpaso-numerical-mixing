import os
import xarray as xr
import numpy as np
import netCDF4 as nc
import argparse
import datetime
import numpy as np
import subprocess

# Horizontal resolutions (len-edges)
# hres = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900])
#hres = np.array([500, 600, 700, 800, 900])
# hres = np.array([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
#hres = np.array([100])
hres = np.array([600])
y_len = 100000  # along-channel distance in meters

# Loop through each resolution
for h in hres:
    num_x = y_len / h
    num_x = np.ceil(num_x)  # round up

    num_y = 2 * np.sqrt(3) * num_x
    num_y = int(np.ceil(num_y))
    if num_y % 2 != 0:
        num_y += 1 # make divisible by 2

    case_name = f"dx{int(h)}m"

    cmd = [
        "python", "channel_case.py",
        f"--case-name=channel_{h}m_100_layers",
        f"--num-xcell={int(num_x)}",
        f"--num-ycell={int(num_y)}",
        f"--len-edges={int(h)}",
        "--num-layer=100",
        "--wind-stress=0.1"
    ]

    print(f"Running: {case_name} with xcells={num_x}, ycells={num_y}")
    subprocess.run(cmd)
