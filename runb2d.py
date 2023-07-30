"""
This will take a json file containing a parameter set for blob2d, run blob2d,
do analysis on the output to return the outputs we want, then save it to b2dout.csv
"""

import sys
import json
from pathlib import Path
import urllib.request
import numpy as np
from xbout import open_boutdataset

# Get input parameters for blob2d
json_input = sys.argv[1]
#with open(json_input, "r") as f:
#    inputs = json.load(f)
inputs = json.load(open(json_input, "r"))
output_filename = inputs['outfile']

# Get number of uncertain parameters
d = inputs['d']

"""
Run blob 2D (identify sim by parameters)
"""

# Load blob2d output
ds = open_boutdataset(chunks={"t":8})
ds = ds.squeeze(drop=True)
dx = ds["dx"].isel(x=0).values
ds = ds.drop("x")
ds = ds.assign_coords(x=np.arange(ds.sizes["x"])*dx)

# Extract important data
background_density = 1.0
ds["delta_n"] = ds["n"] - background_density
integrated_density = ds.bout.integrate_midpoints("delta_n")
ds["delta_n*x"] = ds["delta_n"] * ds["x"]
ds["delta_n*z"] = ds["delta_n"] * (ds["z"] - 38.4)
ds["CoM_x"] = ds.bout.integrate_midpoints("delta_n*x") / integrated_density
ds["CoM_z"] = ds.bout.integrate_midpoints("delta_n*z") / integrated_density
ds2 = ds.isel(t=slice(4))
v_x = ds["CoM_x"].differentiate("t")
v_z = ds["CoM_z"].differentiate("t")

# Define blob velocity
blobVel = max(v_x)# Maybe redefine, this will do for now

# output csv file
header = 'blobVel'
np.savetxt(
        output_filename,
        np.array([blobVel]),
        delimiter=",",
        comments='',
        header=header)
