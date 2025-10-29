# mpaso-numerical-mixing
This repository contains information to analyze numerical and physical mixing in a suite of idealized and realistic simulations with the Model for Prediction Across Scales. The domains are: 
- 1D linear advection in a doubly periodic domain
- Simulating Ocean Mesoscale Activity (SOMA)
- Submesoscale baroclinic channel (bichan)
- E3SM v3.0 coupled ocean sea ice simulation 

Note: Repository under development.
## Compiling and running cases
All cases are run and analyzed on Perlmutter (PM) at NERSC. Compiling instructions are specific to this machine. Reach out to an E3SM expert if you try to compile MPAS-O on an unsupported machine. The DVD code is stored in my E3SM branch discussed in bichan section. 

### Dependencies
I recommend making forks of E3SM, Compass, and Polaris so that you can easily track changes to source code. 
#### Miniforge
```
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"

bash Miniforge3-$(uname)-$(uname -m).sh

```
It will prompt you with a path to install miconda too. I put it in scratch to avoid memory errors in the home directory
```/pscratch/sd/d/dylan617/miniforge3```
#### Polaris
Bichan is compiled with the master branch of Polaris. To build:
```
git clone git@github.com:E3SM-Project/polaris.git

cd polaris

git submodule update --init --recursive

./conda/configure_polaris_envs.py --conda /pscratch/sd/d/dylan617/miniforge3 -c gnu --mpi mpich -m pm-cpu
```

#### Compass 
SOMA is built with Compass and relies on my personal branch. Compass is used instead of Polaris because at time of this study, SOMA hasn't been ported to Polaris. To build:
```
git remote add dylanschlichting git@github.com:dylanschlichting/compass.git
git fetch dylanschlichting dylanschlichting/ocn/modify_soma
git clone -b dylanschlichting/ocn/modify_soma git@github.com:dylanschlichting/compass.git

git submodule update --init --recursive

./conda/configure_compass_env.py --conda /pscratch/sd/d/dylan617/miniforge3 -c gnu --mpi mpich -m pm-cpu
```

### Bichan 
Bichan is a combination of two git branches, one with DVD + one with modified MPAS-O boundary conditions that includes free slip walls made by Darren Engwirda. 
```
# Branch with DVD 
dylanschlichting/ocn/add-br08-dvd
# Branch with modified wall slip BC, needed for bichan
https://github.com/E3SM-Coastal-Discussion/E3SM/pull/4
```
First, clone my remote branch of E3SM that has the DVD algorithm: 
```
git remote add dylanschlichting https://github.com/dylanschlichting/E3SM.git
git fetch dylanschlichting dylanschlichting/add-br08-dvd
git clone -b dylanschlichting/add-br08-dvd git@github.com:dylanschlichting/E3SM.git

mv E3SM/ bichan

git submodule update --init --recursive
```
The DVD branch should be good go. Grab the commits from Darren's branch second because it is less invasive than DVD:
```
git remote add dengwirda https://github.com/dengwirda/E3SM.git
git fetch dengwirda dengwirda/wall-slip
# Get the commit logs from https://github.com/E3SM-Project/E3SM/compare/master...dengwirda:E3SM:dengwirda/wall-slip, go from oldest to newest
# This one can be copied & pasted
git cherry-pick 74540c09b6c70c76c7b98ef4c43f046dbcbf53e7^..548bdbf8944ccdab6b5d7cd735d1738a694ff3b3
# Remove Darren, we won't need this branch anymore
git remote remove dengwirda
```
A basic check to make sure you have all commits from both branches is to use ```git log```. You can also check for key parts of the code like 
```
alias ggin='echo '\''git grep -in '\''; git grep -in '

# A lot of text will show up here
ggin numericalMixingTracers
ggin wall_slip 
```
To make 

Next, just load your polaris conda environment with ```source path/to/polaris_env.sh```. 

Now we're ready to compile, just run ```make gnu-cray```.

To generate the bichan mesh, initial conditions, and forcing, the main script is:
```
python channel_case.py --help

# Key arguments:
- `--case-name`: Case name
- `--num-xcell`: Number of horizontal cells (e.g., domain length / edge-len)  
- `--num-ycell`: [2*sqrt(3)] * num-xcell. The 2*sqrt(3) is because of hexagons/
- `--len-edges`: Hexagon edge length [m]  
- `--num-layer`: Number of vertical layers
- `--wind-stress`: Zonal wind stress [m^2/s]
```
Since we are running many horizontal and vertical resolutions, we want to automate this. It's done in an easily configurable script 
```
python bichan/gen_inputs/run_channel.py
```
which if formatted like this:
```

# Horizontal resolutions (len-edges)
# hres = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900])
# hres = np.array([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
x_len = 100000  # along-channel distance in meters

# Loop through each resolution
for h in hres:
    num_x = x_len / h
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
```
The 100-300m cases with 100 vertical layers will take a long time (few hours max) and may give you an OOM error. Run it on a compute node if that happens. The script will dump all files into the current directory, but could be modified to go into the case directories easily. 

To run the model, I don't use job scripts for 1-10 km. I recommend doing interactive nodes, with 1 node for 2-10 km and 4 nodes for 500 m to 1 km. 100 m uses 32 nodes, 200 m uses 16 nodes, and 300-400 m use 8 nodes. 

For each resolution, you need to set the graph partition. For example
```
gpmetis graph.info ncores
```
where ncores changes depending on the amount of nodes. You need to change the namelist depending on the processor layout, where 
```
    config_pio_num_iotasks = 1 # Number of nodes
    config_pio_stride = 128 # Number of cores
```
Now, we're ready to run
```
salloc --nodes 1 --qos interactive --time 02:00:00 --constraint cpu --account=m4572
# Set nranks equal to number of cores
srun -n <nranks> ./ocean_model -n namelist.ocean -s streams.ocean
```

### SOMA 
```cd``` to your compass directory. We will setup two vertical grids: ```60layerPHC``` and ```100layerE3SMv1```, for four horizontal resolutions: 4km, 8km, 16km, and 32km:
```
compass setup [-h] [-t PATH] [-n NUM [NUM ...]] [-f FILE] [-m MACH]
              [-w PATH] [-b PATH] [-p PATH] [--suite_name SUITE]

compass setup -t ocean/soma/32km/long -m pm-cpu -w /pscratch/sd/d/dylan617/mpaso_numerical_mixing/soma -p /pscratch/sd/d/dylan617/E3SM/add-br08-dvd/components/mpas-ocean/

compass setup -t ocean/soma/16km/long -m pm-cpu -w /pscratch/sd/d/dylan617/mpaso_numerical_mixing/soma -p /pscratch/sd/d/dylan617/E3SM/add-br08-dvd/components/mpas-ocean/

compass setup -t ocean/soma/8km/long -m pm-cpu -w /pscratch/sd/d/dylan617/mpaso_numerical_mixing/soma -p /pscratch/sd/d/dylan617/E3SM/add-br08-dvd/components/mpas-ocean/

compass setup -t ocean/soma/4km/long -m pm-cpu -w /pscratch/sd/d/dylan617/mpaso_numerical_mixing/soma -p /pscratch/sd/d/dylan617/E3SM/add-br08-dvd/components/mpas-ocean/

compass setup -t ocean/soma/100layerE3SMv1/32km/long -m pm-cpu -w /pscratch/sd/d/dylan617/mpaso_numerical_mixing/soma -p /pscratch/sd/d/dylan617/E3SM/add-br08-dvd/components/mpas-ocean/

compass setup -t ocean/soma/100layerE3SMv1/16km/long -m pm-cpu -w /pscratch/sd/d/dylan617/mpaso_numerical_mixing/soma -p /pscratch/sd/d/dylan617/E3SM/add-br08-dvd/components/mpas-ocean/

compass setup -t ocean/soma/100layerE3SMv1/8km/long -m pm-cpu -w /pscratch/sd/d/dylan617/mpaso_numerical_mixing/soma -p /pscratch/sd/d/dylan617/E3SM/add-br08-dvd/components/mpas-ocean/

compass setup -t ocean/soma/100layerE3SMv1/4km/long -m pm-cpu -w /pscratch/sd/d/dylan617/mpaso_numerical_mixing/soma -p /pscratch/sd/d/dylan617/E3SM/add-br08-dvd/components/mpas-ocean/
```
To generate the initial conditions, request a compute node:
```
salloc --nodes 1 --qos interactive --time 02:00:00 --constraint cpu --account=m4572
```
and load your compass environment. Then go to ```long/initial_state``` for each case and do ```compass run```. To run the model on a processor layout of your choosing, the graph partition namelist needs to be setup as described in the bichan section. Go to ```long/forward``` and then do
```
srun -n <nranks> ./ocean_model -n namelist.ocean -s streams.ocean
```
I prefer to run it explicitly so namelist options are not overwritten here by ```compass run```. amelists for each horizontal resolution, which are the same for both vertical grids, can be copied from this repository in ```soma/gen_inputs/``` into each case. 
