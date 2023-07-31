"""
Place in folder with blob2d, b2d.template, a folder called blobDir and possibly a decoder
Running will run the SC campaign on blob2d as defined below
"""

import easyvvuq as uq
import numpy as np
import chaospy as cp
import os
import matplotlib.pyplot as plt
from easyvvuq.actions import CreateRunDirectory, Encode, Decode, CleanUp, ExecuteLocal, Actions

class b2dDecoder:
    """
    Custom decoder for blob2d output.

    Parameters
    ----------
    target_filename: str
        Filename to decode.
    ouput_columns: list
        A list of column names that will be selected to appear in the output.
    """
    
    def __init__(self, target_filename=None, output_columns=None):
    def __init__(self, target_filename, output_columns):
        if len(output_columns) == 0:
            msg = "output_columns cannot be empty."
            logger.error(msg)
            raise RuntimeError(msg)
        self.target_filename = target_filename
        self.output_columns = output_columns
        self.output_type = OutputType('sample')
    
    def getBlobVelocity():
        import numpy as np
        from xbout import open_boutdataset
        
        ds = open_boutdataset(chunks={"t": 4})
        ds = ds.squeeze(drop=True)
        dx = ds["dx"].isel(x=0).values
        ds = ds.drop("x")
        ds = ds.assign_coords(x=np.arange(ds.sizes["x"])*dx)
        
        background_density = 1.0
        ds["delta_n"] = ds["n"] - background_density
        integrated_density = ds.bout.integrate_midpoints("delta_n")
        ds["delta_n*x"] = ds["delta_n"] * ds["x"]
        ds["CoM_x"] = ds.bout.integrate_midpoints("delta_n*x") / integrated_density
        v_x = ds["CoM_x"].differentiate("t")
        
        return max(v_x)# Maybe redefine, this will do for now
    
    def parse_sim_output(self, run_info={}):
        return {}

def refine_sampling_plan(number_of_refinements):
        """
        Refine the sampling plan.

        Parameters
        ----------
        number_of_refinements (int)
           The number of refinement iterations that must be performed.

        Returns
        -------
        None. The new accepted indices are stored in analysis.l_norm and the admissible indices
        in sampler.admissible_idx.
        """
        for i in range(number_of_refinements):
            # compute the admissible indices
            sampler.look_ahead(analysis.l_norm)

            # run the ensemble
            campaign.execute().collate(progress_bar=True)

            # accept one of the multi indices of the new admissible set
            data_frame = campaign.get_collation_result()
            analysis.adapt_dimension('f', data_frame)
            #analysis.adapt_dimension('f', data_frame, method='var')

# Define parameters & whoch are uncertain
vary = {
    "height": cp.Normal(0.5, 0.1),
    "width": cp.Normal(0.9, 0.02)# Different distribution?
}
params = {
    "Te0": {"type": "float", "default": 5.0},
    "n0": {"type": "float", "default": 2.0e+18},
    "D_vort": {"type": "float", "default": 1.0e-6},
    "D_n": {"type": "float", "default": 1.0e-6},
    "height": {"type": "float", "min": 0.25, "max": 0.75, "default": 0.5},
    "width": {"type": "float", "min": 0.03, "max": 0.15, "default": 0.09},
    
    "outfile": {"type": "string", "default": "b2dout.csv"},####################################################
    "d": {"type": "integer", "default": len(vary)}
}

# Note output filename and output value name
output_filename = params["outfile"]["default"]
output_columns = ["blobVel"]

# Create encoder, decoder & executor
encoder = uq.encoders.GenericEncoder(
    template_fname='b2d.template',
    delimiter='$',
    target_filename='blobDir/BOUT.inp')
execute = ExecuteLocal('mpirun -np 4 {}/blob2d -d blobDir nout=10'.format(os.getcwd()))# Add more timesteps later (worth mpi at all?)#########
decoder = b2dDecoder(
        target_filename=output_filename,
        output_columns=output_columns)
actions = Actions(CreateRunDirectory('/tmp'), Encode(encoder), execute, Decode(decoder))

# Create campaign and sampler
campaign = uq.Campaign(name='sc_adaptive', work_dir='/tmp', params=params, actions=actions)
sampler = uq.sampling.SCSampler(
        vary=vary,
        polynomial_order=1,
        quadrature_rule="C",
        sparse=True,
        growth=True,
        midpoint_level1=True,
        dimension_adaptive=True)

# Run campaign
campaign.set_sampler(sampler)
campaign.execute().collate(progress_bar=True)
