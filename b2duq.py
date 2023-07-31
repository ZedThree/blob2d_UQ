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

import os
from string import Template
import logging

class b2dEncoder:
    """Encoder for blob2d.

    Parameters
    ----------

    Attributes
    ----------

    """

    def __init__(self, template_fname, delimiter='$', target_filename="app_input.txt"):
        self.delimiter = delimiter
        self.target_filename = target_filename
        self.template_fname = template_fname

    def encode(self, params={}, target_dir=''):
        """Substitutes `params` into a template application input, saves in
        `target_dir`

        Parameters
        ----------
        params        : dict
            Parameter information in dictionary.
        target_dir    : str
            Path to directory where application input will be written.
        """

        try:
            with open(self.template_fname, 'r') as template_file:
                template_txt = template_file.read()
                self.template = Template(template_txt)
        except FileNotFoundError:
            raise RuntimeError(
                "the template file specified ({}) does not exist".format(self.template_fname))

        if not target_dir:
            raise RuntimeError('No target directory specified to encoder')

        str_params = {}
        for key, value in params.items():
            str_params[key] = str(value)

        try:
            app_input_txt = self.template.substitute(str_params)
        except KeyError as e:
            self._log_substitution_failure(e)

        # Write target input file
        target_file_path = os.path.join(target_dir, self.target_filename)
        os.mkdir(os.path.join(target_dir, "blobDir"))
        with open(target_file_path, 'w') as fp:
            fp.write(app_input_txt)

    def _log_substitution_failure(self, exception):
        reasoning = (f"\nFailed substituting into template "
                     f"{self.template_fname}.\n"
                     f"KeyError: {str(exception)}.\n")
        logging.error(reasoning)

        raise KeyError(reasoning)

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
    
    def __init__(self, target_filename, output_columns):
        # target_filename is a folder, but can't change variable name for compatibility with easyvvuq
        from easyvvuq import OutputType
        
        if len(output_columns) == 0:
            msg = "output_columns cannot be empty."
            logger.error(msg)
            raise RuntimeError(msg)
        self.target_filename = target_filename
        self.output_columns = output_columns
        self.output_type = OutputType('sample')
        
    #def _get_output_path(run_info=None, outfile=None):
    #    run_path = run_info['run_dir']
    #    if not os.path.isdir(run_path):
    #        raise RuntimeError(f"Run directory does not exist: {run_path}")
    #    return os.path.join(run_path, outfile)
    
    def getBlobVelocity(self, out_path):
        import numpy as np
        from xbout import open_boutdataset
        
        os.chdir(out_path)#################################### Need to unset this later or not?
        
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
        
        return {"maxV": [float(max(v_x))]}# Maybe redefine, this will do for now - float or other datatype?  eg numpy
    
    def parse_sim_output(self, run_info={}):
        #out_path = self._get_output_path(run_info, self.target_filename)
        out_path = os.path.join(run_info['run_dir'], self.target_filename)
        blobVels = self.getBlobVelocity(out_path)
        return blobVels

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
    "width": cp.Normal(0.09, 0.02)# Different distribution?
}
params = {
    "Te0": {"type": "float", "default": 5.0},
    "n0": {"type": "float", "default": 2.0e+18},
    "D_vort": {"type": "float", "default": 1.0e-6},
    "D_n": {"type": "float", "default": 1.0e-6},
    "height": {"type": "float", "min": 0.25, "max": 0.75, "default": 0.5},
    "width": {"type": "float", "min": 0.03, "max": 0.15, "default": 0.09},
    
    "outfolder": {"type": "string", "default": "blobDir/"},
    "d": {"type": "integer", "default": len(vary)}
}

# Note output filename and output value name
output_folder = params["outfolder"]["default"]
output_columns = ["maxV"]

# Create encoder, decoder & executor
encoder = b2dEncoder(
    template_fname='b2d.template',
    delimiter='$',
    target_filename='blobDir/BOUT.inp')
execute = ExecuteLocal('mpirun -np 4 {}/blob2d -d blobDir nout=6'.format(os.getcwd()))# Add more timesteps later (worth mpi at all?)#########
decoder = b2dDecoder(
        target_filename=output_folder,
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
