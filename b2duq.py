"""
Runs a dimension adaptive stochastic colocation UQ campaign on the blob2d model
"""

import easyvvuq as uq
import numpy as np
import chaospy as cp
import os
import matplotlib.pyplot as plt
from easyvvuq.actions import CreateRunDirectory, Encode, Decode, CleanUp, ExecuteLocal, Actions

class b2dEncoder:
    """
    Encoder for blob2d, just generic decoder with extra line to create the b2d input folder.

    Parameters
    ----------
    Note targetfile/folder thing
    """

    def __init__(self, template_fname, delimiter='$', target_filename="app_input.txt"):
        self.delimiter = delimiter
        self.target_filename = target_filename
        self.template_fname = template_fname

    def encode(self, params={}, target_dir=''):
        """
        Substitutes `params` into a template application input, saves in
        `target_dir`

        Parameters
        ----------
        params        : dict
            Parameter information in dictionary.
        target_dir    : str
            Path to directory where application input will be written.
        """
        
        from string import Template
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
        import logging
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
    
    def getBlobVelocity(self, out_path):
        from xbout import open_boutdataset
        
        # Set working directory to location of b2d output files
        os.chdir(out_path)
        
        # Unpack data from blob2d
        ds = open_boutdataset(chunks={"t": 4})
        ds = ds.squeeze(drop=True)
        dx = ds["dx"].isel(x=0).values
        ds = ds.drop("x")
        ds = ds.assign_coords(x=np.arange(ds.sizes["x"])*dx)
        
        # Obtain blob velocity from data
        background_density = 1.0
        ds["delta_n"] = ds["n"] - background_density
        integrated_density = ds.bout.integrate_midpoints("delta_n")
        ds["delta_n*x"] = ds["delta_n"] * ds["x"]
        ds["CoM_x"] = ds.bout.integrate_midpoints("delta_n*x") / integrated_density
        v_x = ds["CoM_x"].differentiate("t")
        
        return {"maxV": [float(max(v_x))]}# Add other velocity / distance metrics?
    
    def parse_sim_output(self, run_info={}):
        out_path = os.path.join(run_info['run_dir'], self.target_filename)
        blobVels = self.getBlobVelocity(out_path)
        return blobVels

###############################################################################

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
            
###############################################################################

def defineParams(paramFile=None):
    if paramFile == None:
        params = {
                "Te0": {"type": "float", "default": 5.0},
                "n0": {"type": "float", "default": 2.0e+18},
                "D_vort": {"type": "float", "default": 1.0e-6},
                "D_n": {"type": "float", "default": 1.0e-6},
                "height": {"type": "float", "min": 0.25, "max": 0.75, "default": 0.5},
                "width": {"type": "float", "min": 0.03, "max": 0.15, "default": 0.09},
        }
        vary = {
                "height": cp.Normal(0.5, 0.1),
                "width": cp.Normal(0.09, 0.02)# Different distribution?
        }
        
        output_folder = "blobDir/"# Arbitrary but must be defined (location of BOUT.inp & output files)
        output_columns = ["maxV"]
        template = 'b2d.template'
        
        return params, vary, output_folder, output_columns, template
    else:
        # Don't plan to use parameter files but will write in code to do so if time
        pass
        #return params, vary, output_folder, output_columns, template

def setupCampaign(params, output_folder, output_columns, template):
    # Create & package encoder, decoder & executor into actions
    
    encoder = b2dEncoder(
            template_fname=template,
            delimiter='$',
            target_filename='{}BOUT.inp'.format(output_folder))
    
    execute = ExecuteLocal('mpirun -np 4 {}/blob2d -q -d {} nout=6'.format(os.getcwd(), output_folder))# 60+ timesteps resonable
    
    decoder = b2dDecoder(
            target_filename=output_folder,# must use "target_filename" even though a folder for compatibility with executor
            output_columns=output_columns)
    
    actions = Actions(CreateRunDirectory('/tmp'), Encode(encoder), execute, Decode(decoder))

    campaign = uq.Campaign(name='sc_adaptive', work_dir='/tmp', params=params, actions=actions)
    
    return campaign

def setupSampler(vary):
    """
    Creates and returns an easyvvuq sampler object according to the uncertain parameters

    Parameters
    ----------
    vary
        Dictionary of uncertain parameters (subset of params)

    Returns
    -------
    sampler
        Easyvvuq sampler object for an easyvvuq campaign
    """
    
    sampler = uq.sampling.SCSampler(
            vary=vary,
            polynomial_order=1,
            quadrature_rule="C",
            sparse=True,
            growth=True,
            midpoint_level1=True,
            dimension_adaptive=True) 
    
    return sampler

def runCampaign(campaign, sampler):
    campaign.set_sampler(sampler)
    campaign.execute().collate(progress_bar=True)
    print("Campaign run & collated successfuly")

###############################################################################

def main():
    params, vary, output_folder, output_columns, template = defineParams()
    campaign = setupCampaign(params, output_folder, output_columns, template)
    sampler = setupSampler(vary)
    runCampaign(campaign, sampler)

if __name__ == "__main__":
    main()
