"""
Runs a dimension adaptive stochastic colocation UQ campaign on the blob2d model
Should be run with python3 in the same folder as blob2d and a blob2d input template.

Dependencies: easyvvuq-1.2, matplotlib-3.7.2 & xbout-0.3.5.
"""

import easyvvuq as uq
import numpy as np
import chaospy as cp
import os
import matplotlib.pyplot as plt
from easyvvuq.actions import CreateRunDirectory, Encode, Decode, ExecuteLocal, Actions

from easyvvuq import OutputType
from xbout import open_boutdataset

class b2dDecoder:
    """
    Custom decoder for blob2d output.

    Parameters
    ----------
    target_filename (str)
        Name of blob2d output file to be decoded.
    ouput_columns (list)
        List of output quantities to considered by the campaign
    """
    
    def __init__(self, target_filename, output_columns):
        self.target_filename = target_filename
        self.output_columns = output_columns
        self.output_type = OutputType('sample')
    
    def getBlobInfo(self, out_path):    
        """
        Uses xbout to extract the data from blob2d output files and convert to useful quantities.
        
        Parameters
        ----------
        out_path (str)
            Absolute path to the blob2d output files.

        Returns
        -------
        blobInfo (dict)
            Dictionary of quantities which may be called by the campaign.
        """
        
        # Set working directory to location of b2d output files
        os.chdir(out_path)
        
        # Unpack data from blob2d
        ds = open_boutdataset(chunks={"t": 4})
        ds = ds.squeeze(drop=True)
        dx = ds["dx"].isel(x=0).values
        ds = ds.drop("x")
        ds = ds.assign_coords(x=np.arange(ds.sizes["x"])*dx)
        
        # Obtain blob info from data
        blobInfo = {}
        background_density = 1.0
        ds["delta_n"] = ds["n"] - background_density
        integrated_density = ds.bout.integrate_midpoints("delta_n")
        ds["delta_n*x"] = ds["delta_n"] * ds["x"]
        ds["CoM_x"] = ds.bout.integrate_midpoints("delta_n*x") / integrated_density
        v_x = ds["CoM_x"].differentiate("t")
        
        # Save useful quantities to dictionary (add more?)
        maxV = float(max(v_x))
        maxX = float(ds["CoM_x"][list(v_x).index(max(v_x))])
        blobInfo = {"maxV": maxV, "maxX": maxX}
    
        return blobInfo
    
    def parse_sim_output(self, run_info={}):
        """
        Parses a BOUT.dmp.*.nc file from the output of blob2d and converts it to the EasyVVUQ
        internal dictionary based format.  The file is parsed in such a way that each column
        appears as a vector QoI in the output dictionary.

        E.g. if the file contains the LHS and `a` & `b` are specified as `output_columns` then:
        a,b
        1,2  >>>  {'a': [1, 3], 'b': [2, 4]}.
        3,4

        Parameters
        ----------
        run_info: dict
            Information about the run used to construct the absolute path to
            the blob2d output files.
        
        Returns
        -------
        outQtts (dict)
            Dictionary of quantities which may be of interest
        """
        
        out_path = os.path.join(run_info['run_dir'], self.target_filename)
        outQtts = self.getBlobInfo(out_path)
        return outQtts

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
    """
    Defines parameters to be applied to the system.

    Parameters
    ----------
    paramFile (string)
        Name of file containing system parameters, not implemented yet.

    Returns
    -------
    params (dict)
        Dictionary of parameters, their default values and their range of uncertainty.
    vary (dict)
        Dictionary of uncertain parameters and their distributions.
    output_columns (list)
        List of the quantities extracted by the decoder we want to return.
        Options:
            maxV: the maximum major radial CoM velocity achieved by the blob
            maxX: the distance the blob propagates before disintegration
    template (str)
        Filename of the template to be used.
    """
    
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
                "width": cp.Normal(0.09, 0.02)# Try different distribution?
        }
        
        output_columns = ["maxV"]
        template = 'b2d.template'
        
        return params, vary, output_columns, template
    
    else:
        # Don't plan to use parameter files but will write in code to do so if needed
        pass
        #return params, vary, output_columns, template

def setupCampaign(params, output_columns, template):
    """
    Builds a campaign using the parameters provided.

    Parameters
    ----------
    params (dict)
        Dictionary of parameters, their default values and their range of uncertainty.
    output_columns (list)
        List of the quantities we want the decoder to pass out.
    template (str)
        Filename of the template to be used.

    Returns
    -------
    campaign (easyvvuq campaign object)
        The campaign, build accoring to the provided parameters.
    """
    
    # Create encoder
    encoder = uq.encoders.GenericEncoder(
            template_fname=template,
            delimiter='$',
            target_filename='BOUT.inp')
    
    # Create executor - 60+ timesteps should be resonable (higher np?)
    execute = ExecuteLocal('mpirun -np 4 {}/blob2d -q -d ./ nout=6'.format(os.getcwd()))
    
    # Create decoder
    decoder = b2dDecoder(
            target_filename="./",
            output_columns=output_columns)
    
    # Pack up encoder, decoder and executor
    actions = Actions(CreateRunDirectory('/tmp'), Encode(encoder), execute, Decode(decoder))
    
    # Build campaign
    campaign = uq.Campaign(name='sc_adaptive', work_dir='/tmp', params=params, actions=actions)
    
    return campaign

def setupSampler(vary):
    """
    Creates and returns an easyvvuq sampler object with the uncertain parameters from vary.
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
    """
    Runs a campaign using provided sampler.
    """
    
    campaign.set_sampler(sampler)
    campaign.execute().collate(progress_bar=True)

###############################################################################

def main():
    # Get campaign parameters
    params, vary, output_columns, template = defineParams()
    
    # Build campaign
    campaign = setupCampaign(params, output_columns, template)
    
    # Set up sampler
    sampler = setupSampler(vary)
    
    # Run the campaign
    runCampaign(campaign, sampler)
    
    print("Campaign run & collated successfuly")

if __name__ == "__main__":
    main()
