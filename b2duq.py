"""
Runs a dimension adaptive stochastic colocation UQ campaign on the blob2d model
Should be run with python3 in the same folder as blob2d and a blob2d input template.

Dependencies: easyvvuq-1.2 & xbout-0.3.5.
"""

import easyvvuq as uq
import numpy as np
import chaospy as cp
import os
import matplotlib.pyplot as plt
from easyvvuq.actions import CreateRunDirectory, Encode, Decode, ExecuteLocal, Actions
import pickle

from easyvvuq import OutputType
from xbout import open_boutdataset

class B2dDecoder:
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
    
    def get_blob_info(self, out_path):    
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
        
        # Unpack data from blob2d
        ds = open_boutdataset(out_path, info=False)
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
        ds["transpRate"] = ds.bout.integrate_midpoints("delta_n*x")
        ds["CoM_x"] = ds["transpRate"] / integrated_density
        v_x = ds["CoM_x"].differentiate("t")
        
        # Save useful quantities to dictionary
        maxV = float(max(v_x))
        maxX = float(ds["CoM_x"][list(v_x).index(max(v_x))])
        avgTransp = float(np.mean(ds["transpRate"][:(list(v_x).index(max(v_x)))+1]))
        massLoss = float(integrated_density[list(v_x).index(max(v_x))] / integrated_density[0])
        
        blobInfo = {"maxV": maxV, "maxX": maxX, "avgTransp": avgTransp, "massLoss": massLoss}
        
        return blobInfo
    
    def show_out_options():
        print("""Possible outputs:
            maxV: the maximum major radial CoM velocity achieved by the blob
            maxX: the distance the blob propagates before disintegration
            avgTransp: the average rate of transport of the blob (i.e. flux) over its lifetime
            massLoss: the ratio of blob mass at disintegration to initial blob mass""")
    
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
        outQtts = self.get_blob_info(out_path)
        return outQtts

class B2dDecoder2:
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
    
    def get_blob_info(self, out_path):    
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
        
        # Unpack data from blob2d
        ds = open_boutdataset(out_path, info=False)
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
    
    def show_out_options():
        print("""Possible outputs:
            maxV: the maximum major radial CoM velocity achieved by the blob
            maxX: the distance the blob propagates before disintegration""")
    
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
        outQtts = self.get_blob_info(out_path)
        return outQtts

###############################################################################

def refine_sampling_plan(number_of_refinements, campaign, sampler, analysis):
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
        analysis.adapt_dimension('avgTransp', data_frame)
        #analysis.adapt_dimension('maxV', data_frame, method='var')
            
###############################################################################

def define_params(paramFile=None):
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
                "Te0": {"type": "float", "min": 2.5, "max": 7.5, "default": 5.0},# Ambient temperature
                "n0": {"type": "float", "min": 1.0e+18, "max": 4.0e+18, "default": 2.0e+18},# Ambient density
                "D_vort": {"type": "float", "min": 0.9e-7, "max": 1.1e-5, "default": 1.0e-6},# Viscosity
                "D_n": {"type": "float", "min": 0.9e-7, "max": 1.1e-5, "default": 1.0e-6},# Diffusion
                "height": {"type": "float", "min": 0.25, "max": 0.75, "default": 0.5},# Blob amplitude
                "width": {"type": "float", "min": 0.03, "max": 0.15, "default": 0.09},# Blob width
        }
        vary = {
                "Te0": cp.Uniform(2.5, 7.5),
                "n0": cp.Uniform(1.0e+18, 4.0e+18),
                "D_vort": cp.Uniform(1.0e-7, 1.0e-5),
                "D_n": cp.Uniform(1.0e-7, 1.0e-5),
                #"height": cp.Uniform(0.25, 0.75),
                #"width": cp.Uniform(0.03, 0.15)
        }
        
        output_columns = ["avgTransp", "massLoss"]
        #output_columns = ["maxV", "maxX", "avgTransp", "massLoss"]
        # Show user available and selected output options
        B2dDecoder.show_out_options()
        print("Options selected: ", output_columns, "\n")
        
        template = 'b2d.template'
        
        return params, vary, output_columns, template
    
    else:
        # Don't plan to use parameter files but will write in code to do so if needed
        #pFile = load(paramFile)
        #params = pFile[0]
        #...
        #return params, vary, output_columns, template
        pass

def setup_campaign(params, output_columns, template):
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
    execute = ExecuteLocal(f'mpirun -np 16 {os.getcwd()}/blob2d -d ./ nout=3 -q -q -q ')
    
    # Create decoder
    decoder = B2dDecoder(
            target_filename="BOUT.dmp.*.nc",
            output_columns=output_columns)
    
    # Ensure run directory exists, then pack up encoder, decoder, executor and build campaign
    if os.path.exists('outfiles')==0: os.mkdir('outfiles')
    actions = Actions(CreateRunDirectory('outfiles'), Encode(encoder), execute, Decode(decoder))
    campaign = uq.Campaign(name='sc_adaptive', work_dir='outfiles', params=params, actions=actions)
    
    return campaign

def setup_sampler(vary):
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

def run_campaign(campaign, sampler):
    """
    Runs a campaign using provided sampler.
    """
    
    campaign.set_sampler(sampler)
    campaign.execute().collate(progress_bar=True)

def analyse_campaign(campaign, sampler, output_columns):
    """
    Runs a set of analyses on a provided campaign, details often change by commit.
    
    Parameters
    ----------
    campaign (easyvvuq Campaign object)
        The campaign being analysed
    sampler (easyvvuq SCSampler object)
        The sampler being used
    output_columns (dict)
        List of output quantities under consideration

    Returns
    -------
    None - results either printed to screen, plotted or saved to a file.
    """
    # Create analysis class
    dParams = campaign.get_collation_result()
    analysis = uq.analysis.SCAnalysis(sampler=sampler, qoi_cols=output_columns)
    
    # Run analysis
    campaign.apply_analysis(analysis)
    print(analysis.l_norm)
    
    # Refine analysis
    refine_sampling_plan(2, campaign, sampler, analysis)
    campaign.apply_analysis(analysis)
    print(analysis.l_norm)
    
    # Save analysis class
    
    # Print mean and variation of quantity
    #dParams = campaign.get_collation_result()##########################################################
    results = analysis.analyse(dParams)
    print(f'Mean transport rate = {results.describe("avgTransp", "mean")}')
    print(f'Standard deviation = {results.describe("avgTransp", "std")}')
    print(f'Mean mass loss = {results.describe("massLoss", "mean")}')
    print(f'Standard deviation = {results.describe("massLoss", "std")}')
    
    # Plot Analysis
    #analysis.adaptation_table()
    #analysis.adaptation_histogram()
    #analysis.get_adaptation_errors()
    
    # Get Sobol indices (online foor loop automatically creates a list without having to append)
    params = sampler.vary.get_keys()# This is used later too
    sobols = [results._get_sobols_first('avgTransp', param) for param in params]
    
    # Plot sobol indices
    fig = plt.figure()
    ax = fig.add_subplot(111, title='First-order Sobol indices')
    ax.bar(range(len(sobols)), height=np.array(sobols).flatten())
    ax.set_xticks(range(len(sobols)))
    ax.set_xticklabels(params)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("Sobols.png")
    #plt.show()
    analysis.adaptation_table()

###############################################################################

def main():
    params, vary, output_columns, template = define_params()
    campaign = setup_campaign(params, output_columns, template)
    sampler = setup_sampler(vary)
    run_campaign(campaign, sampler)
    analyse_campaign(campaign, sampler, output_columns)
    
    print("Campaign run & analysed successfuly")

if __name__ == "__main__":
    main()
