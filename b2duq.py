#!/usr/bin/env python3

"""
Runs a dimension adaptive stochastic colocation UQ campaign on the blob2d model
Should be run with python3 in the same folder as blob2d and a blob2d input template.

Dependencies: easyvvuq-1.2 & xbout-0.3.5.
"""

import argparse
import pathlib
import easyvvuq as uq
import numpy as np
import chaospy as cp
import os
import matplotlib.pyplot as plt
from easyvvuq.actions import CreateRunDirectory, Encode, Decode, ExecuteLocal, Actions
from pprint import pprint
import pickle
import time

from easyvvuq import OutputType
from xbout import open_boutdataset
from boutdata.data import BoutOptionsFile


CAMPAIGN_NAME = "delta_star."
QOIS = ["com_x", "com_v_x", "peak_com_v_x"]


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
        self.output_type = OutputType("sample")

    @staticmethod
    def _get_output_path(run_info=None, outfile=None):
        """
        Get the path the run directory, and optionally the file outfile
        """
        if run_info is None:
            raise RuntimeError("Passed 'None' to 'run_info'")

        run_path = run_info.get("run_dir", "data")

        if not os.path.isdir(run_path):
            raise RuntimeError(f"Run directory does not exist: {run_path}")

        return os.path.join(run_path, outfile)

    def sim_complete(self, run_info=None):
        """Return True if the simulation has finished"""
        settings_filename = self._get_output_path(run_info, "BOUT.settings")

        if not os.path.isfile(settings_filename):
            return False

        # Check for normal, clean finish
        settings_file = BoutOptionsFile(settings_filename)

        return "run:finished" in settings_file

    def parse_sim_output(self, run_info):
        ds = self.get_outputs(run_info).squeeze()

        dx = ds["dx"].isel(x=0).values
        ds = ds.drop("x")
        ds = ds.assign_coords(x=np.arange(ds.sizes["x"]) * dx)

        background_density = 1.0
        ds["delta_n"] = ds["n"] - background_density
        ds["delta_n*x"] = ds["delta_n"] * ds["x"]
        integrated_density = ds.bout.integrate_midpoints("delta_n")
        ds["CoM_x"] = ds.bout.integrate_midpoints("delta_n*x") / integrated_density
        ds["CoM_vx"] = ds["CoM_x"].differentiate("t")

        return {
            "com_x": ds["CoM_x"].values.flatten().tolist(),
            "com_v_x": ds["CoM_vx"].values.flatten().tolist(),
            "peak_com_v_x": ds["CoM_vx"].max().values.flatten().tolist(),
        }

    def get_outputs(self, run_info):
        """Read the BOUT++ outputs into an xarray dataframe"""
        data_files = self._get_output_path(run_info, self.target_filename)
        return open_boutdataset(data_files, info=False)


###############################################################################


def refine_sampling_plan(campaign, analysis, number_of_refinements, param):
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

    sampler = campaign.get_active_sampler()

    for i in range(number_of_refinements):
        # compute the admissible indices
        sampler.look_ahead(analysis.l_norm)

        # run the ensemble
        campaign.execute().collate(progress_bar=True)

        # accept one of the multi indices of the new admissible set
        data_frame = campaign.get_collation_result()
        analysis.adapt_dimension(param, data_frame)
        analysis.save_state(f"{campaign.campaign_dir}/analysis.state")


def plot_sobols(params, sobols):
    """
    Plots a bar chart of the sobol indices for each input parameter
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, title="First-order Sobol indices")
    ax.bar(range(len(sobols)), height=np.array(sobols).flatten())
    ax.set_xticks(range(len(sobols)))
    ax.set_xticklabels(params)
    ax.set_yscale("log")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("Sobols.png")
    # plt.show()


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

    params = {
        "Te0": {
            "type": "float",
            "min": 2.5,
            "max": 7.5,
            "default": 5.0,
        },  # Ambient temperature
        "n0": {
            "type": "float",
            "min": 1.0e18,
            "max": 4.0e18,
            "default": 2.0e18,
        },  # Ambient density
        "D_vort": {
            "type": "float",
            "min": 0.9e-7,
            "max": 1.1e-5,
            "default": 1.0e-6,
        },  # Viscosity
        "D_n": {
            "type": "float",
            "min": 0.9e-7,
            "max": 1.1e-5,
            "default": 1.0e-6,
        },  # Diffusion
        "height": {
            "type": "float",
            "min": 0.25,
            "max": 0.75,
            "default": 0.5,
        },  # Blob amplitude
        "width": {
            "type": "float",
            "min": 0.03,
            "max": 0.15,
            "default": 0.09,
        },  # Blob width
        "L_par": {
            "type": "float",
            "min": 1.0,
            "max": 100.0,
            "default": 10.0,
        },
        "R_c": {
            "type": "float",
            "min": 0.01,
            "max": 100,
            "default": 1.5,
        },
        "B0": {
            "type": "float",
            "min": 0.01,
            "max": 3,
            "default": 0.35,
        },
    }
    vary = {
        "Te0": cp.Uniform(2.5, 7.5),
        # "n0": cp.Uniform(1.0e+18, 4.0e+18),
        # "D_vort": cp.Uniform(1.0e-7, 1.0e-5),
        # "D_n": cp.Uniform(1.0e-7, 1.0e-5),
        # "height": cp.Uniform(0.25, 0.75),
        # "width": cp.Uniform(0.03, 0.15)
        "L_par": cp.Uniform(1.0, 100.0),
        "R_c": cp.Uniform(0.1, 10.0),
        "B0": cp.Uniform(0.03, 2.5),
    }  # Try latin hypercube?

    template = "b2d.template"

    return params, vary, QOIS, template


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
        template_fname=template, delimiter="$", target_filename="BOUT.inp"
    )

    # Create executor - 50+ timesteps should be resonable (higher np?)
    execute = ExecuteLocal(f"mpirun -np 8 {os.getcwd()}/blob2d -d ./ -q -q -q")

    # Create decoder
    decoder = B2dDecoder(target_filename="BOUT.dmp.*.nc", output_columns=output_columns)

    # Ensure run directory exists, then pack up encoder, decoder, executor and build campaign
    actions = Actions(
        CreateRunDirectory("."), Encode(encoder), execute, Decode(decoder)
    )
    campaign = uq.Campaign(
        name=CAMPAIGN_NAME,
        params=params,
        actions=actions,
    )

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
        dimension_adaptive=True,
    )

    return sampler


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
    frame = campaign.get_collation_result()
    analysis = uq.analysis.SCAnalysis(sampler=sampler, qoi_cols=output_columns)

    # Run analysis
    campaign.apply_analysis(analysis)
    print(analysis.l_norm)

    # Print mean and variation of quantity and get adaptation errors
    results = analysis.analyse(frame)
    print(f'Mean transport rate = {results.describe("avgTransp", "mean")}')
    print(f'Standard deviation = {results.describe("avgTransp", "std")}')
    print(f'Mean mass loss = {results.describe("massLoss", "mean")}')
    print(f'Standard deviation = {results.describe("massLoss", "std")}')
    analysis.get_adaptation_errors()

    # Get Sobol indices (online for loop automatically creates a list without having to append)
    params = sampler.vary.get_keys()  # This is also used in plot_sobols
    sobols = [results._get_sobols_first("avgTransp", param) for param in params]
    print(sobols)

    # Plot Analysis
    analysis.adaptation_table()
    analysis.adaptation_histogram()
    analysis.get_adaptation_errors()
    plot_sobols(params, sobols)


###############################################################################


def plot_grid_2D(campaign, analysis, i, filename="out.pdf"):
    fig = plt.figure(figsize=[12, 4])
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    accepted_grid = campaign.get_active_sampler().generate_grid(analysis.l_norm)
    ax1.plot(accepted_grid[:, 0], accepted_grid[:, 1], "o")
    ax2.plot(accepted_grid[:, 2], accepted_grid[:, 3], "o")
    ax1.set_title(f"iteration {i}")

    fig.tight_layout()
    fig.savefig(filename)


def custom_moments_plot(results, filename, i, qoi="com_v_x"):
    fig, ax = plt.subplots()
    xvalues = np.arange(len(results.describe(qoi, "mean")))
    ax.fill_between(
        xvalues,
        results.describe(qoi, "mean") - results.describe(qoi, "std"),
        results.describe(qoi, "mean") + results.describe(qoi, "std"),
        label="std",
        alpha=0.2,
    )
    ax.plot(xvalues, results.describe(qoi, "mean"), label="mean")
    try:
        ax.plot(xvalues, results.describe(qoi, "1%"), "--", label="1%", color="black")
        ax.plot(xvalues, results.describe(qoi, "99%"), "--", label="99%", color="black")
    except RuntimeError:
        pass
    ax.grid(True)
    ax.set_ylabel(qoi)
    ax.set_xlabel("time")
    ax.set_title("iteration " + str(i))
    ax.legend()
    fig.savefig(filename)


def refine_once(campaign, analysis, iteration):
    refine_sampling_plan(campaign, analysis, 1, "peak_com_v_x")
    campaign.apply_analysis(analysis)
    analysis.save_state(f"{campaign.campaign_dir}/analysis.state")

    results = campaign.last_analysis
    plot_grid_2D(
        campaign,
        analysis,
        iteration,
        f"{campaign.campaign_dir}/grid{iteration:02}.png",
    )
    moment_plot_filename = os.path.join(
        f"{campaign.campaign_dir}", f"moments{iteration:02}.png"
    )
    sobols_plot_filename = os.path.join(
        f"{campaign.campaign_dir}", f"sobols_first{iteration:02}.png"
    )
    results.plot_sobols_first(
        "com_v_x",
        ylabel=f"iteration{iteration}",
        xlabel="time",
        filename=sobols_plot_filename,
    )
    plt.ylim(0, 1)
    plt.savefig(f"{campaign.campaign_dir}/sobols{iteration:02}.png")

    custom_moments_plot(results, moment_plot_filename, iteration)

    with open(f"{campaign.campaign_dir}/last_iteration", "w") as f:
        f.write(f"{iteration}")


def create_analysis(campaign, qoi_cols):
    return uq.analysis.SCAnalysis(
        sampler=campaign.get_active_sampler(), qoi_cols=qoi_cols
    )


def first_time_setup():
    params, vary, output_columns, template = define_params()
    campaign = setup_campaign(params, output_columns, template)
    sampler = setup_sampler(vary)
    campaign.set_sampler(sampler)

    print(f"Output will be in {campaign.campaign_dir}")

    sampler = campaign.get_active_sampler()

    print(f"Computing {sampler.n_samples} samples")

    time_start = time.time()
    campaign.execute().collate(progress_bar=True)

    # Create an analysis class and run the analysis.
    analysis = create_analysis(campaign, output_columns)
    campaign.apply_analysis(analysis)
    analysis.save_state(f"{campaign.campaign_dir}/analysis.state")
    plot_grid_2D(campaign, analysis, 0, f"{campaign.campaign_dir}/grid0.png")

    for i in np.arange(1, 5):
        refine_once(campaign, analysis, i)
    time_end = time.time()

    print(f"Finished, took {time_end - time_start}")

    return campaign


def reload_campaign(directory):
    """Reload a campaign from a directory

    Returns the campaign, analysis, and last iteration number
    """

    campaign = uq.Campaign(
        name=CAMPAIGN_NAME,
        db_location=f"sqlite:///{os.path.abspath(directory)}/campaign.db",
    )
    analysis = create_analysis(campaign, QOIS)
    analysis.load_state(f"{campaign.campaign_dir}/analysis.state")

    with open(f"{campaign.campaign_dir}/last_iteration", "r") as f:
        iteration = int(f.read())

    return campaign, analysis, iteration


def main():
    parser = argparse.ArgumentParser(
        "conduction_sc",
        description="Adaptive dimension refinement for 1D conduction model",
    )
    parser.add_argument(
        "--restart", type=str, help="Restart previous campaign", default=None
    )
    parser.add_argument(
        "-n", "--refinement-num", type=int, default=1, help="Number of refinements"
    )

    args = parser.parse_args()

    if args.restart is None:
        first_time_setup()
    else:
        campaign, analysis, last_iteration = reload_campaign(args.restart)
        for iteration in range(
            last_iteration + 1, last_iteration + args.refinement_num + 1
        ):
            refine_once(campaign, analysis, iteration)


if __name__ == "__main__":
    main()
