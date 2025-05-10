from sys import path as pythonpath
import numpy as np
import pandas as pd
from itertools import product as cartesian_product
from tqdm import tqdm
import nest


# Time of an extra presynaptic spike.
# Should be so high as to cause no weight change.
# Needed because NEST only updates the weight on a pre-spike.
very_big_time = 2000  # [ms]
simulation_resolution = 0.1  # [ms]

def get_delta_G_from_nest(delta_t, **synapse_parameters):
    '''Return weight change reproduced using NEST.'''
    nest.set_verbosity('M_WARNING')
    nest.ResetKernel()
    nest.SetKernelStatus({'resolution': simulation_resolution})

    simulation_duration = 3*delta_t + very_big_time

    presynaptic_neuron, postsynaptic_neuron = nest.Create(
        'parrot_neuron',
        2
    )

    start_time = max(-delta_t, 0) + 1.
    pre_spike_times = [
        start_time,
        # Needed because NEST only updates the weight
        # on a pre-spike.
        start_time + very_big_time
    ]
    post_spike_times = [
        start_time + delta_t
    ]

    (
        pre_spike_sender,
        post_spike_sender
    ) = nest.Create(
        'spike_generator',
        2,
        params=({'spike_times': pre_spike_times},
                {'spike_times': post_spike_times})
    )

    nest.Connect(
        (
            pre_spike_sender,
            post_spike_sender
        ),
        (
            presynaptic_neuron,
            postsynaptic_neuron
        ),
        syn_spec={'model': 'static_synapse'},
        conn_spec='one_to_one')
    # The synapse of interest
    nest.Connect(
        (presynaptic_neuron,),
        (postsynaptic_neuron,),
        syn_spec=synapse_parameters)
    plastic_synapse_of_interest = nest.GetConnections(synapse_model=synapse_parameters['model'])

    nest.Simulate(simulation_duration)

    final_weight = nest.GetStatus(plastic_synapse_of_interest, keys='weight')[0]
    return final_weight - synapse_parameters['weight']


if __name__ == '__main__':
    pythonpath.append('.')
    from parameters import delta_ts, initial_weights, synapse_parameters
    from fitted_formulae import get_delta_G_from_fit

    dt_and_g_initial_dataframe = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [
                delta_ts,
                initial_weights,
            ],
            names=[
                'delta_t',
                'weight',
            ]
        )
    ).reset_index()

    delta_G_from_nest, delta_G_from_fit = [
        dt_and_g_initial_dataframe.apply(
            lambda row: get_delta_G(**dict(
                synapse_parameters,
                **row
            )),
            axis=1
        )
        for get_delta_G in (
            get_delta_G_from_nest,
            get_delta_G_from_fit
        )
    ]
    results_dataframe = dt_and_g_initial_dataframe.rename(
        columns={'weight': 'G_initial'}
    )
    results_dataframe['delta_G_from_nest'] = delta_G_from_nest
    results_dataframe['delta_G_from_fit'] = delta_G_from_fit
    results_dataframe.to_csv(
        'delta_G_vs_delta_t_and_G_initial.csv',
        index=False
    )
