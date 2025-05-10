import numpy as np
import time


# def spikes_to_times(inp_spikes, sim_time, tau_s, resolution = 0.1):
#     spike_times = (np.arange(0, sim_time, resolution).reshape((1,-1)) + resolution + tau_s).round(1)
#     spike_times = np.repeat(spike_times, inp_spikes.shape[0], axis=0) # number of neurons
#     spike_times = spike_times * inp_spikes
#     return spike_times

def get_time_dict(spike_times, special_value=None):
    if special_value:
        return [
            {"spike_times": [] if st_i == special_value else np.round([st_i], 1)}
            for st_i in spike_times
        ]
    else:
        return [
            {"spike_times": np.round([st_i], 1)}
            for st_i in spike_times
        ]

class CorrelationEncoder(object):
     
    def __init__(self, simulation_time=1, special_value=None):
        self.simulation_time = simulation_time
        self.special_value = special_value

    def __call__(self, X: np.ndarray):
        return np.round(X * self.simulation_time, 1)
