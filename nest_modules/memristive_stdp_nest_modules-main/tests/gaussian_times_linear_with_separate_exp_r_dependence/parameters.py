import numpy as np

max_delta_t = 500  # [ms]
delta_ts = np.arange(-max_delta_t, max_delta_t, 1)
initial_weights = np.array([0.0001, 0.0006666666666666666, 0.0002, 0.00125]) / 1e-2

synapse_parameters = {
    'model': 'stdp_gaussian_times_linear_with_separate_exp_r_dependence_synapse',
}
