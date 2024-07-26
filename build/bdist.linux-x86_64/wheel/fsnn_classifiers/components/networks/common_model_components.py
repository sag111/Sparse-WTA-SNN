def disable_plasticity(synapse_parameters):
    new_synapse_parameters = synapse_parameters.copy()
    amplitude_constant_names = {
        'stdp_synapse': ['lambda'],
        'stdp_nn_symm_synapse': ['lambda'],
        'stdp_nn_pre-centered_synapse': ['lambda'],
        'stdp_nn_restr_synapse': ['lambda'],
        'stdp_tanh_synapse': ['a_plus', 'a_minus'],
        'stdp_gaussian_times_linear_with_separate_exp_r_dependence_synapse': ['alpha_plus', 'alpha_minus'],
    }[
        synapse_parameters['synapse_model']
    ]
    new_synapse_parameters.update({
        constant_name: 0.
        for constant_name in amplitude_constant_names
    })
    return new_synapse_parameters