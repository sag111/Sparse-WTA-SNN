from collections import namedtuple

snn_parameters = {
                'synapse_parameters': {
                        'exc_to_inh': {
                            'synapse_model': 'static_synapse',
                            'weight': 10.4
                        },
                        'inh_to_exc': {
                            'synapse_model': 'static_synapse',
                            'weight': -17.0
                        },
                        'input_to_exc': {  
                            'Wmax': 1.0,
                            'alpha': 0.5534918994526379,
                            'delay': {
                                'parametertype': 'uniform',
                                'specs': {
                                    'min': 0.1,
                                    'max': 10.0,
                                },
                            },
                            'lambda': 0.01,
                            'synapse_model': 'stdp_nn_symm_synapse',
                            'mu_minus': 0.0,
                            'mu_plus': 0.0,
                            'tau_plus': 20.0,
                            'weight': {
                                'parametertype': 'uniform',
                                'specs': {
                                    'min': 0.0,
                                    'max': 1.0,
                                },
                            },
                        },
                        'input_to_inh': {
                            'delay': {
                                'parametertype': 'uniform',
                                'specs': {
                                    'min': 0.1,
                                    'max': 5.0,
                                },
                            },
                            'synapse_model': 'static_synapse',
                            'weight': {
                                'parametertype': 'uniform',
                                'specs': {
                                    'min': 0.0,
                                    'max': 0.2,
                                },
                            },
                        },
                    },
                'neuron_parameters': {
                    'exc_neurons': {
                        'C_m': 100.0,
                        'E_L': -65.0,
                        'E_ex': 0.0,
                        'E_in': -100.0,
                        'I_e': 0.0,
                        'Theta_plus': 0.05,
                        'Theta_rest': -72.0,
                        'V_m': -105.0,
                        'V_reset': -65.0,
                        'V_th': -52.0,
                        't_ref': 5.0,
                        'tau_m': 100.0,
                        'tau_synE': 1.0,
                        'tau_synI': 2.0,
                        'tc_theta': 10000000.0
                    },
                    'inh_neurons': {
                        'C_m': 10.0,
                        'E_L': -60.0,
                        'E_ex': 0.0,
                        'E_in': -100.0,
                        'I_e': 0.0,
                        'Theta_plus': 0.0,
                        'Theta_rest': -40.0,
                        'V_m': -100.0,
                        'V_reset': -45.0,
                        'V_th': -40.0,
                        't_ref': 2.0,
                        'tau_m': 10.0,
                        'tau_synE': 1.0,
                        'tau_synI': 2.0,
                        'tc_theta': 1e+20
                    }
                },
                'network_parameters': {
                    'cross_validation_splits': 5,
                    'epochs': 1,
                    'high_rate': 43.37900464236736,
                    'input_to_inh_connection_prob': 0.1,
                    'intervector_pause': 50.0,
                    'low_rate': 0.017862529028207064,
                    'number_of_exc_neurons': 100,
                    'number_of_inh_neurons': 100,
                    'one_vector_longtitude': 350.0,
                    'weight_normalization_during_training': None
                },
            }



network_objects_tuple = namedtuple(
    'network_objects_tuple',
    (
        'exc_neuron_ids',
        'inh_neuron_ids',
        'generators_ids',
        'inputs_ids',
        'all_connection_descriptors',
        'exc_neurons_spike_recorder_id',
        'inh_neurons_spike_recorder_id',
    )
)
