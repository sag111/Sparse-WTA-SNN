import os
os.environ["PYNEST_QUIET"] = "1"

import numpy as np

from collections import namedtuple

from random import sample, choices
from sklearn.model_selection import StratifiedShuffleSplit

from fsnn_classifiers.components.networks.base_spiking_transformer import BaseSpikingTransformer
from fsnn_classifiers.components.networks.common_model_components import disable_plasticity, flip_plasticity
from fsnn_classifiers.components.networks.utils import convert_random_parameters_to_nest

import nest

class BaseClasswiseBaggingNetwork(BaseSpikingTransformer):

    def __init__(
        self,
        n_fields, # number of features before GRF
        synapse_model,
        V_th=-54.,
        t_ref=4.,
        tau_m=60.,
        Wmax=1.0, # max synaptic weight
        mu_plus=0.0,
        mu_minus=0.0,
        n_estimators=1, # number of sub-networks
        max_features=1., # number of features for each sub-network
        max_samples=1., # number of samples for each sub-network
        w_inh=None,
        w_init=1.0,
        weight_normalization=None,
        bootstrap_features=False,
        random_state=None,
        **kwargs,
        ):

        self.n_estimators = int(n_estimators)
        self.max_features = max_features
        self.max_samples = max_samples
        self.bootstrap_features = bootstrap_features
        self.weight_normalization = weight_normalization
        self.w_inh = w_inh

        self.w_init = w_init

        if n_fields is not None:
            self.n_fields = int(n_fields)
        else:
            self.n_fields = 1

        self.synapse_model = synapse_model
        self.V_th = V_th
        self.t_ref = t_ref
        self.tau_m = tau_m
        self.Wmax = Wmax
        self.mu_plus = mu_plus
        self.mu_minus = mu_minus

        self.random_state = random_state


        assert 0. < self.max_features <= 1., "<max_features> should be a float in range (0.,1.]"
        assert 0. < self.max_samples <= 1., "<max_features> should be a float in range (0.,1.]"

        if self.max_samples != 1.:
            self.data_sampler = StratifiedShuffleSplit(test_size=self.max_samples, 
                                                       random_state=random_state, 
                                                       n_splits=int(n_estimators))
        else:
            self.data_sampler = None

    def _get_network_objects_tuple(self, has_teacher=False):

        objects = ['neuron_ids',
                    'generators_ids',
                    'teacher_ids',
                    'inputs_ids',
                    'all_connection_descriptors',
                    'spike_recorder_id']

        return namedtuple(
                            'network_objects_tuple',
                            tuple(objects)
                        )

    def _get_parameters(self, number_of_inputs, number_of_classes, testing_mode):
        
        neuron_parameters = {
            'C_m': 1.5374180586077273, 
            'I_e': 0.0, 
            'V_th': self.V_th, 
            'tau_syn_in': 5.0, 
            'tau_syn_ex': 5.0, 
            'tau_minus': 59.96278052520938, 
            'E_L': -70.0, 
            't_ref': self.t_ref, 
            'tau_m': self.tau_m,
        }

        synapse_parameters = {
            'synapse_model': self.synapse_model,
        }

        if self.synapse_model not in ['stdp_tanh_synapse',
                             'stdp_gaussian_times_linear_with_separate_exp_r_dependence_synapse']:
            synapse_parameters["Wmax"] = self.Wmax
            synapse_parameters["mu_plus"] = self.mu_plus
            synapse_parameters["mu_minus"] = self.mu_minus

        if hasattr(self, 'weights_'):
            synapse_parameters.update(weight=self.weights_['weight'])
            feature_indices = self.weights_['pre_index']
            class_indices = self.weights_['post_index']
        else:
            # (n_fields * n_features) -> (n_features, n_fields)
            # (n_classes * n_estimators) -> (n_estimators, n_classes)
            self.n_features = int(number_of_inputs / self.n_fields)

            out_idx = np.arange(0, 
                                number_of_classes * self.n_estimators, 
                                1).astype(np.int32).reshape(self.n_estimators, number_of_classes)
            
            in_idx = np.arange(0, 
                               number_of_inputs, 
                               1).astype(np.int32).reshape(self.n_features, self.n_fields)
            
            feat_idx = list(np.arange(0, self.n_features, 1).astype(np.int32))

            self.out_idx = out_idx # save it to use later

            k = int(self.max_features * self.n_features)

            feature_indices = []
            class_indices = []

            for i in range(self.n_estimators): # iterate over sub-networks
                if self.bootstrap_features:
                    f_idx = choices(feat_idx, k=k)
                else:
                    f_idx = sample(feat_idx, k=k)

                # each subnetwork gets access to all receptive fields of a feature
                feat_ = np.ravel(in_idx[f_idx, :]) 
                cls_ = np.ravel(out_idx[i, :])

                for f in feat_:
                    for c in cls_:
                        feature_indices.append(f)
                        class_indices.append(c)

            weights = np.random.rand(len(feature_indices)) * self.w_init
            
            synapse_parameters.update(weight=weights)

        if testing_mode:
            synapse_parameters = disable_plasticity(synapse_parameters)

        neuron_model = 'iaf_psc_exp'

        neuron_parameters, synapse_parameters = map(
            convert_random_parameters_to_nest,
            (neuron_parameters, synapse_parameters),
        )

        return (neuron_parameters, 
                synapse_parameters, 
                neuron_model, 
                feature_indices, 
                class_indices)
    
    def _create_neuron_populations(self, 
                                   number_of_inputs,
                                   number_of_classes,
                                   neuron_model, 
                                   generator_type,
                                   neuron_parameters, 
                                   has_teacher=False,
                                   create_spike_recorders=False):

        # create neuron populations
        neuron_ids = nest.Create(
            neuron_model,
            number_of_classes * self.n_estimators,
            params=neuron_parameters
        )
        
        inputs_ids = nest.Create('parrot_neuron', number_of_inputs)

        if create_spike_recorders:
            spike_recorder_id = nest.Create('spike_recorder')
        else:
            spike_recorder_id = None

        generators_ids = nest.Create(generator_type, number_of_inputs)

        if has_teacher:
            teacher_ids = nest.Create("step_current_generator", 
                                    number_of_classes * self.n_estimators)
        else:
            teacher_ids = None
        
        return (inputs_ids, generators_ids, neuron_ids, teacher_ids, spike_recorder_id)
    
    def _connect_neuron_populations(self,
                                    synapse_parameters,
                                    feature_indices,
                                    class_indices, 
                                    inputs_ids, 
                                    generators_ids, 
                                    neuron_ids, 
                                    teacher_ids=None, 
                                    spike_recorder_id=None,
                                    ):

        # connect neuron populations
        nest.Connect(
            pre=generators_ids,
            post=inputs_ids,
            conn_spec="one_to_one",
            syn_spec="static_synapse",
        )

        nest.Connect(
            pre=np.array(inputs_ids)[np.array(feature_indices)],
            post=np.array(neuron_ids)[np.array(class_indices)],
            conn_spec="one_to_one",
            syn_spec=synapse_parameters,
        )

        if teacher_ids is not None:
            nest.Connect(
                pre=teacher_ids,
                post=neuron_ids,
                conn_spec="one_to_one",
                syn_spec="static_synapse",
            )

        if self.w_inh is not None and teacher_ids is not None:
            out_idx = np.arange(0, len(neuron_ids), 1).astype(np.int32).reshape((self.n_estimators, -1))
            n_classes = out_idx.shape[-1]
            # iterate over classes
            for i in range(n_classes):
                for c in range(n_classes):
                    if c != i:
                        nest.Connect(
                            pre=np.array(neuron_ids)[out_idx[:, i]],
                            post=np.array(neuron_ids)[out_idx[:, c]],
                            conn_spec="one_to_one",
                            syn_spec={'synapse_model': 'static_synapse',
                            'weight': np.ones(self.n_estimators) * self.w_inh}
                            )

        if spike_recorder_id is not None:
            nest.Connect(neuron_ids, spike_recorder_id, conn_spec='all_to_all')

        return None
        
    
    def _bootstrap_samples(self, X, y):
        
        active_neurons = {k:set() for k in range(len(X))}

        if self.data_sampler is not None:
            # collect indices of neurons that will be active for a given sample
            for i, (_, test_idx) in enumerate(self.data_sampler.split(X,y)):
                # get silent neuron indices
                cls_neurons = np.ravel(self.out_idx[i, :])
                for s_i in test_idx:
                    active_neurons[s_i].add(cls_neurons[y[s_i]])
        else:
            
            for i in range(self.n_estimators):
                for s_i, y_i in enumerate(y):
                    cls_neurons = np.ravel(self.out_idx[i, :])
                    active_neurons[s_i].add(cls_neurons[y_i])

        for key, y_i in zip(active_neurons.keys(), y):
            # we want to use each sample at least once
            if len(active_neurons[key]) == 0:
                idx = choices(range(self.out_idx.shape[0]), k=1)
                active_neurons[key].add(int(self.out_idx[idx, y_i]))
            
        return active_neurons
    
    def _early_stopping(self, weights, previous_weights):
        flag = False
        if (
            np.abs(
                weights - previous_weights
            ) < 0.001
        ).all():
            print(
                'Early stopping because none of the weights'
                'have changed by more than 0.001 for an epoch.',
                'This usually means that the neuron emits no spikes.'
            )
            flag = True
        if np.logical_or(
            weights < 0.1,
            weights > 0.9
        ).all():
            print('Early stopping on weights convergence to 0 or 1.')
            flag = True
        return flag
