import os
os.environ["PYNEST_QUIET"] = "1"

import numpy as np

import itertools

from collections import namedtuple

from time import time as time_time
from random import sample, choices
import random

from sklearn.model_selection import StratifiedShuffleSplit

import sys

path_to_remove = '/s/ls4/users/romanrybka/YD/fsnn-classifiers'
if path_to_remove in sys.path:
    sys.path.remove(path_to_remove)
    
sys.path.insert(0, '/s/ls4/users/selibrin/spiking_researches/Sparse-WTA-SNN')

from fsnn_classifiers.components.networks.base_spiking_transformer import BaseSpikingTransformer
from fsnn_classifiers.components.networks.common_model_components import disable_plasticity, flip_plasticity
from fsnn_classifiers.components.networks.utils import convert_random_parameters_to_nest

import nest

class BaseClasswiseBaggingNetwork(BaseSpikingTransformer):

    def __init__(
        self,
        n_fields, # number of features before GRF
        synapse_model,
        neuron_model="iaf_psc_exp", 
        V_th=-54.,
        t_ref=0.0,
        tc_theta=None,
        Theta_plus=None,
        tau_m=60.,
        tau_minus=60., 
        Wmax=1.0, # max synaptic weight
        tau_plus=20.,
        mu_plus=0.0,
        mu_minus=0.0,
        alpha=1.0,
        # lambda_stdp=0.01,
        n_estimators=1, # number of sub-networks
        max_features=1., # number of features for each sub-network
        max_samples=1., # number of samples for each sub-network
        w_inh=None,
        w_init=1.0,
        # weight_normalization=None,
        bootstrap_features=False,
        random_state=None,
        use_entire_train_data=False,
        convolution_window=None,
        class_name=None,
        **kwargs,
        ):

        self.n_estimators = int(n_estimators)
        self.max_features = max_features
        self.max_samples = max_samples
        self.bootstrap_features = bootstrap_features
        # self.weight_normalization = weight_normalization
        self.w_inh = w_inh

        self.w_init = w_init
        self.class_name = class_name

        if n_fields is not None:
            self.n_fields = int(n_fields)
        else:
            self.n_fields = 1

        self.synapse_model = synapse_model
        self.neuron_model = neuron_model
        self.tc_theta = tc_theta
        self.Theta_plus = Theta_plus
        self.V_th = V_th
        self.t_ref = t_ref
        self.tau_m = tau_m
        self.tau_minus = tau_minus # 59.96278052520938, 
        self.Wmax = Wmax
        self.tau_plus = tau_plus
        self.mu_plus = mu_plus
        self.mu_minus = mu_minus
        self.alpha = alpha
        self.shift_plus = kwargs["shift_plus"] if "shift_plus" in kwargs else None
        self.shift_minus = kwargs["shift_minus"] if "shift_minus" in kwargs else None

        # self.lambda_stdp = lambda_stdp
        self.convolution_window = convolution_window

        self.random_state = random_state

        self.use_entire_train_data = use_entire_train_data

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
            'tau_minus': self.tau_minus, # 59.96278052520938, 
            'E_L': -70.0, 
            't_ref': self.t_ref, 
            'tau_m': self.tau_m,
        }
        if self.neuron_model == 'iaf_cond_exp_adaptive':
            neuron_parameters['tc_theta'] = self.tc_theta # 1e7 ms
            neuron_parameters['Theta_plus'] = self.Theta_plus # 0.05 mV
            neuron_parameters['Theta_rest'] = self.V_th

        synapse_parameters = {
            'synapse_model': self.synapse_model,
            'delay': np.full(number_of_inputs*number_of_classes*self.n_estimators, self.resolution),
        }

        if self.synapse_model not in [
            'stdp_tanh_synapse',
            'stdp_gaussian_times_linear_with_separate_exp_r_dependence_synapse'
        ]:
            synapse_parameters["Wmax"] = self.Wmax
            synapse_parameters["mu_plus"] = self.mu_plus
            synapse_parameters["mu_minus"] = self.mu_minus
            synapse_parameters["tau_plus"] = self.tau_plus
            synapse_parameters["alpha"] = self.alpha
            synapse_parameters["lambda"] = self.lr

        if self.synapse_model in [
            'shifted_stdp_synapse',
            'shifted_nn_symm_stdp_synapse',
            'shifted_nn_pre_centered_stdp_synapse',
            'shifted_nn_restr_stdp_synapse'
        ]:
            if self.shift_plus is None or self.shift_minus is None:
                raise ValueError("`shift_plus` and `shift_minus` must be specified for the selected synapse_model")
            synapse_parameters["shift_plus"] = self.shift_plus
            synapse_parameters["shift_minus"] = self.shift_minus

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

            random_float = time_time()
            seed = (random_float - int(random_float)) * 1e6
            random.seed(seed)

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

            # np.random.seed(42)
            weights = np.random.rand(len(feature_indices)) * self.w_init
            # np.save('/s/ls4/users/selibrin/initial_weights.npy', weights)
            
            synapse_parameters.update(weight=weights)

        if testing_mode:
            synapse_parameters = disable_plasticity(synapse_parameters)

        neuron_model = self.neuron_model # 'iaf_psc_exp'

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
                                   create_spike_recorders=False,
                                   time=None,
                                   ):

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
                                    spike_recorder_id=None
                                    ):

        # connect neuron populations
        nest.Connect(
            pre=generators_ids,
            post=inputs_ids,
            conn_spec="one_to_one",
            syn_spec={"synapse_model": "static_synapse", "delay": np.full(len(generators_ids), self.resolution)}
        )

        nest.Connect(
            pre=np.array(inputs_ids)[np.array(feature_indices)],
            post=np.array(neuron_ids)[np.array(class_indices)],
            conn_spec="one_to_one",
            syn_spec=synapse_parameters
        )

        if teacher_ids is not None:
            nest.Connect(
                pre=teacher_ids,
                post=neuron_ids,
                conn_spec="one_to_one",
                syn_spec={"synapse_model": "static_synapse", "delay": np.full(len(teacher_ids), self.resolution)}
            )

        if spike_recorder_id is not None:
            nest.Connect(neuron_ids, spike_recorder_id, conn_spec='all_to_all')

        return None


    def _bootstrap_samples(self, X, y):
        # np.save('/s/ls4/users/selibrin/out_idx.npy', self.out_idx)
        # class_name используем в случае анти стдп, поэтому все нейроны активны на всех примерах 
        if self.class_name is not None:
            return {k: set(np.ravel(self.out_idx)) for k in range(len(X))}

        random_float = time_time()
        seed = (random_float - int(random_float)) * 1e6
        random.seed(seed)

        # if hasattr(self, 'class_name') and self.class_name:
        #     active_neurons = {k:{0} if y_i == self.class_name else set() for k, y_i in enumerate(y)}
        #     return active_neurons
            
        active_neurons = {k:set() for k in range(len(X))}

        if self.data_sampler is not None and not self.use_entire_train_data:
            # collect indices of neurons that will be active for a given sample
            for i, (_, test_idx) in enumerate(self.data_sampler.split(X,y)):
                # get silent neuron indices
                cls_neurons = np.ravel(self.out_idx[i, :])
                for s_i in test_idx:
                    active_neurons[s_i].add(cls_neurons[y[s_i]])

        elif self.use_entire_train_data:
            # индексы сортированных по возрастанию классов
            idxs = np.argsort(y)
            # стакнутые индексы и метки их классов
            sorted_y_train_with_idxs = np.column_stack((idxs, y[idxs])) 

            for c_i in range(self.out_idx.shape[-1]):
                # индексы в трейне, пренадлежащие классу c_i
                c_i_idxs = list(sorted_y_train_with_idxs[sorted_y_train_with_idxs[:, 1] == c_i][:, 0]) 
                # сколько примеров попадет на каждую сетку
                length = int(self.max_samples * len(c_i_idxs) + 1) 
                # зацикленные индексы в трейне, пренадлежащие классу c_i
                gen1 = (x for x in itertools.cycle(c_i_idxs)) 
                # зацикленные номера нейронов класса c_i
                gen2 = (y for y in itertools.cycle(self.out_idx[:, c_i])) 
                # Счетчик итераций
                count = 0 
                while count < length * self.n_estimators: 
                    active_neurons[next(gen1)].add(next(gen2))
                    count += 1

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
