import os
os.environ["PYNEST_QUIET"] = "1"

from tqdm import tqdm
import numpy as np
import nest

import sys

path_to_remove = '/s/ls4/users/romanrybka/YD/fsnn-classifiers'
if path_to_remove in sys.path:
    sys.path.remove(path_to_remove)
    
sys.path.insert(0, '/s/ls4/users/selibrin/spiking_researches/Sparse-WTA-SNN')

from fsnn_classifiers.components.networks.base_classwise_bagging_network import BaseClasswiseBaggingNetwork
from fsnn_classifiers.components.networks.utils import (
    generate_random_state,
    convert_neuron_ids_to_indices,
)
from fsnn_classifiers.components.preprocessing.correlation_encoder_one_spk_input_multi_spk_out import CorrelationEncoder
from fsnn_classifiers.components.networks.common_model_components import flip_plasticity
from sklearn.preprocessing import minmax_scale, MinMaxScaler

from collections import Counter

import os

import pickle


class CorrelationClasswiseNetwork(BaseClasswiseBaggingNetwork):
    def __init__(
        self,
        n_fields, # number of features before GRF
        scale, # multiplier for spike encoding
        clip_val, # the minimum value in dataset that is not zeroed before spike encoding
        n_estimators=1, # number of sub-networks
        max_features=1., # number of features for each sub-network
        max_samples=1., # number of samples for each sub-network
        bootstrap_features=False,
        synapse_model="stdp_nn_pre_centered_synapse",
        neuron_model="iaf_psc_exp",
        V_th=-54.,
        t_ref=0.0,
        tau_m=60.,
        tau_minus=60.,
        I_exc=1e7,
        epochs=1,
        time=1000,
        intervector_pause = 0,
        Wmax=1.0, # max synaptic weight
        tau_plus=20.,
        mu_plus=0.0,
        mu_minus=0.0,
        alpha=1.,
        lr=0.001,
        resolution=0.1,
        sample_norm=1,
        w_inh=None,
        w_init=1.0,
        sigma_w=0.0,
        weight_normalization=None,
        random_state=None,
        early_stopping=False,
        n_jobs=1,
        warm_start=False,
        quiet=True,
        job_id=None,
        **kwargs,
    ):
        
        super(CorrelationClasswiseNetwork, self).__init__(
            n_fields=n_fields,
            synapse_model=synapse_model,
            neuron_model=neuron_model,
            V_th=V_th,
            t_ref=t_ref, 
            tau_m=tau_m,
            tau_minus=tau_minus,
            Wmax=Wmax,
            tau_plus=tau_plus,
            mu_plus=mu_plus,
            mu_minus=mu_minus,
            alpha=alpha,
            n_estimators=n_estimators,
            max_features=max_features,
            max_samples=max_samples,
            bootstrap_features=bootstrap_features,
            random_state=random_state,
            w_inh=w_inh,
            w_init=w_init,
            **kwargs,
        )
        
        self._check_modules(quiet)

        self.sigma_w = sigma_w
        self.lr = lr

        self.early_stopping = early_stopping
        self.n_jobs = n_jobs
        self.warm_start = warm_start
        self.quiet=quiet
        self.sample_norm = sample_norm

        self.weight_normalization = weight_normalization
        self.job_id = job_id

        self.resolution = resolution

        self.epochs = epochs
        self.sample_time = time
        self.intervector_pause = intervector_pause
        self.full_time = self.sample_time + self.intervector_pause
        self.I_exc = I_exc
        
        self.scale = scale
        self.clip_val = clip_val
        self.encoder = CorrelationEncoder(0, self.full_time, self.scale, self.resolution, self.clip_val, self.I_exc)


    def _create_network(self, testing_mode):

        number_of_inputs = self.n_features_in_
        random_state = (
            self.random_state if not self.random_state is None
            else generate_random_state()
        )
        self.number_of_classes = len(self.classes_)

        # Remove existing NEST objects if any exist.
        nest.ResetKernel()
        n_threads = self.n_jobs
        nest.SetKernelStatus({
            'resolution': self.resolution,
            'local_num_threads': n_threads,

        })
        nest.rng_seed = random_state

        (neuron_parameters, 
        synapse_parameters, 
        neuron_model, 
        feature_indices, 
        class_indices) = self._get_parameters(number_of_inputs, self.number_of_classes, testing_mode)

        #synapse_parameters = flip_plasticity(synapse_parameters)

        self.E_L = neuron_parameters.get("E_L", -70.0) # just in case

        # create neuron populations
        (inputs_ids, 
         generators_ids, 
         neuron_ids, 
         teacher_ids, 
         spike_recorder_id) = self._create_neuron_populations(number_of_inputs=number_of_inputs, 
                                                              number_of_classes=self.number_of_classes, 
                                                              neuron_model=neuron_model,
                                                              generator_type="spike_generator",
                                                              neuron_parameters=neuron_parameters, 
                                                              has_teacher=True, 
                                                              create_spike_recorders=testing_mode,
                                                              time=self.sample_time - 1
                                                              )

        # connect neuron populations
        self._connect_neuron_populations(synapse_parameters=synapse_parameters,
                                         feature_indices=feature_indices,
                                         class_indices=class_indices,
                                         inputs_ids=inputs_ids,
                                         generators_ids=generators_ids,
                                         neuron_ids=neuron_ids,
                                         teacher_ids=teacher_ids,
                                         spike_recorder_id=spike_recorder_id
                                         )

        # Now that all connections have been created,
        # request their descriptors from NEST.
        all_connection_descriptors = nest.GetConnections(source=inputs_ids, target=neuron_ids)

        # save connections to all neurons as attribute
        self.parrot_to_neuron_connection_descriptors = {
            neuron_id.tolist()[0]: nest.GetConnections(source=inputs_ids, target=neuron_id)
            for neuron_id in neuron_ids
        }

        network_objects_tuple = self._get_network_objects_tuple(has_teacher=True)

        self.network_objects = network_objects_tuple(
            neuron_ids=neuron_ids,
            inputs_ids=inputs_ids,
            generators_ids=generators_ids,
            teacher_ids=teacher_ids,
            all_connection_descriptors=all_connection_descriptors,
            spike_recorder_id=spike_recorder_id
        )

    def _most_frequent_class(self, votes):
        counter = Counter(votes)
        most_common = counter.most_common(1)
        if most_common:
            return int(most_common[0][0])
        else:
            return int(np.round(np.median(votes), 0))

    def _voltage_decoding(self, all_voltages, current_neuron, sample_time):
        return np.sum(all_voltages['V_m'][all_voltages['senders'] == current_neuron])

    def norm_weights(self):
        for neuron_id in self.network_objects.neuron_ids:
            this_neuron_input_synapses = nest.GetConnections(
                source=self.network_objects.inputs_ids, target=neuron_id
            )
            w = np.array(nest.GetStatus(this_neuron_input_synapses, "weight"))

            w *= self.weight_normalization * w.size / w.sum()
            
            w = np.clip(w, 0., 1.)

            nest.SetStatus(this_neuron_input_synapses, "weight", w)

    def norm_samples(self, X, testing_mode=False):
        assert len(X[X < 0]) == 0, "Input features cannot be negative."
        result = X / X.sum(axis=-1, keepdims=True) # L1-normalization
        # result /= self.sample_norm
        return result

    def get_weights(self):
        weights = np.asarray(
            nest.GetStatus(self.network_objects.all_connection_descriptors, 'weight')
        )
        
        weights = convert_neuron_ids_to_indices(
            weights,
            self.network_objects.all_connection_descriptors,
            self.network_objects.inputs_ids,
            self.network_objects.neuron_ids
        )

        return weights
        

    def run_the_simulation(self, X, y_train=None):

        testing_mode = y_train is None
        
        n_epochs = self.epochs if not testing_mode else 1

        record_weights = not testing_mode
        record_potentials = testing_mode
        early_stopping = self.early_stopping and not testing_mode

        if not testing_mode:
            # minmax scaling is necessary for this network
            active_neurons = self._bootstrap_samples(X, y_train)
        
        # X_s = self.norm_samples(X, testing_mode)
        
        progress_bar = tqdm(
            total=n_epochs * len(X),
            disable=self.quiet,
        )
        if early_stopping:
            previous_weights = np.asarray(
                [-1] * len(self.network_objects.all_connection_descriptors)
            )

        for epoch in range(n_epochs):

            self.encoder.epoch_time = epoch * len(X) * self.full_time

            input_data, teacher_data = self.encoder(X)

            if record_potentials:
                output_correlations = np.zeros((len(X), self.number_of_classes*self.n_estimators))

            for vector_number, (inp_time_list, teacher_list) in enumerate(zip(input_data, teacher_data)):

                nest.SetStatus(self.network_objects.generators_ids, inp_time_list)

                if not testing_mode:
                    nest.SetStatus(self.network_objects.teacher_ids, teacher_list)

                nest.SetStatus(self.network_objects.neuron_ids, {"V_m": self.E_L})
                nest.Simulate(self.sample_time)

                if record_potentials:
                    all_spikes = nest.GetStatus(self.network_objects.neuron_ids, 'V_m')[0]
                    output_correlations[vector_number] = all_spikes
            
                nest.Simulate(self.intervector_pause)
                progress_bar.update()

            if record_weights or early_stopping:
                weights = np.asarray(
                    nest.GetStatus(self.network_objects.all_connection_descriptors, 'weight')
                )

            if early_stopping:
                flag = self._early_stopping(weights, previous_weights)
                previous_weights = weights
                if flag:
                    break

        progress_bar.close()

        if record_weights:
            weights = np.asarray(
                nest.GetStatus(self.network_objects.all_connection_descriptors, 'weight')
            )

            if self.sigma_w:
                mu_w = np.mean(weights[weights>0])
                sigma_w = -self.sigma_w*np.std(weights[weights>0])
                #max_w = np.max(weights)

                weights[weights <= mu_w + sigma_w] = 0.0
                #weights /= max_w

            weights = convert_neuron_ids_to_indices(
                weights,
                self.network_objects.all_connection_descriptors,
                self.network_objects.inputs_ids,
                self.network_objects.neuron_ids
            )
            self.weights_ = weights

        if record_potentials:
            return output_correlations
    
    def predict(self, X):
        # assume that all samples belong to one class

        self._create_network(testing_mode=True)
        # Record what the last action has been,
        # in order to force re-creating the network
        # when switching actions.
        self.last_state_ = 'test'
        
        y_pred = np.zeros(len(X), dtype=np.int32)
        output_correlations = self.run_the_simulation(X, y_train=None).reshape(
            (len(X), 
             self.n_estimators, 
             self.number_of_classes))

        for i, s in enumerate(output_correlations):
            y_pred[i] = self._most_frequent_class(np.argmax(s, axis=1))
        
        return y_pred

    
