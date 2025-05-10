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
from fsnn_classifiers.components.preprocessing.one_spike_correlation_encoder import CorrelationEncoder, get_time_dict
from fsnn_classifiers.components.networks.common_model_components import flip_plasticity
from sklearn.preprocessing import minmax_scale, MinMaxScaler

from collections import Counter

import os
import random

import pickle


class CorrelationClasswiseNetwork(BaseClasswiseBaggingNetwork):
    def __init__(
        self,
        n_fields, # number of features before GRF
        n_estimators=1, # number of sub-networks
        max_features=1., # number of features for each sub-network
        max_samples=1., # number of samples for each sub-network
        bootstrap_features=False,
        synapse_model="stdp_nn_pre_centered_synapse",
        decoding="frequency",
        V_th=-54.,
        t_ref=0.0,
        tau_m=60.,
        tau_minus=60.,
        I_exc=1e7,
        epochs=1,
        time=1000,
        intervector_pause=0.0, # pause in-between vectors
        Wmax=1.0, # max synaptic weight
        tau_plus=20.,
        mu_plus=0.0,
        mu_minus=0.0,
        alpha=1.,
        # lambda_stdp=0.01,
        tau_s=0.2,
        corr_time=0.0, # correlation interval (ms)
        resolution=0.1,
        sample_norm=1,
        w_inh=None,
        w_init=1.0,
        sigma_w=None,
        weight_normalization=None,
        random_state=None,
        early_stopping=False,
        n_jobs=1,
        warm_start=False,
        quiet=True,
        job_directory=None,
        job_id=None,
        task_id=1,
        task_max=1,
        record_weights=False,
        use_entire_train_data=False,
        class_name=None,
        use_poisson=False,
        I_negative=-1e7,
        V_th_train=1e6,
        lr=0.001,
        correct_encoding=True,
        allow_offgrid_times=True,
        convolution_window=None,
        anti_correlation=False,
        every_input_spikes=True,
        ensemble_number=None,
        scaling_factor=1,
        **kwargs,
    ):
        
        super(CorrelationClasswiseNetwork, self).__init__(
            n_fields=n_fields,
            synapse_model=synapse_model,
            V_th=V_th,
            t_ref=t_ref, 
            tau_m=tau_m,
            tau_minus=tau_minus,
            Wmax=Wmax,
            tau_plus=tau_plus,
            mu_plus=mu_plus,
            mu_minus=mu_minus,
            alpha=alpha,
            # lambda_stdp=lambda_stdp,
            n_estimators=n_estimators,
            max_features=max_features,
            max_samples=max_samples,
            bootstrap_features=bootstrap_features,
            random_state=random_state,
            w_inh=w_inh,
            w_init=w_init,
            # weight_normalization=weight_normalization,
            use_entire_train_data=use_entire_train_data,
        )
        
        self._check_modules(quiet)

        self.sigma_w = sigma_w
        self.lr = lr

        self.early_stopping = early_stopping
        self.allow_offgrid_times = allow_offgrid_times
        self.n_jobs = n_jobs
        self.warm_start = warm_start
        self.quiet=quiet
        self.sample_norm = sample_norm
        self.anti_correlation = anti_correlation

        self.decoding = decoding
        # self._decode_spikes = self._time_decoding

        self.weight_normalization = weight_normalization
        self.job_directory = job_directory
        self.job_id = job_id
        self.task_id = task_id
        self.task_max = task_max
        self.record_weights = record_weights
        self.class_name = class_name
        self.use_poisson = use_poisson
        self.I_negative = I_negative
        self.V_th_train = V_th_train

        self.convolution_window = convolution_window
        self.ensemble_number = ensemble_number
        # self.parrot_to_neuron_connection_descriptors = {}

        # have to create these for sklearn

        self.resolution = resolution

        self.rate = int(time/self.resolution)
        self.epochs = epochs
        self.time = time
        self.intervector_pause = intervector_pause
        self.corr_time = corr_time
        self.tau_s = tau_s # shift time between input and teacher sequence (in ms)
        self.I_exc = I_exc
        self.every_input_spikes = every_input_spikes

        # some pre-computed values to reduce boilerplate code
        # self.exp_time = self.time + self.corr_time # sample exposure time (including correlation shift)
        self.exp_time = self.time + 1 # sample exposure time
        self.full_time = self.exp_time + self.intervector_pause
        self.exp_steps = int(self.exp_time / self.resolution)
        self.corr_steps = int(self.corr_time / self.resolution)
        self.steps = int(self.time / self.resolution)

        self.correct_encoding = correct_encoding
        if self.every_input_spikes:
            self.encoder = CorrelationEncoder(simulation_time=self.time)
        else:
            self.encoder = CorrelationEncoder(simulation_time=self.time, special_value=0)

    def _create_network(self, testing_mode):

        number_of_inputs = self.n_features_in_
        random_state = (
            self.random_state if not self.random_state is None
            else generate_random_state()
        )
        self.number_of_classes = len(self.classes_) if not self.class_name else 1

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
                                                              create_spike_recorders=True, # testing_mode,
                                                              time=self.time - 1
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

    # def _corr_integral(self, x_ij):
    #     # x_ij.shape = (time,)
        
    #     out = 0
    #     for i in range(self.corr_steps):
    #         s0_ = np.zeros(self.exp_steps)
    #         s0_[i:i+self.steps] = self.encoder.S0 # shift the reference frequency in time
    #         out += np.sum(s0_ * x_ij)

    #     return out

    def _most_frequent_class(self, votes):
        counter = Counter(votes)
        most_common = counter.most_common(1)
        if most_common:
            return int(most_common[0][0])
        else:
            return int(np.round(np.median(votes), 0))

    def _voltage_decoding(self, all_voltages, current_neuron, sample_time):
        return np.sum(all_voltages['V_m'][all_voltages['senders'] == current_neuron])
        
    def _frequency_decoding(self, all_spikes, current_neuron, sample_time):
        return np.sum(all_spikes['senders'] == current_neuron)
    
    def _correlation_decoding(self, all_spikes, current_neuron, sample_time):

        # convert output spike times into array indices
                        
        output_spike_times = all_spikes['times'][all_spikes['senders'] == current_neuron]
        
        output_spike_times = np.round((output_spike_times - sample_time)/self.resolution, 0).astype(int) - 1

        output_spike_times = output_spike_times[output_spike_times > 0]

        # convert array indices into the spike train
        output_spikes = np.zeros(self.exp_steps, dtype=np.uint8)

        if len(output_spike_times) > 0:
            output_spikes[output_spike_times] = 1
        
            # compute correlations
            return self._corr_integral(output_spikes)
        else:
            return 0
        
    def _generate_inhibition_dict(self, X, active_neurons):

        inhibition_dict = dict()
        for vector_number in range(len(X)):
            inhibition_dict[vector_number] = [
                                                {
                                                    # Inject negative stimulation current
                                                    # into all neurons that do not belong
                                                    # to the current class, so that to
                                                    # prevent them from spiking
                                                    # (and thus from learning
                                                    # the current class).
                                                    # 'I_e': 0. if current_neuron in active_neurons[vector_number] else self.I_negative,
                                                    # That current may have made the neuron's
                                                    # potential too negative.
                                                    # We reset the potential, so that previous
                                                    # stimulation not inhibit spiking
                                                    # in response to the current input.
                                                    'V_m': self.E_L,
                                                }
                                                for current_neuron in range(self.n_estimators*len(self.classes_))
                                            ]
            
        return inhibition_dict

    def _get_teacher_dict(self, active_neurons, sample_time, I, sample_spikes):
        start_times = sample_spikes + sample_time
        end_time = np.max(start_times) + self.resolution
        amplitude_times = np.round([end_time, end_time + self.resolution], 1)
        amplitude_values = [I, 0]

        return [{"amplitude_times": amplitude_times if current_neuron in active_neurons else [],
                 "amplitude_values": amplitude_values if current_neuron in active_neurons else [],
                 "allow_offgrid_times": self.allow_offgrid_times}
                 for current_neuron in range(self.n_estimators*len(self.classes_))]

    def norm_weights(self):

        for neuron_id in self.network_objects.neuron_ids:
            this_neuron_input_synapses = nest.GetConnections(
                source=self.network_objects.inputs_ids, target=neuron_id
            )
            w = np.array(nest.GetStatus(this_neuron_input_synapses, "weight"))

            w *= self.weight_normalization * w.size / w.sum()
            
            w = np.clip(w, 0., 1.)

            nest.SetStatus(this_neuron_input_synapses, "weight", w)

    def norm_samples(self, X):
        assert len(X[X < 0]) == 0, "Input features cannot be negative."
        result = X / X.max() # max-normalization
        
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

        if not testing_mode:
            nest.SetStatus(self.network_objects.neuron_ids, {"V_th": self.V_th_train})
        else:
            nest.SetStatus(self.network_objects.neuron_ids, {"V_th": self.V_th})
        
        n_epochs = self.epochs if not testing_mode else 1

        record_weights = not testing_mode
        record_spikes = testing_mode
        early_stopping = self.early_stopping and not testing_mode

        if not testing_mode:
            # minmax scaling is necessary for this network
            active_neurons = self._bootstrap_samples(X, y_train)
            inhibition_dict = self._generate_inhibition_dict(X, active_neurons)
        
        X_s = self.norm_samples(X)
        X_s = self.encoder(X_s) + 0.1
        
        progress_bar = tqdm(
            total=n_epochs * len(X),
            disable=self.quiet,
        )
        if early_stopping:
            previous_weights = np.asarray(
                [-1] * len(self.network_objects.all_connection_descriptors)
            )

        for epoch in range(n_epochs):

            epoch_time = epoch * len(X) * self.full_time
            
            if record_spikes:
                output_correlations = np.zeros((len(X), self.number_of_classes*self.n_estimators))

            for vector_number, x in enumerate(X_s):

                sample_time = epoch_time + vector_number * self.full_time
                inp_time_list = get_time_dict(x + sample_time, special_value=self.encoder.special_value)

                nest.SetStatus(self.network_objects.generators_ids, inp_time_list)

                if not testing_mode:

                    teacher_list = self._get_teacher_dict(active_neurons[vector_number], sample_time, self.I_exc, x)

                    nest.SetStatus(self.network_objects.teacher_ids, teacher_list)

                    if self.w_inh is None:
                        nest.SetStatus(
                            self.network_objects.neuron_ids,
                            inhibition_dict[vector_number]
                        )

                nest.Simulate(self.exp_time)

                if record_spikes:

                    all_spikes = nest.GetStatus(self.network_objects.spike_recorder_id, keys='events')[0]
                    
                    for n_idx, current_neuron in enumerate(self.network_objects.neuron_ids):
                        output_correlations[vector_number, n_idx] = all_spikes["senders"][0] if len(all_spikes["senders"]) > 0 else self.time + 0.1

                    # Empty the detector.
                    nest.SetStatus(self.network_objects.spike_recorder_id, {'n_events': 0})

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

        if record_spikes:
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
             self.n_estimators))

        for i, s in enumerate(output_correlations):
            y_pred[i] = self._most_frequent_class(np.argmin(s, axis=1))
        
        return y_pred

    
