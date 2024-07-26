import os
os.environ["PYNEST_QUIET"] = "1"

from tqdm import tqdm
import numpy as np
import nest

from fsnn_classifiers.components.networks.base_classwise_bagging_network import BaseClasswiseBaggingNetwork
from fsnn_classifiers.components.networks.utils import (
    generate_random_state,
    convert_neuron_ids_to_indices,
)
from fsnn_classifiers.components.preprocessing.probabilistic_correlation_encoder import ProbabilisticCorrelationEncoder, spikes_to_times, get_time_dict

from collections import Counter

import os

from fsnn_classifiers.components.networks.common_model_components import disable_plasticity
from fsnn_classifiers.components.networks.utils import convert_random_parameters_to_nest

from random import sample, choices

class ProbabilisticCorrelationClasswiseNetwork(BaseClasswiseBaggingNetwork):
    def __init__(
        self,
        n_fields, # number of features before GRF
        n_estimators=1, # number of sub-networks
        max_features=1., # number of features for each sub-network
        max_samples=1., # number of samples for each sub-network
        bootstrap_features=False,
        synapse_model="stdp_nn_restr_synapse",
        V_th=-54.,
        t_ref=4.,
        tau_m=60.,
        tau_minus=10.0,
        tau_plus=10.0,
        I_exc=1e7,
        epochs=1,
        #time=1000,
        #intervector_pause=50.0, # pause in-between vectors
        Wmax=1.0, # max synaptic weight
        mu_plus=0.0,
        mu_minus=0.0,
        learning_rate = 0.01,
        tau_s=0.2,
        ref_seq_interval=5,
        rate = 500,
        resolution=0.1,
        sample_norm=1,
        w_init=1.0,
        sigma_w = 0.0,
        random_state=None,
        early_stopping=True,
        n_jobs=1,
        warm_start=False,
        quiet=True,
        **kwargs,
    ):
        
        super(ProbabilisticCorrelationClasswiseNetwork, self).__init__(
            n_fields=n_fields,
            synapse_model=synapse_model,
            V_th=V_th,
            t_ref=t_ref, 
            tau_m=tau_m,
            Wmax=Wmax,
            mu_plus=mu_plus,
            mu_minus=mu_minus,
            n_estimators=n_estimators,
            max_features=max_features,
            max_samples=max_samples,
            bootstrap_features=bootstrap_features,
            random_state=random_state,
            w_init=w_init,
        )

        
        
        self._check_modules(quiet)
        
        #try:
        #    nest.Install("probabilistic_neuron_module")
        #except:
        #    print("Probabilistic neuron loaded.")

        self.sigma_w = sigma_w
        self.learning_rate = learning_rate

        self.early_stopping = early_stopping
        self.n_jobs = n_jobs
        self.warm_start = warm_start
        self.quiet=quiet
        self.sample_norm = sample_norm
        self.resolution = resolution
        self.tau_minus = tau_minus
        self.tau_plus = tau_plus

        
        self.epochs = epochs
        
        #self.corr_time = corr_time
        self.tau_s = 0 #tau_s # shift time between input and teacher sequence (in ms)
        self.I_exc = I_exc
        self.ref_seq_interval = ref_seq_interval

        self.intervector_pause = 0#intervector_pause
        self.rate = rate#int(time/self.resolution)

        self.encoder = ProbabilisticCorrelationEncoder(rate=self.rate, resolution=self.resolution, min_spikes=1)

        #self.init_encoder(time)
        
    #def init_encoder(self, time):

        # some pre-computed values to reduce boilerplate code
        #self.time = time
        
        #self.full_time = self.time + self.intervector_pause

        #self.encoder = CorrelationEncoder(rate=self.rate, # does not change even if inference time is higher
        #                                  tau_s=self.tau_s, 
        #                                  time=self.time, 
        #                                  resolution=self.resolution, 
        #                                  interval=self.ref_seq_interval)
        
    def _get_parameters(self, number_of_inputs, number_of_classes, testing_mode):
        
        #neuron_parameters = {
        #    'V_th': self.V_th, 
        #    'E_L': -70.0, 
        #    'V_reset': -70.0,
        #    't_ref': 0.0, 
        #    'tau_m': 1e20,
        #    'tau_minus': self.tau_minus,
        #}

        neuron_parameters = {"I_syn":0.0}

        synapse_parameters = {
            'synapse_model': self.synapse_model,
        }

        if self.synapse_model not in ['stdp_tanh_synapse',
                             'stdp_gaussian_times_linear_with_separate_exp_r_dependence_synapse']:
            synapse_parameters["Wmax"] = self.Wmax
            synapse_parameters["mu_plus"] = self.mu_plus
            synapse_parameters["mu_minus"] = self.mu_minus
            synapse_parameters["tau_plus"] = self.tau_plus

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

        neuron_model = 'probabilistic_neuron'

        neuron_parameters, synapse_parameters = map(
            convert_random_parameters_to_nest,
            (neuron_parameters, synapse_parameters),
        )

        return (neuron_parameters, 
                synapse_parameters, 
                neuron_model, 
                feature_indices, 
                class_indices)

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

        try:
            nest.Install("probabilistic_neuron_module")
        except:
            print("Probabilistic neuron loaded.")


        (neuron_parameters, 
        synapse_parameters, 
        neuron_model, 
        feature_indices, 
        class_indices) = self._get_parameters(number_of_inputs, self.number_of_classes, testing_mode)

        if self.synapse_model in ["stdp_synapse", "stdp_nn_restr_synapse", "stdp_nn_symm_synapse"]:
            synapse_parameters["lambda"] = self.learning_rate

        #synapse_parameters = flip_plasticity(synapse_parameters)

        #self.E_L = neuron_parameters.get("E_L", -70.0) # just in case

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
                                                              create_spike_recorders=testing_mode)

        # connect neuron populations
        self._connect_neuron_populations(synapse_parameters=synapse_parameters,
                                         feature_indices=feature_indices,
                                         class_indices=class_indices,
                                         inputs_ids=inputs_ids,
                                         generators_ids=generators_ids,
                                         neuron_ids=neuron_ids,
                                         teacher_ids=teacher_ids,
                                         spike_recorder_id=spike_recorder_id)

        # Now that all connections have been created,
        # request their descriptors from NEST.
        all_connection_descriptors = nest.GetConnections(source=inputs_ids, target=neuron_ids)

        network_objects_tuple = self._get_network_objects_tuple(has_teacher=True)

        self.network_objects = network_objects_tuple(
            neuron_ids=neuron_ids,
            inputs_ids=inputs_ids,
            generators_ids=generators_ids,
            teacher_ids=teacher_ids,
            all_connection_descriptors=all_connection_descriptors,
            spike_recorder_id=spike_recorder_id
        )

    def _create_neuron_populations(self, 
                                   number_of_inputs,
                                   number_of_classes,
                                   neuron_model, 
                                   generator_type,
                                   neuron_parameters, 
                                   has_teacher=False,
                                   create_spike_recorders=False):

        # create neuron populations
        #if create_spike_recorders:
        neuron_ids = nest.Create(
            neuron_model,
            int(number_of_classes * self.n_estimators),
            params=neuron_parameters
        )
        #else:
        #    neuron_ids = nest.Create('parrot_neuron', number_of_classes*self.n_estimators)
        
        inputs_ids = nest.Create('parrot_neuron', number_of_inputs)

        if create_spike_recorders:
            spike_recorder_id = nest.Create('spike_recorder')
        else:
            spike_recorder_id = None

        #spike_recorder_id = nest.Create('spike_recorder')

        generators_ids = nest.Create(generator_type, number_of_inputs)

        if has_teacher:
            #teacher_ids = nest.Create("step_current_generator", 
            #                        number_of_classes * self.n_estimators)
            teacher_ids = nest.Create("spike_generator", number_of_classes * self.n_estimators)
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

        #if spike_recorder_id is None:
            #print("THIS")
        #    synapse_parameters["receptor_type"] = 1

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

        if spike_recorder_id is not None:
            nest.Connect(neuron_ids, spike_recorder_id, conn_spec='all_to_all')

        return None

    def _most_frequent_class(self, votes):
        counter = Counter(votes)
        most_common = counter.most_common(1)
        if most_common:
            return int(most_common[0][0])
        else:
            return int(np.round(np.median(votes), 0))
        
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
                                                    'I_e': 0. if current_neuron in active_neurons[vector_number] else -1e+3,
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
    
    def _get_teacher_dict(self, active_neurons, sample_time, I):
        start_times = self.encoder.ref_times + sample_time + self.tau_s
        end_times = start_times + self.resolution
        amplitude_times = sorted(list(start_times) + list(end_times))
        #assert len(set(amplitude_times)) == len(amplitude_times), "Teacher spikes are too close, increase <spike_ref_interval>."
        assert all(t > 0 for t in amplitude_times), "Amplitude times must be positive."
        amplitude_values = [I if i % 2 == 0 else 0 for i in range(len(amplitude_times))]

        return [{"amplitude_times":amplitude_times if current_neuron in active_neurons else [],
                 "amplitude_values":amplitude_values if current_neuron in active_neurons else [],
                 "allow_offgrid_times":True} 
                 for current_neuron in range(self.n_estimators*len(self.classes_))]

    def norm_samples(self, X, testing_mode=False):
        assert len(X[X < 0]) == 0, "Input features cannot be negative."
        result = X / X.sum(axis=-1, keepdims=True) # L1-normalization
        result /= self.sample_norm
        
        return result


    def run_the_simulation(self, X, y_train=None):

        testing_mode = y_train is None

        if not testing_mode:
            self.encoder.fit(X)
            self.time = self.encoder.time
            self.intervector_pause = 0 # later
            self.full_time = self.time + self.intervector_pause

        #if testing_mode:
        #    nest.SetStatus(self.network_objects.neuron_ids, {"V_th": 1000.0})
        #else:
            #nest.SetStatus(self.network_objects.neuron_ids, {"V_th": self.V_th})
        
        n_epochs = self.epochs if not testing_mode else 1

        record_weights = not testing_mode
        record_spikes = testing_mode
        early_stopping = self.early_stopping and not testing_mode

        if not testing_mode:
            # minmax scaling is necessary for this network
            active_neurons = self._bootstrap_samples(X, y_train)
            #inhibition_dict = self._generate_inhibition_dict(X, active_neurons)
        
        #X_s = self.norm_samples(X, testing_mode)
            
        
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

            for vector_number, x in enumerate(X):


                x = self.encoder.transform(x.reshape((1, -1))).reshape((X.shape[-1], -1)) # convert into spike sequences
                assert (np.count_nonzero(x) > 0), "Feature magnitude is too low. Increase time or decrease resolution."
                
                                
                #self.norm_weights()

                sample_time = epoch_time + vector_number * self.full_time

                inp_time_list = get_time_dict(spikes_to_times(inp_spikes=x, 
                                                              time=self.time, 
                                                              tau_s=sample_time, 
                                                              resolution=self.resolution))

                # The simulation itself.
                # set input spike times
                
                nest.SetStatus(self.network_objects.generators_ids, inp_time_list)

                if not testing_mode:

                    #teacher_list = self._get_teacher_dict(active_neurons[vector_number], sample_time, self.I_exc)
                    teacher_list = [{"spike_times":list(self.encoder.ref_times + sample_time + self.tau_s) 
                                     if current_neuron in active_neurons[vector_number] else []} 
                                    for current_neuron in range(self.n_estimators*len(self.classes_))]

                    nest.SetStatus(self.network_objects.teacher_ids, teacher_list)

                    neuron_list = [{"I_syn": 0.0 if current_neuron in active_neurons[vector_number] else -1000.0}
                                   for current_neuron in range(self.n_estimators*len(self.classes_))]
                    
                    nest.SetStatus(self.network_objects.neuron_ids, neuron_list)

                    #if vector_number == 10:
                    #    import pickle
                    #    import sys
                    #    with open("check_times.pkl", 'wb') as fp:
                    #        pickle.dump((inp_time_list, teacher_list, sample_time), fp)

                    #    sys.exit(1)

                    #if self.w_inh is None:
                    #nest.SetStatus(
                    #    self.network_objects.neuron_ids,
                    #    inhibition_dict[vector_number]
                    #)
                
                nest.Simulate(self.time)

                #if vector_number <= 10 and testing_mode:
                #    import pickle
                #    import sys

                #    all_spikes = nest.GetStatus(self.network_objects.spike_recorder_id, keys='events')[0]
                #    teacher_list = [{"spike_times":np.array(all_spikes["times"])[all_spikes['senders'] == current_neuron]} for current_neuron in range(self.n_estimators*len(self.classes_))]
                #    with open(f"check_times_test_{vector_number}.pkl", 'wb') as fp:
                #        pickle.dump((inp_time_list, teacher_list, sample_time), fp)

                #    weights = np.asarray(
                #    nest.GetStatus(self.network_objects.all_connection_descriptors, 'weight')
                #    )

                #    weights = convert_neuron_ids_to_indices(
                #        weights,
                #        self.network_objects.all_connection_descriptors,
                #        self.network_objects.inputs_ids,
                #        self.network_objects.neuron_ids,
                #        remove_zeros=True
                #    )
                    
                #    with open("weights.pkl", 'wb') as fp:
                #       pickle.dump(weights, fp)

                #if vector_number == 10 and testing_mode:

                #    sys.exit(1)

                if record_spikes:
                    
                    # NEST returns all_spikes == {
                    #   'times': spike_times_array,
                    #   'senders': senders_ids_array
                    # }

                    all_spikes = nest.GetStatus(self.network_objects.spike_recorder_id, keys='events')[0]
                    N = int(self.time/self.resolution)

                    #teacher_idxs = ((self.encoder.ref_times + sample_time + self.resolution)/self.resolution).astype(np.int32)
                    #teacher_spikes = np.zeros(N)
                    #teacher_spikes[teacher_idxs[teacher_idxs < N]] = 1

                    input_spikes = np.zeros(N)
                    inp_spike_times = [np.array(time_dict["spike_times"]) - sample_time for time_dict in inp_time_list if len(time_dict["spike_times"]) > 0]
                    if len(inp_spike_times) > 0:
                        inp_spike_times = np.hstack(inp_spike_times)
                        inp_spike_times.sort()
                        inp_spike_times = (inp_spike_times/self.resolution).astype(np.int32)
                        input_spikes[inp_spike_times[inp_spike_times < N]] = 1
                    
                    for n_idx, current_neuron in enumerate(self.network_objects.neuron_ids):
                        #output_correlations[vector_number, n_idx] = np.sum(all_spikes['senders'] == current_neuron) #self._decode_spikes(all_spikes, current_neuron, sample_time)
                        class_times = np.array(all_spikes["times"])[all_spikes['senders'] == current_neuron] - sample_time - 2.0
                        class_times = (class_times/self.resolution).astype(np.int32)
                        class_spikes = np.zeros(N)
                        class_spikes[class_times[class_times < N]] = 1
                        output_correlations[vector_number, n_idx] = np.sum(class_spikes * input_spikes)
                        
                    
                    # Empty the detector.
                    nest.SetStatus(self.network_objects.spike_recorder_id, {'n_events': 0})

                #nest.Simulate(self.intervector_pause)

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

            # clip weights based on mean and std

            mu_w = np.mean(weights[weights>0])
            sigma_w = self.sigma_w*np.std(weights[weights>0])

            weights[weights <= mu_w + sigma_w] = 0.0

            w_max = weights.max()
            w_min = weights.min()

            weights = (weights - w_min)/(w_max - w_min)

            #weights = np.ones_like(weights) # make all weights 1

            weights = convert_neuron_ids_to_indices(
                weights,
                self.network_objects.all_connection_descriptors,
                self.network_objects.inputs_ids,
                self.network_objects.neuron_ids,
                remove_zeros=True
            )
            self.weights_ = weights

        if record_spikes: 
            return output_correlations
        
    def transform(self, X):

        self._create_network(testing_mode=True)
        # Record what the last action has been,
        # in order to force re-creating the network
        # when switching actions.
        self.last_state_ = 'test'

        output_correlations = self.run_the_simulation(X, y_train=None).reshape(
            (len(X), 
             self.n_estimators, 
             self.number_of_classes))
        
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

    
