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

from sklearn.preprocessing import minmax_scale, MinMaxScaler

from collections import Counter

import os


class CorrelationClasswiseNetwork(BaseClasswiseBaggingNetwork):
    def __init__(
        self,
        n_fields, # number of features before GRF
        n_estimators=1, # number of sub-networks
        max_features=1., # number of features for each sub-network
        max_samples=1., # number of samples for each sub-network
        bootstrap_features=False,
        synapse_model="stdp_nn_symm_synapse",
        decoding="frequency",
        V_th=-54.,
        t_ref=4.,
        tau_m=60.,
        I_exc=1e7,
        I_inh=-1.,
        spike_p=0.1,
        inp_mul=1,
        epochs=1,
        time=1000,
        intervector_pause=50.0, # pause in-between vectors
        Wmax=1.0, # max synaptic weight
        mu_plus=0.0,
        mu_minus=0.0,
        tau_s=0.2,
        corr_time=20.0, # correlation interval (ms)
        random_state=None,
        early_stopping=True,
        n_jobs=1,
        warm_start=False,
        quiet=True,
        log_weights=False,
        **kwargs,
    ):
        
        super(CorrelationClasswiseNetwork, self).__init__(
            n_fields=n_fields,
            synapse_model=synapse_model,
            V_th=V_th,
            t_ref=t_ref, 
            tau_m=tau_m,
            Wmax=Wmax,
            mu_plus=mu_plus,
            mu_minus=mu_minus,
            inp_mul=inp_mul,
            n_estimators=n_estimators,
            max_features=max_features,
            max_samples=max_samples,
            bootstrap_features=bootstrap_features,
            random_state=random_state,
        )
        
        self._check_modules(quiet)

        self.early_stopping = early_stopping
        self.n_jobs = n_jobs
        self.warm_start = warm_start
        self.quiet=quiet
        self.log_weights = log_weights

        self.decoding = decoding
        if self.decoding == "frequency":
            self._decode_spikes = self._frequency_decoding
        else:
            self._decode_spikes = self._correlation_decoding

        # have to create these for sklearn

        self.spike_p = spike_p
        self.epochs = epochs
        self.time = time
        self.intervector_pause = intervector_pause
        self.corr_time = corr_time
        self.tau_s = tau_s # shift time between input and teacher sequence (in ms)
        self.I_exc = I_exc
        self.I_inh = I_inh

        # some pre-computed values to reduce boilerplate code
        self.exp_time = self.time + self.corr_time # sample exposure time (including correlation shift)
        self.full_time = self.exp_time + self.intervector_pause
        self.exp_steps = int(self.exp_time / 0.1)
        self.corr_steps = int(self.corr_time / 0.1)
        self.steps = int(self.time / 0.1)

        self.scaler = MinMaxScaler()


    def _create_network(self, testing_mode):
        
        number_of_inputs = self.n_features_in_ * self.inp_mul
        random_state = (
            self.random_state if not self.random_state is None
            else generate_random_state()
        )
        self.number_of_classes = len(self.classes_)

        # Remove existing NEST objects if any exist.
        nest.ResetKernel()
        n_threads = self.n_jobs
        nest.SetKernelStatus({
            'resolution': 0.1,
            'local_num_threads': n_threads,
        })
        nest.rng_seed = random_state

        (neuron_parameters, 
        synapse_parameters, 
        neuron_model, 
        feature_indices, 
        class_indices) = self._get_parameters(number_of_inputs, self.number_of_classes, testing_mode)

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

        # S0 -- reference sequence (shape = (time,)), S1 -- adversarial sequence (shape = (time,))
        self.S0 = self._generate_reference_sequence(self.spike_p, self.steps)
        #self.S1 = self._generate_correlated_sequence(np.random.rand(len(self.S0)), self.spike_p, 0, ensure_pearson=False)

        # precompute teacher-related sequences for efficiency
        self.rnf_seq = np.repeat(self.S0.reshape((1,1,self.steps)), self.inp_mul, axis=1)
        #self.adv_seq = np.repeat(self.S1.reshape((1,1,self.steps)), self.inp_mul, axis=1)

        assert np.count_nonzero(self.S0) > 0, "No spikes in reference frequency. Please increase spike_p."

    def test_correlations(self,):
        seq = np.random.rand(len(self.S0))
        results = {}
        for c in np.arange(0.1, 1.0, 0.1):
            cur_cor = []
            for rep in range(100):
                der = self._generate_correlated_sequence(seq, self.spike_p, c)
                switch_num = int((1-c) * np.count_nonzero(der)) # how many spikes to erase
                switch_idx = np.where(der > 0)[0]
                np.random.shuffle(switch_idx)
                der[switch_idx[:switch_num]] = 0

                cor = np.corrcoef(der, self.S1)[0,1]
                cur_cor.append(cor)
            print(f"C={c}, cor={np.mean(cur_cor)}, f={np.sum(der)/len(der)}")
            results[c] = cur_cor
        return results

    def _generate_phi_theta(self, p, c):
        phi = p * (1 - c ** 0.5)
        theta = p + (1 - p) * c ** 0.5

        return phi, theta
    
    def _generate_reference_sequence(self, p, N):
        S0 = np.random.rand(N)

        return np.where(S0 < p, 1, 0).astype(np.uint8)
    
    def _generate_correlated_sequence(self, s, p, c, ensure_pearson=True):
        phi, theta = self._generate_phi_theta(p, c)
        der = np.where(((s < self.S0) & (s < theta)) | ((s >= self.S0) & (s < phi)), 1, 0)
        if ensure_pearson:
            switch_num = int((1-c) * np.count_nonzero(der)) # how many spikes to erase
            switch_idx = np.where(der > 0)[0]
            np.random.shuffle(switch_idx)
            der[switch_idx[:switch_num]] = 0
        return der.astype(np.uint8)

    def _vector_to_sequence(self, v, N, K):
        Ymatrix = np.random.rand(len(v), K, N)

        for i in range(len(v)):

            for j in range(K):
                Ymatrix[i, j] = self._generate_correlated_sequence(Ymatrix[i,j], self.spike_p, v[i])

        return Ymatrix.astype(np.uint8)
    
    def _spikes_to_times(self, Ymatrix, tau_s):

        time_list = [{'spike_times': []} for _ in range(np.prod(Ymatrix.shape[:2]))]

        d = 0
        for i in range(len(Ymatrix)):
            for j in range(len(Ymatrix[0])):
                time_list[d]['spike_times'] = (((np.where(Ymatrix[i, j] == 1)[0]) * 0.1 + 0.1).round(1) + tau_s).tolist()
                d += 1
        
        return time_list
    
    
    def _get_teacher_current(self, spike_times: list, I: float = 1e7):

        start_times = []
        end_times = []
        start_values = []
        end_values = []

        if len(spike_times) == 0:
            return {'amplitude_times':[], 
                'amplitude_values':[], 
                'allow_offgrid_times':True,
                }
        
        for i in range(len(spike_times)-1):
            if spike_times[i+1] - spike_times[i] > 1.1:
                # change the current
                start_times.append(spike_times[i])
                start_values.append(I)
                end_times.append(spike_times[i]+0.1)
                end_values.append(0.0)
            else:
                # keep the current
                start_times.append(spike_times[i])
                start_values.append(I)
                end_times.append(np.inf)
                end_values.append(np.inf)
                

        amplitude_times = np.stack((np.array(start_times),np.array(end_times)), axis=-1).flatten()
        amplitude_values = np.stack((np.array(start_values),np.array(end_values)), axis=-1).flatten()

        mask = np.isfinite(amplitude_times)
        
        return {'amplitude_times':amplitude_times[mask].astype(float), 
                'amplitude_values':amplitude_values[mask].astype(float), 
                'allow_offgrid_times':True,
                }

    def _corr_integral(self, x_ij):
        # x_ij.shape = (time,)
        
        out = 0
        for i in range(self.corr_steps):
            s0_ = np.zeros(self.exp_steps)
            s0_[i:i+self.steps] = self.S0 # shift the reference frequency in time
            out += np.sum(s0_ * x_ij)

        return out
    
    def _most_frequent_class(self, votes):
        counter = Counter(votes)
        most_common = counter.most_common(1)
        if most_common:
            return int(most_common[0][0])
        else:
            return int(np.round(np.median(votes), 0))
        
    def _frequency_decoding(self, all_spikes, current_neuron, sample_time):
        return np.sum(all_spikes['senders'] == current_neuron)
    
    def _correlation_decoding(self, all_spikes, current_neuron, sample_time):

        # convert output spike times into array indices
                        
        output_spike_times = all_spikes['times'][all_spikes['senders'] == current_neuron]
        
        output_spike_times = np.round((output_spike_times - sample_time)/0.1, 0).astype(int) - 1

        output_spike_times = output_spike_times[output_spike_times > 0]

        # convert array indices into the spike train
        output_spikes = np.zeros(self.exp_steps, dtype=np.uint8)

        if len(output_spike_times) > 0:
            output_spikes[output_spike_times] = 1
        
            # compute correlations
            return self._corr_integral(output_spikes)
        else:
            return 0
        
    def run_the_simulation(self, X, y_train=None):

        if self.log_weights:
            weights = np.asarray(
            nest.GetStatus(self.network_objects.all_connection_descriptors, 'weight')
        )
            save_dir = f"{os.getcwd()}/ccn_weights/"
            os.makedirs(save_dir, exist_ok=True)
            with open(f"{save_dir}/weight_0_0.npy", 'wb') as fp:
                np.save(fp, weights)

        testing_mode = y_train is None
        
        n_epochs = self.epochs if not testing_mode else 1

        record_weights = not testing_mode
        record_spikes = testing_mode
        early_stopping = self.early_stopping and not testing_mode

        if not testing_mode:
            # minmax scaling is necessary for this network
            X_s = self.scaler.fit_transform(X)
            active_neurons = self._bootstrap_samples(X, y_train)
        else:
            X_s = self.scaler.transform(X)

        X_s = np.clip(X_s, 0, 1)
        
        progress_bar = tqdm(
            total=n_epochs * len(X),
            disable=self.quiet,
        )
        if early_stopping:
            previous_weights = np.asarray(
                [-1] * len(self.network_objects.all_connection_descriptors)
            )

        if self.log_weights and not testing_mode:
            input_freqs = np.zeros((self.number_of_classes, self.n_features_in_))

        for epoch in range(n_epochs):

            epoch_time = epoch * len(X) * self.full_time
            
            if record_spikes:
                output_correlations = np.zeros((len(X), self.number_of_classes*self.n_estimators))

            for vector_number, x in enumerate(X_s):

                sample_time = epoch_time + vector_number * self.full_time

                input_spike_trains = self._vector_to_sequence(x, self.steps, self.inp_mul)

                if self.log_weights and not testing_mode:
                    input_freqs[y_train[vector_number]] += input_spike_trains.sum(axis=(-2,-1))

                inp_time_list = self._spikes_to_times(input_spike_trains, sample_time)
                rnf_time_list = self._spikes_to_times(self.rnf_seq, 
                                                      sample_time+self.tau_s)
                #adv_time_list = self._spikes_to_times(self.adv_seq, 
                #                                      sample_time+self.tau_s)


                # The simulation itself.
                # set input spike times
                nest.SetStatus(self.network_objects.generators_ids, inp_time_list)

                if not testing_mode:

                    teacher_dict = {}
            
                    nest.SetStatus(
                        self.network_objects.neuron_ids,
                        [
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
                    )

                    for current_neuron, cur_teacher_id in enumerate(self.network_objects.teacher_ids):
                        # for current class neurons -- teacher signal is the time-shifted reference sequence
                        # for other classes -- the time-shifted adversarial sequence with negative current
                        if current_neuron in active_neurons[vector_number]:
                            teacher_dict[cur_teacher_id.get('global_id')]=self._get_teacher_current(rnf_time_list[0]['spike_times'], I=self.I_exc)
                        else:
                            teacher_dict[cur_teacher_id.get('global_id')]=self._get_teacher_current(rnf_time_list[0]['spike_times'], I=0)


                    self.network_objects.teacher_ids.set(list(teacher_dict.values()))
                
                nest.Simulate(self.exp_time)

                if record_spikes:
                    # NEST returns all_spikes == {
                    #   'times': spike_times_array,
                    #   'senders': senders_ids_array
                    # }
                   
                    all_spikes = nest.GetStatus(self.network_objects.spike_recorder_id, keys='events')[0]
                    
                    for n_idx, current_neuron in enumerate(self.network_objects.neuron_ids):
                        output_correlations[vector_number, n_idx] = self._decode_spikes(all_spikes, current_neuron, sample_time)
                    
                    # Empty the detector.
                    nest.SetStatus(self.network_objects.spike_recorder_id, {'n_events': 0})
                   
                 # reset generators
                nest.SetStatus(self.network_objects.generators_ids, {"spike_times": []})
                nest.SetStatus(self.network_objects.teacher_ids, {"amplitude_times": [], "amplitude_values":[]})

                nest.Simulate(self.intervector_pause)

                progress_bar.update()

                if self.log_weights and not testing_mode:
                    weights = np.asarray(
                    nest.GetStatus(self.network_objects.all_connection_descriptors, 'weight')
                )
                    save_dir = f"{os.getcwd()}/ccn_weights/"
                    os.makedirs(save_dir, exist_ok=True)
                    with open(f"{save_dir}/weight_{epoch}_{vector_number+1}.npy", 'wb') as fp:
                        np.save(fp, weights)

                    if epoch > 0:
                        with open(f"{save_dir}/mean_train_rates.npy", 'wb') as fp:
                            np.save(fp, input_freqs)

            if record_weights or early_stopping:
                weights = np.asarray(
                    nest.GetStatus(self.network_objects.all_connection_descriptors, 'weight')
                )

            if early_stopping:
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
                    break
                if np.logical_or(
                    weights < 0.1,
                    weights > 0.9
                ).all():
                    print('Early stopping on weights convergence to 0 or 1.')
                    break
                previous_weights = weights
        progress_bar.close()

        if record_weights:
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
             self.n_estimators, 
             self.number_of_classes))

        for i, s in enumerate(output_correlations):
            y_pred[i] = self._most_frequent_class(np.argmax(s, axis=1))
        
        return y_pred

    