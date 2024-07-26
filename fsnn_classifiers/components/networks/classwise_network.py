import os
os.environ["PYNEST_QUIET"] = "1"

from tqdm import tqdm
import numpy as np
import nest
from copy import deepcopy

from fsnn_classifiers.components.networks.base_spiking_transformer import BaseSpikingTransformer
from fsnn_classifiers.components.networks.common_model_components import disable_plasticity
from fsnn_classifiers.components.networks.utils import (
    generate_random_state,
    convert_neuron_ids_to_indices,
    convert_random_parameters_to_nest
)

from fsnn_classifiers.components.networks.configs.classwise_network import snn_parameters, network_objects_tuple


class ClasswiseNetwork(BaseSpikingTransformer):
    def __init__(
        self,
        synapse_model="stdp_nn_restr_synapse",
        V_th=-54.,
        t_ref=4.,
        tau_m=60.,
        high_rate=500.,
        low_rate=0.11,
        epochs=1,
        random_state=None,
        early_stopping=True,
        n_jobs=1,
        warm_start=False,
        quiet=True,
        **kwargs,
    ):
        
        self._check_modules(quiet)
        _parameters = self._get_initial_parameters()
        
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.n_jobs = n_jobs
        self.warm_start = warm_start
        self.quiet=quiet

        # have to create these for sklearn
        self.synapse_model = synapse_model
        self.V_th = V_th
        self.t_ref = t_ref
        self.tau_m = tau_m
        self.high_rate = high_rate
        self.low_rate = low_rate
        self.epochs = epochs

        ## set base arguments
        # neuron parameters
        for key, val in zip(['V_th', 't_ref', 'tau_m'], [V_th, t_ref, tau_m]):
            _parameters['neuron_parameters'][key] = val

       
        _parameters["synapse_parameters"]["synapse_model"] = synapse_model

        if synapse_model != 'stdp_nn_restr_synapse':
            # remove extra parameters for memristive synapse models
            _parameters["synapse_parameters"]['input_to_exc'].pop("Wmax", None)
            _parameters["synapse_parameters"]['input_to_exc'].pop("alpha", None)
            _parameters["synapse_parameters"]['input_to_exc'].pop("lambda", None)
            _parameters["synapse_parameters"]['input_to_exc'].pop("mu_plus", None)
            _parameters["synapse_parameters"]['input_to_exc'].pop("mu_minus", None)
            _parameters["synapse_parameters"]['input_to_exc'].pop("tau_plus", None)

        # set network parameters
        _parameters['network_parameters']['high_rate'] = high_rate
        _parameters['network_parameters']['low_rate'] = low_rate
        _parameters['network_parameters']['epochs'] = int(epochs)

        ## set optional keyword arguments
        # must be in format parameter_type+key1+key2+...
        parameter_types = ['network', 'neuron', 'synapse']
        for par_str, val in kwargs.items():
            location = par_str.split('+')
            assert location[0] in parameter_types, "{par_str} -- wrong format!"
            assert len(location) in [2,3], f"Wrong number of parameter keys: {len(location)}!"
            if len(location) == 2:
                _parameters[f"{location[0]}_parameters"][location[1]] = val
            else:
                _parameters[f"{location[0]}_parameters"][location[1]][location[2]] = val

        # initialize parameters
        self.network_parameters = _parameters['network_parameters']
        self.neuron_parameters = _parameters['neuron_parameters']
        self.synapse_parameters = _parameters['synapse_parameters']

    def _create_network(self, testing_mode):
        number_of_inputs = self.n_features_in_
        random_state = (
            self.random_state if not self.random_state is None
            else generate_random_state()
        )
        number_of_classes = len(self.classes_)
        create_spike_recorders = testing_mode
        # Make a copy because we will tamper with synapse_parameters
        # if creating connections with pre-recorded weights.
        # Also, run nest.CreateParameter on those parameters
        # that are dictionaries describing random distributions. 
        neuron_parameters, synapse_parameters = map(
            convert_random_parameters_to_nest,
            (self.neuron_parameters, self.synapse_parameters)
        )
        if testing_mode:
            synapse_parameters = disable_plasticity(synapse_parameters)

        # Remove existing NEST objects if any exist.
        nest.ResetKernel()
        n_threads = self.n_jobs
        nest.SetKernelStatus({
            'resolution': 0.1,
            'local_num_threads': n_threads,
        })
        nest.rng_seed = random_state

        neuron_ids = nest.Create(
            self.network_parameters['neuron_model'],
            number_of_classes,
            params=self.neuron_parameters
        )
        generators_ids = nest.Create('poisson_generator', number_of_inputs)
        inputs_ids = nest.Create('parrot_neuron', number_of_inputs)
        if create_spike_recorders:
            spike_recorder_id = nest.Create('spike_recorder')

        # Create connections.
        nest.Connect(
            pre=generators_ids,
            post=inputs_ids,
            conn_spec='one_to_one',
            syn_spec='static_synapse'
        )
        if hasattr(self, 'weights_'):
            synapse_parameters.update(weight=self.weights_['weight'])
            nest.Connect(
                pre=np.array(inputs_ids)[self.weights_['pre_index']],
                post=np.array(neuron_ids)[self.weights_['post_index']],
                conn_spec='one_to_one',
                syn_spec=synapse_parameters
            )        
        else:
            nest.Connect(
                pre=inputs_ids,
                post=neuron_ids,
                conn_spec='all_to_all',
                syn_spec=synapse_parameters
            )
        if create_spike_recorders:
            nest.Connect(neuron_ids, spike_recorder_id, conn_spec='all_to_all')

        # Now that all connections have been created,
        # request their descriptors from NEST.
        all_connection_descriptors = nest.GetConnections(source=inputs_ids, target=neuron_ids)
        self.network_objects = network_objects_tuple(
            neuron_ids=neuron_ids,
            generators_ids=generators_ids,
            inputs_ids=inputs_ids,
            all_connection_descriptors=all_connection_descriptors,
            spike_recorder_id=spike_recorder_id if create_spike_recorders else None
        )

    def _to_spike_rates(self, X):
        return X * (self.high_rate - self.low_rate) + self.low_rate

    def _get_initial_parameters(self):
        return deepcopy(snn_parameters)

    def run_the_simulation(self, X, y_train=None):
        testing_mode = y_train is None
        n_epochs = self.network_parameters['epochs'] if not testing_mode else 1
        input_spike_rates = self._to_spike_rates(X)
        record_weights = not testing_mode
        record_spikes = testing_mode
        early_stopping = self.early_stopping and not testing_mode

        progress_bar = tqdm(
            total=n_epochs * len(input_spike_rates),
            disable=self.quiet,
        )
        if early_stopping:
            previous_weights = np.asarray(
                [-1] * len(self.network_objects.all_connection_descriptors)
            )
        for epoch in range(n_epochs):
            if record_spikes:
                #output_spiking_rates = []
                output_spiking_rates = np.zeros((len(input_spike_rates), len(self.classes_)))
            for vector_number, x, in enumerate(input_spike_rates):
                # The simulation itself.
                nest.SetStatus(self.network_objects.generators_ids, [{'rate': r} for r in x])
                if not testing_mode:
                    y = y_train[vector_number]
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
                                'I_e': 0. if current_neuron == y else -1e+3,
                                # That current may have made the neuron's
                                # potential too negative.
                                # We reset the potential, so that previous
                                # stimulation not inhibit spiking
                                # in response to the current input.
                                'V_m': self.neuron_parameters['E_L'],
                            }
                            for current_neuron in self.classes_
                        ]
                    )
                nest.Simulate(self.network_parameters['one_vector_longtitude'])

                if record_spikes:
                    # NEST returns all_spikes == {
                    #   'times': spike_times_array,
                    #   'senders': senders_ids_array
                    # }
                    all_spikes = nest.GetStatus(self.network_objects.spike_recorder_id, keys='events')[0]
                    #current_input_vector_output_rates = [
                        # 1000.0 * len(all_spikes['times'][
                        #   all_spikes['senders'] == current_neuron
                        # ]) / network_parameters['one_vector_longtitude']
                   #     len(all_spikes['times'][
                   #         all_spikes['senders'] == current_neuron
                   #     ])
                   #     for current_neuron in self.network_objects.neuron_ids
                    #]
                    
                    #current_input_vector_output_rates = np.zeros(len(self.classes_))
                    
                    for n_idx, current_neuron in enumerate(self.network_objects.neuron_ids):
                        output_spiking_rates[vector_number, n_idx] = np.sum(all_spikes['senders'] == current_neuron)
                    
                    # Empty the detector.
                    nest.SetStatus(self.network_objects.spike_recorder_id, {'n_events': 0})
                    #output_spiking_rates.append(
                    #    current_input_vector_output_rates
                    #)
                    
                    #output_spiking_rates[vector_number,:] = current_input_vector_output_rates                  
                progress_bar.update()
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
            return output_spiking_rates

if __name__ == "__main__":

    from sklearn.datasets import load_iris
    X, y = load_iris(as_frame=False, return_X_y=True)

    network = ClasswiseNetwork(quiet=False)

    # test all main methods
    network.fit(X, y)

    X_ = network.transform(X)
    X_ = network.fit_transform(X, y)