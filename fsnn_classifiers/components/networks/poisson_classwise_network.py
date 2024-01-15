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

from sklearn.preprocessing import MinMaxScaler


class PoissonClasswiseNetwork(BaseClasswiseBaggingNetwork):
    def __init__(
        self,
        n_fields, # number of features before GRF
        n_estimators=1, # number of sub-networks
        max_features=1., # number of features for each sub-network
        max_samples=1., # number of samples for each sub-network
        bootstrap_features=False,
        inp_mul=1,
        synapse_model="stdp_nn_restr_synapse",
        V_th=-54.,
        t_ref=4.,
        tau_m=60.,
        high_rate=500.,
        low_rate=0.11,
        epochs=1,
        time=1000,
        Wmax=1.0, # max synaptic weight
        mu_plus=0.0,
        mu_minus=0.0,
        minmax_scale=False,
        random_state=None,
        early_stopping=True,
        n_jobs=1,
        warm_start=False,
        quiet=True,
        **kwargs,
    ):
        
        super(PoissonClasswiseNetwork, self).__init__(
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
        self.inp_mul = inp_mul

        # have to create these for sklearn
        self.high_rate = high_rate
        self.low_rate = low_rate
        self.epochs = epochs
        self.time = time
        self.minmax_scale = minmax_scale

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

        self.E_L = neuron_parameters.get("E_L", -70.0)

        # create neuron populations
        (inputs_ids, 
         generators_ids, 
         neuron_ids, 
         teacher_ids, 
         spike_recorder_id) = self._create_neuron_populations(number_of_inputs=number_of_inputs, 
                                                              number_of_classes=self.number_of_classes, 
                                                              neuron_model=neuron_model,
                                                              generator_type="poisson_generator",
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

        network_objects_tuple = self._get_network_objects_tuple(has_teacher=False)

        self.network_objects = network_objects_tuple(
            neuron_ids=neuron_ids,
            inputs_ids=inputs_ids,
            generators_ids=generators_ids,
            teacher_ids=teacher_ids,
            all_connection_descriptors=all_connection_descriptors,
            spike_recorder_id=spike_recorder_id
        )

    def _to_spike_rates(self, X):
        return X * (self.high_rate - self.low_rate) + self.low_rate

    def run_the_simulation(self, X, y_train=None):
        testing_mode = y_train is None
        n_epochs = self.epochs if not testing_mode else 1

        if len(np.ravel(X[X<0])) > 0 or self.minmax_scale:
            print("Inputs will be scaled to (0,1) range.")
            if not testing_mode:
                X = self.scaler.fit_transform(X)
            else:
                X = self.scaler.transform(X)

        input_spike_rates = self._to_spike_rates(X)

        record_weights = not testing_mode
        record_spikes = testing_mode
        early_stopping = self.early_stopping and not testing_mode

        if not testing_mode:
            active_neurons = self._bootstrap_samples(X, y_train)

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

                output_spiking_rates = np.zeros((len(input_spike_rates), len(self.classes_)*self.n_estimators))
            for vector_number, x in enumerate(input_spike_rates):
                
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
                nest.Simulate(self.time)

                if record_spikes:
                    # NEST returns all_spikes == {
                    #   'times': spike_times_array,
                    #   'senders': senders_ids_array
                    # }
                    all_spikes = nest.GetStatus(self.network_objects.spike_recorder_id, keys='events')[0]
                    
                    for n_idx, current_neuron in enumerate(self.network_objects.neuron_ids):
                        output_spiking_rates[vector_number, n_idx] = np.sum(all_spikes['senders'] == current_neuron)
                    
                    # Empty the detector.
                    nest.SetStatus(self.network_objects.spike_recorder_id, {'n_events': 0})
                  
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
    from fsnn_classifiers.components.preprocessing import GRF
    from fsnn_classifiers.components.decoding.own_rate_population_decoder import OwnRatePopulationDecoder
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_validate, StratifiedKFold
    from sklearn.preprocessing import MinMaxScaler, Normalizer

    def run_experiment(epochs, n_estimators, plasticity, quiet=True):

        X, y = load_iris(as_frame=False, return_X_y=True)

        n_fields = 15

        preprocessor = Normalizer(norm='l2')
        encoder = GRF(n_fields=n_fields)
        network = PoissonClasswiseNetwork(n_fields=n_fields,
                                            n_estimators=n_estimators,
                                            synapse_model=plasticity,
                                            tau_m=70.0,
                                            epochs=epochs,
                                            boostrap_features=False,
                                            max_features=1.0,
                                            max_samples=0.7,
                                            t_ref=9.0,
                                            high_rate=800,
                                            time=1200,
                                            early_stopping=True,
                                            minmax_scale=False, 
                                            quiet=quiet)
        decoder = OwnRatePopulationDecoder()


        pipe = Pipeline([ ('prp', preprocessor),
                        ('enc', encoder), 
                        ('net', network),
                        ('dec', decoder)
                        ])
        
        
        cv_results = cross_validate(pipe, 
                                    X, 
                                    y, 
                                    cv=StratifiedKFold(n_splits=5),
                                    scoring='f1_micro'
                                    )
        
        return cv_results
    
    epochs = 1
    n_estimators = 301

    quiet = False
    print_all = True

    plasticity = "stdp_tanh_synapse"

    
    cv_results = run_experiment(epochs, n_estimators, plasticity, quiet)

    if print_all:
        for i in range(5):
            print(f"FOLD {i} (f1-micro): {cv_results['test_score'][i]}")
    print(f"AVG: {np.mean(cv_results['test_score'])} \pm {np.std(cv_results['test_score'])}")