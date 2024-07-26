import os
os.environ["PYNEST_QUIET"] = "1"

import numpy as np
import nest
import random

from collections import namedtuple

from fsnn_classifiers.components.networks.base_spiking_transformer import BaseSpikingTransformer
from fsnn_classifiers.components.networks.utils import convert_random_parameters_to_nest, generate_random_state
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm

class Reservoir(BaseSpikingTransformer):

    def __init__(self, 
                 V_th=-54.,
                 t_ref=4.,
                 tau_m=60.,
                 r=1.885, 
                 A=0.3, 
                 B=5.9,
                 w_min=0.0,
                 w_max=1.0,
                 weight_type = 'log',
                 w_init = None,
                 out_dim = 100, 
                 high_rate = 600,
                 low_rate = 0.1,
                 minmax_scale = False,
                 quiet=False,
                 time=100, 
                 n_jobs=1,
                 warm_start=False,
                 random_state=None):
        
        self.V_th = V_th
        self.t_ref = t_ref
        self.tau_m = tau_m
        self.w_init = w_init

        self.r = r
        self.A = A
        self.B = B
        self.out_dim = out_dim
        self.w_min = w_min
        self.w_max = w_max

        self.high_rate = high_rate
        self.low_rate = low_rate

        self.minmax_scale = minmax_scale
        self.quiet = quiet

        self.scaler = MinMaxScaler()

        self.weight_type = weight_type
        self.warm_start = warm_start

        if weight_type == 'log' and w_init is None:
            print("Using logarithmic weight initialization")
            self._init_weights = self.log_func_weights
        elif w_init is not None:
            print("Using provided weights.")
        else:
            print("Using random weight initialization")
            self._init_weights = self.random_weights
        
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.time = time
        self.is_fitted = False

    def _get_network_objects_tuple(self):

        objects = ['neuron_ids',
                    'generators_ids',
                    'inputs_ids',
                    'all_connection_descriptors',
                    'spike_recorder_id']

        return namedtuple(
                            'network_objects_tuple',
                            tuple(objects)
                        )

    def _get_parameters(self, number_of_inputs):
        
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

        neuron_parameters = convert_random_parameters_to_nest(neuron_parameters)

        if self.w_init is None:
            self.weights_ = np.array(
                [list(self._init_weights(i)) 
                for i in range(number_of_inputs)])
        else:
            self.weights_ = self.w_init
        
        return neuron_parameters
    
    def _create_neuron_populations(self, number_of_inputs, neuron_parameters, create_spike_recorders=False):

        generators_ids = nest.Create('poisson_generator', number_of_inputs)
        inputs_ids = nest.Create('parrot_neuron', number_of_inputs)
        neuron_ids = nest.Create('iaf_psc_exp', self.out_dim, params=neuron_parameters)
        if create_spike_recorders:
            spike_recorder_id = nest.Create('spike_recorder')
        else:
            spike_recorder_id = None

        return (generators_ids, inputs_ids, neuron_ids, spike_recorder_id)
    
    def _connect_neuron_populations(self, generators_ids, inputs_ids, neuron_ids, spike_recorder_id):

        # connect neuron populations
        nest.Connect(
            pre=generators_ids,
            post=inputs_ids,
            conn_spec="one_to_one",
            syn_spec="static_synapse",
        )

        nest.Connect(generators_ids, 
                     neuron_ids, 
                     syn_spec={'weight': np.transpose(self.weights_)})
        
        if spike_recorder_id is not None:
            nest.Connect(neuron_ids, spike_recorder_id, conn_spec='all_to_all')

        return None
    
    def _create_network(self, testing_mode):

        number_of_inputs = self.n_features_in_
        random_state = (
            self.random_state if not self.random_state is None
            else generate_random_state())
        

        # Remove existing NEST objects if any exist.
        nest.set_verbosity("M_ERROR")
        nest.ResetKernel()
        n_threads = self.n_jobs
        nest.SetKernelStatus({
            'resolution': 0.1,
            'local_num_threads': n_threads,
        })
        nest.rng_seed = random_state

        neuron_parameters = self._get_parameters(number_of_inputs)

        (generators_ids, 
         inputs_ids, 
         neuron_ids, 
         spike_recorder_id) = self._create_neuron_populations(number_of_inputs, 
                                                              neuron_parameters,
                                                              testing_mode)
        
        self._connect_neuron_populations(generators_ids, inputs_ids, neuron_ids, spike_recorder_id)

        all_connection_descriptors = nest.GetConnections(source=inputs_ids, target=neuron_ids)

        network_objects_tuple = self._get_network_objects_tuple()

        self.network_objects = network_objects_tuple(
            neuron_ids=neuron_ids,
            inputs_ids=inputs_ids,
            generators_ids=generators_ids,
            all_connection_descriptors=all_connection_descriptors,
            spike_recorder_id=spike_recorder_id
        )
        return None
    
    def _to_spike_rates(self, X):
        return np.exp(X) * (self.high_rate - self.low_rate) + self.low_rate
    
    def run_the_simulation(self, X, y_train=None):
        testing_mode = y_train is None

        #if len(np.ravel(X[X<0])) > 0 or self.minmax_scale:
        #    print("Inputs will be scaled to (0,1) range.")
        #    if not testing_mode:
        #        X = self.scaler.fit_transform(X)
        #    else:
        #        X = self.scaler.transform(X)

        if testing_mode:

            input_spike_rates = self._to_spike_rates(X)

            progress_bar = tqdm(
                total=len(input_spike_rates),
                disable=self.quiet,
            )

            output_spiking_rates = np.zeros((len(input_spike_rates), self.out_dim))

            for vector_number, x in enumerate(input_spike_rates):
                
                # The simulation itself.
                nest.SetStatus(self.network_objects.generators_ids, [{'rate': r} for r in x])
                nest.Simulate(self.time)

                
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
                
            progress_bar.close()

            return output_spiking_rates

    
    def log_func_weights(self, i=None):
        for j in range(self.out_dim):
            if j==0:
                prev = self.A*np.sin(i*np.pi/(self.out_dim*self.B))
                yield prev
            else:
                prev=1.0-self.r*(prev**2.0)
                yield prev

    def random_weights(self, i=None):
        return [random.uniform(self.w_min, self.w_max) for _ in range(self.out_dim)]


if __name__ == "__main__":

    from sklearn.datasets import load_iris
    from sklearn.preprocessing import Normalizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_validate
    from sklearn.pipeline import make_pipeline

    X, y = load_iris(return_X_y=True)

    res = Reservoir(out_dim=100, high_rate=600, quiet=True)
    prp = Normalizer(norm='l2')
    log = LogisticRegression()

    pipe = make_pipeline(prp,res,log)

    result = cross_validate(pipe, X, y, cv=5, scoring='f1_micro')

    for key, val in result.items():
        print(key)
        print(val)
