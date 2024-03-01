
import os
os.environ["PYNEST_QUIET"] = "1"

from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm
import numpy as np


import nest

from collections import namedtuple

from fsnn_classifiers.components.networks.base_spiking_transformer import BaseSpikingTransformer
from fsnn_classifiers.components.networks.utils import (
    generate_random_state,
    convert_neuron_ids_to_indices,
    convert_random_parameters_to_nest
)
from fsnn_classifiers.components.networks.common_model_components import disable_plasticity


class DiehlNetwork(BaseSpikingTransformer):
    def __init__(
        self,
        synapse_model="stdp_nn_symm_synapse",
        V_th=(-50.0, -40.0),  # (exc_neurons, inh_neurons)
        t_ref=(5.0, 2.0),  # (exc_neurons, inh_neurons)
        tau_m=(100.0, 10.0),  # (exc_neurons, inh_neurons)
        w_syn=(10.4, -17.0),  # (exc_to_inh, inh_to_exc)
        n_neurons=(100, 100),  # (exc_neurons, inh_neurons)
        p=(None, None, None),  # sparcity prob (gen-exc, exc-inh, inh-exc)
        r=(None, None, None),  # sparcity radius (gen-exc, exc-inh, inh-exc)
        spatial="free",
        alpha=0.55,
        nu=0.01,
        high_rate=43.379,
        low_rate=0.018,
        epochs=1,
        time=350,
        intervector_pause=50.0,
        random_state=None,
        n_jobs=1,
        warm_start=False,
        quiet=True,
        weight_normalization=None,
        **kwargs,
    ):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.warm_start = warm_start
        self.quiet = quiet

        # have to create these for sklearn
        self.V_th = V_th
        self.t_ref = t_ref
        self.tau_m = tau_m
        self.w_syn = w_syn
        self.n_neurons = n_neurons
        self.alpha = alpha
        self.nu = nu
        self.high_rate = high_rate
        self.low_rate = low_rate
        self.time = time
        self.epochs = epochs
        self.synapse_model = synapse_model
        self.intervector_pause = intervector_pause
        self.weight_normalization = weight_normalization

        # sparcity
        self.p = p
        self.r = r
        self.spatial = spatial

        self._check_modules(quiet)
        
    def _get_network_objects_tuple(self, ):

        return namedtuple(
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
    
    def _get_parameters(self, testing_mode=False):

        self.inp_to_inh_prob = 0.1

        neuron_parameters = {
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
                        'V_th': self.V_th[0],
                        't_ref': self.t_ref[0],
                        'tau_m': self.tau_m[0],
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
                        'V_th': self.V_th[1],
                        't_ref': self.t_ref[1],
                        'tau_m': self.tau_m[1],
                        'tau_synE': 1.0,
                        'tau_synI': 2.0,
                        'tc_theta': 1e+20
                    }
                }
        
        synapse_parameters = {
                        'exc_to_inh': {
                            'synapse_model': 'static_synapse',
                            'weight': self.w_syn[0]
                        },
                        'inh_to_exc': {
                            'synapse_model': 'static_synapse',
                            'weight': self.w_syn[1]
                        },
                        'input_to_exc': {  
                            'delay': {
                                'parametertype': 'uniform',
                                'specs': {
                                    'min': 0.1,
                                    'max': 10.0,
                                },
                            },
                            'synapse_model': self.synapse_model,
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
                    }
        
        # add extra parameters for default NEST plasticity
        if self.synapse_model not in ['stdp_tanh_synapse',
                             'stdp_gaussian_times_linear_with_separate_exp_r_dependence_synapse']:
            synapse_parameters["input_to_exc"]["Wmax"] = 1.0
            synapse_parameters["input_to_exc"]["mu_plus"] = 0.0
            synapse_parameters["input_to_exc"]["mu_minus"] = 0.0
            synapse_parameters["input_to_exc"]["tau_plus"] = 20.0
            synapse_parameters["input_to_exc"]["lambda"] = 0.01
            synapse_parameters["input_to_exc"]['alpha'] = 0.5534918994526379

        if testing_mode:
            # Disable dynamic threshold.
            neuron_parameters["exc_neurons"]["Theta_plus"] = 0.0
            # Disable synaptic plasticity.
            synapse_parameters["input_to_exc"] = disable_plasticity(
                synapse_parameters["input_to_exc"]
            )
        if hasattr(self, "exc_neurons_thresholds_"):
            # Set the recorded dynamic threshold values.
            neuron_parameters["exc_neurons"] = [
                # because, unlike synapse parameters, NEST only accepts
                # setting varying neuron parameters
                # by passing a dictionary per each neuron.
                dict(neuron_parameters["exc_neurons"], V_th=V_th)
                for V_th in self.exc_neurons_thresholds_
            ]

        neuron_parameters, synapse_parameters = map(
            convert_random_parameters_to_nest,
            (neuron_parameters, synapse_parameters),
        )

        return neuron_parameters, synapse_parameters
    
    def _create_spatial_grid_neurons(self, neuron_parameters):
        pos = nest.spatial.grid(
            shape=[10, 10],  # the number of rows and column in this grid ...
            extent=[1.0, 1.0],
        )  # the size of the grid in mm

        # Create nodes.
        exc_neuron_ids = nest.Create(
            "iaf_cond_exp_adaptive",
            params=neuron_parameters["exc_neurons"],
            positions=pos,
        )
        inh_neuron_ids = nest.Create(
            "iaf_cond_exp_adaptive",
            params=neuron_parameters["inh_neurons"],
            positions=pos,
        )

        return (exc_neuron_ids, inh_neuron_ids)
    
    def _create_spatial_free_neurons(self, neuron_parameters):
        pos = nest.spatial.free(
            nest.random.uniform(
                min=-0.5, max=0.5
            ),  # using random positions in a uniform distribution
            num_dimensions=2,
        )  # have to specify number of dimensions

        # Create nodes.
        exc_neuron_ids = nest.Create(
            "iaf_cond_exp_adaptive",  # ADAPTIVE IN ORIGINAL DIEHL
            self.n_neurons[0],
            params=neuron_parameters["exc_neurons"],
            positions=pos,
        )
        inh_neuron_ids = nest.Create(
            "iaf_cond_exp_adaptive",
            self.n_neurons[1],
            params=neuron_parameters["inh_neurons"],
            positions=pos,
        )

        return (exc_neuron_ids, inh_neuron_ids)
    
    def _create_non_spatial_neurons(self, neuron_parameters):

        # Create nodes.
        exc_neuron_ids = nest.Create(
            "iaf_cond_exp_adaptive",  # ADAPTIVE IN ORIGINAL DIEHL
            self.n_neurons[0],
            params=neuron_parameters["exc_neurons"],
        )
        inh_neuron_ids = nest.Create(
            "iaf_cond_exp_adaptive",
            self.n_neurons[1],
            params=neuron_parameters["inh_neurons"],
        )

        return (exc_neuron_ids, inh_neuron_ids)

    
    def _create_neuron_populations(self, neuron_parameters, number_of_inputs, create_spike_recorders=False):

        inputs_ids = nest.Create("parrot_neuron", number_of_inputs)
        generators_ids = nest.Create("poisson_generator", number_of_inputs)

        if any([x is not None for x in self.p]):
            if self.spatial == "grid":
                exc_neuron_ids, inh_neuron_ids = self._create_spatial_grid_neurons(neuron_parameters)
            elif self.spatial == "free":
                exc_neuron_ids, inh_neuron_ids = self._create_spatial_free_neurons(neuron_parameters)
            else:
                raise NotImplementedError(f"Unknown spatial arrangement: {self.spatial}")
        else:
            exc_neuron_ids, inh_neuron_ids = self._create_non_spatial_neurons(neuron_parameters)

        if create_spike_recorders:
            exc_neurons_spike_recorder_id = nest.Create("spike_recorder")
            inh_neurons_spike_recorder_id = nest.Create("spike_recorder")
        else:
            exc_neurons_spike_recorder_id = None
            inh_neurons_spike_recorder_id = None


        return (
            generators_ids,
            inputs_ids,
            exc_neuron_ids,
            inh_neuron_ids,
            exc_neurons_spike_recorder_id,
            inh_neurons_spike_recorder_id,
        )
    
    def _get_connection_spec(self, def_conn, p, r):
        if p == None:
            return def_conn
        else:
            if r == None:
                conn = {
                    "rule": "pairwise_bernoulli",
                    "p": p,
                }
            else:
                conn = {
                    "rule": "pairwise_bernoulli",
                    "p": p,
                    "mask": {"circular": {"radius": r}},
                }

            return conn
    
    def _connect_neuron_populations(self, 
                                    synapse_parameters,
                                    generators_ids,
                                    inputs_ids, 
                                    exc_neuron_ids,
                                    inh_neuron_ids,
                                    exc_neurons_spike_recorder_id,
                                    inh_neurons_spike_recorder_id,
                                    ):

        # Create connections.
        # ------------------
        # Static synapses from the generators
        # to parrot neurons representing the input.
        nest.Connect(
            pre=generators_ids,
            post=inputs_ids,
            conn_spec="one_to_one",
            syn_spec="static_synapse",
        )
        # Connections from the input-representing parrot neurons
        # to excitatory and inhibitory neurons.
        if hasattr(self, "weights_"):
            # Re-create connections from the saved weights.
            for exc_or_inh, exc_or_inh_neuron_ids in (
                ("exc", exc_neuron_ids),
                ("inh", inh_neuron_ids),
            ):
                conn_type_name = "input_to_" + exc_or_inh
                sparse_weight_array = self.weights_[conn_type_name]

                synapse_parameters[conn_type_name].update(
                    weight=sparse_weight_array["weight"],
                    delay=sparse_weight_array["delay"],
                )
                nest.Connect(
                    pre=np.array(inputs_ids)[sparse_weight_array["pre_index"]],
                    post=np.array(exc_neuron_ids)[sparse_weight_array["post_index"]],
                    conn_spec="one_to_one",
                    syn_spec=synapse_parameters[conn_type_name],
                )
        else:
            # Plastic synapses from the input parrot neurons
            # to the excitatory neurons.
            nest.Connect(
                pre=inputs_ids,
                post=exc_neuron_ids,
                conn_spec=self._get_connection_spec("all_to_all", self.p[0], self.r[0]),
                syn_spec=synapse_parameters["input_to_exc"],
            )
            # Static synapses from the input parrot neurons
            # to the inhibitory neurons.
            nest.Connect(
                pre=inputs_ids,
                post=inh_neuron_ids,
                conn_spec={
                    "rule": "fixed_total_number",
                    "N": int(
                        self.inp_to_inh_prob
                        * len(inputs_ids)
                        * len(inh_neuron_ids)
                    ),
                },
                syn_spec=synapse_parameters["input_to_inh"],
            )
        # Static connections from excitatory neurons
        # to their inhibitory counterparts.
        nest.Connect(
            pre=exc_neuron_ids,
            post=inh_neuron_ids,
            conn_spec=self._get_connection_spec("one_to_one", self.p[1], self.r[1]),
            syn_spec=synapse_parameters["exc_to_inh"],
        )
        # Static connections from inhibitory neurons
        # to excitatory ones.
        if self.p[2] is not None:
            nest.Connect(
                pre=inh_neuron_ids,
                post=exc_neuron_ids,
                conn_spec=self._get_connection_spec("all_to_all", self.p[2], self.r[2]),
                syn_spec=synapse_parameters["inh_to_exc"],
            )
        else:
            for current_neuron_number in range(
                self.n_neurons[1]
            ):
                nest.Connect(
                    pre=inh_neuron_ids[
                        current_neuron_number : current_neuron_number + 1
                    ],
                    post=nest.NodeCollection(
                        list(sorted(list(
                            set(exc_neuron_ids.tolist())
                            - set(
                                exc_neuron_ids[
                                    current_neuron_number : current_neuron_number + 1
                                ].tolist()
                            )
                        )))
                    ),
                    conn_spec="all_to_all",
                    syn_spec=synapse_parameters["inh_to_exc"],
                )

        # Connect neurons to spike detectors.
        if exc_neurons_spike_recorder_id is not None and inh_neurons_spike_recorder_id is not None:
            nest.Connect(
                exc_neuron_ids, exc_neurons_spike_recorder_id, conn_spec="all_to_all"
            )
            nest.Connect(
                inh_neuron_ids, inh_neurons_spike_recorder_id, conn_spec="all_to_all"
            )

        return None  

    def _create_network(self, testing_mode):
        number_of_inputs = self.n_features_in_
        n_threads = self.n_jobs
        random_state = (
            self.random_state
            if not self.random_state is None
            else generate_random_state()
        )

        # Remove existing NEST objects if any exist.
        nest.ResetKernel()
        nest.SetKernelStatus(
            {
                "resolution": 0.1,
                "local_num_threads": n_threads,
            }
        )
        nest.rng_seed = random_state

        neuron_parameters, synapse_parameters = self._get_parameters(testing_mode)

        # create neuron populations

        (
            generators_ids,
            inputs_ids,
            exc_neuron_ids,
            inh_neuron_ids,
            exc_neurons_spike_recorder_id,
            inh_neurons_spike_recorder_id,
        ) = self._create_neuron_populations(neuron_parameters, number_of_inputs, testing_mode)

        # connect neurons

        self._connect_neuron_populations(synapse_parameters,
                                         generators_ids,
                                         inputs_ids,
                                         exc_neuron_ids,
                                         inh_neuron_ids,
                                         exc_neurons_spike_recorder_id,
                                         inh_neurons_spike_recorder_id)

        populations_to_connect = [
            ("input_to_exc", inputs_ids, exc_neuron_ids),
            ("input_to_inh", inputs_ids, inh_neuron_ids),
            ("exc_to_inh", exc_neuron_ids, inh_neuron_ids),
            ("inh_to_exc", inh_neuron_ids, exc_neuron_ids),
        ]

        # Now that all connections have been created,
        # request their descriptors from NEST.
        all_connection_descriptors = {
            conn_type_name: nest.GetConnections(source=pre_ids, target=post_ids)
            for conn_type_name, pre_ids, post_ids in populations_to_connect
        }

        network_objects_tuple = self._get_network_objects_tuple()

        self.network_objects = network_objects_tuple(
            exc_neuron_ids=exc_neuron_ids,
            inh_neuron_ids=inh_neuron_ids,
            generators_ids=generators_ids,
            inputs_ids=inputs_ids,
            all_connection_descriptors=all_connection_descriptors,
            exc_neurons_spike_recorder_id=exc_neurons_spike_recorder_id,
            inh_neurons_spike_recorder_id=inh_neurons_spike_recorder_id,
        )

    def _to_spike_rates(self, X):
        return X * (self.high_rate - self.low_rate) + self.low_rate


    def run_the_simulation(self, X, y_train):
        """Encode X into input spiking rates
        and feed to the network.

        Learning is unsupervised: y_train is only used to indicate
        testing if y_train is None and training otherwise.
        If testing:
        * The simulation duration is fixed at 1 epoch.
        * STDP is disabled.
        * Re-normalization of weights is not applied even if enabled.
        * Output spiking rates are recorded and returned.
        If training:
        * Weights are saved to self.weights_.
        * Nothing is returned.
        """
        testing_mode = y_train is None
        n_epochs = self.epochs if not testing_mode else 1

        if len(np.ravel(X[X<0])) > 0:
            print("Inputs with negative values will be clipped to 0.")
            X = np.clip(X, 0)

        input_spike_rates = self._to_spike_rates(X)

        record_weights = not testing_mode
        record_spikes = testing_mode

        progress_bar = tqdm(
            total=n_epochs * len(input_spike_rates),
            disable=self.quiet,
        )
        for epoch in range(n_epochs):
            if record_spikes:
                output_spiking_rates = {"exc_neurons": [], "inh_neurons": []}
            for x in input_spike_rates:
                # Weight normalization
                if (
                    not testing_mode
                    and not self.weight_normalization
                    is None
                ):
                    for neuron_id in self.network_objects.exc_neuron_ids:
                        this_neuron_input_synapses = nest.GetConnections(
                            source=self.network_objects.inputs_ids, target=[neuron_id]
                        )
                        w = nest.GetStatus(this_neuron_input_synapses, "weight")
                        w = (
                            np.array(w)
                            * self.weight_normalization
                            / sum(w)
                        )
                        nest.SetStatus(this_neuron_input_synapses, "weight", w)

                # The simulation itself.
                nest.SetStatus(
                    self.network_objects.generators_ids, [{"rate": r} for r in x]
                )
                nest.Simulate(self.time)

                nest.SetStatus(self.network_objects.generators_ids, {"rate": 0.0})
                nest.Simulate(self.intervector_pause)

                if record_spikes:
                    for neuron_type_name, neurons_ids, spike_recorder_id in (
                        (
                            "exc_neurons",
                            self.network_objects.exc_neuron_ids,
                            self.network_objects.exc_neurons_spike_recorder_id,
                        ),
                        (
                            "inh_neurons",
                            self.network_objects.inh_neuron_ids,
                            self.network_objects.inh_neurons_spike_recorder_id,
                        ),
                    ):
                        # NEST returns all_spikes == {
                        #   'times': spike_times_array,
                        #   'senders': senders_ids_array
                        # }
                        all_spikes = nest.GetStatus(spike_recorder_id, keys="events")[0]
                        current_input_vector_output_rates = [
                            # 1000.0 * len(all_spikes['times'][
                            #   all_spikes['senders'] == current_neuron
                            # ]) / self.network_parameters['one_vector_longtitude']
                            len(
                                all_spikes["times"][
                                    all_spikes["senders"] == current_neuron
                                ]
                            )
                            for current_neuron in neurons_ids
                        ]
                        output_spiking_rates[neuron_type_name].append(
                            current_input_vector_output_rates
                        )
                        # Empty the detector.
                        nest.SetStatus(spike_recorder_id, {"n_events": 0})
                progress_bar.update()
        progress_bar.close()
        if record_weights:
            exc_neuron_ids = self.network_objects.exc_neuron_ids
            inh_neuron_ids = self.network_objects.inh_neuron_ids
            generators_ids = self.network_objects.generators_ids
            inputs_ids = self.network_objects.inputs_ids
            all_connection_descriptors = self.network_objects.all_connection_descriptors

            weights_of_all_connection_types = {
                conn_type_name: convert_neuron_ids_to_indices(
                    weights=nest.GetStatus(
                        all_connection_descriptors[conn_type_name], "weight"
                    ),
                    delays=nest.GetStatus(
                        all_connection_descriptors[conn_type_name], "delay"
                    ),
                    connection_descriptors=all_connection_descriptors[conn_type_name],
                    pre_neuron_ids=pre_ids,
                    post_neuron_ids=post_ids,
                )
                for conn_type_name, pre_ids, post_ids in (
                    ("input_to_exc", inputs_ids, exc_neuron_ids),
                    ("input_to_inh", inputs_ids, inh_neuron_ids),
                    ("exc_to_inh", exc_neuron_ids, inh_neuron_ids),
                    ("inh_to_exc", inh_neuron_ids, exc_neuron_ids),
                )
            }
            self.weights_ = weights_of_all_connection_types

        if record_spikes:
            return output_spiking_rates["exc_neurons"]


if __name__ == "__main__":
    import os
    os.environ["PYNEST_QUIET"] = "1"

    from fsnn_classifiers.datasets import load_data
    from fsnn_classifiers.components.preprocessing.grf import GRF

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold

    import numpy as np

    import json
        
    def main():

        plasticity = "stdp_tanh_synapse"


        X_train, X_test, y_train, y_test = load_data(dataset='digits', 
                                                    seed=1337,
                                                    )
        
        # we want to use cross-validation
        X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_test])
        
        # load config

        path_to_configs = './experiments/configs/digits/SparseDiehlNetwork/'
        with open(f"{path_to_configs}/{plasticity}/exp_cfg.json", 'r') as f:
            cfg = json.load(f)

        cfg["DiehlNetwork"]["epochs"] = 1
        cfg["DiehlNetwork"]["synapse_model"] = plasticity
        cfg["DiehlNetwork"]["quiet"] = False

        skf = StratifiedKFold(n_splits=5, random_state=1337, shuffle=True)
        cv_res = []
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if "Normalizer" in cfg.keys():
                nrm = Normalizer(norm=cfg["Normalizer"]["norm"].lower())
            elif "StandardScaler" in cfg.keys():
                nrm = StandardScaler()
            else:
                nrm = MinMaxScaler()

            grf = GRF(**cfg["GRF"])
            net = DiehlNetwork(**cfg["DiehlNetwork"])
            dec = LogisticRegression(**cfg["LogisticRegression"])
                    
            model = make_pipeline(nrm, grf, net, dec)
                    
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            cv_res.append(np.round(f1_score(y_test, y_pred, average='micro'),2))
                
        return cv_res

    cv_res = main()
    print(cv_res)
