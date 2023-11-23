from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from fsnn_classifiers.components.networks.utils import check_nest_objects

import nest


class BaseSpikingTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        if hasattr(self, '_validate_data'):
            # requires sklearn>=0.23
            X, y = self._validate_data(X, y, ensure_2d=True)
        else:
            X, y = check_X_y(X, y)
            self.n_features_in_ = X.shape[1]
        self.classes_ = unique_labels(y)
        
        self._create_network_if_necessary(testing_mode=False)
        self.run_the_simulation(
            X,
            y_train=y
        )

        self.is_fitted_ = True
        # Record what the last action has been,
        # in order to force re-creating the network
        # when switching actions.
        self.last_state_ = 'train'
        return self
        
    def _check_modules(self, quiet):
        nest.set_verbosity('M_QUIET')
        try:
            nest.Install("diehl_neuron_module")
        except:
            if not quiet:
                print("diehl_neuron_module is installed.")

        try:
            nest.Install("stdptanhmodule")
        except:
            if not quiet:
                print("stdptanhmodule is installed.")


    def transform(self, X):
        if hasattr(self, '_validate_data'):
            # requires sklearn>=0.23
            X = self._validate_data(X, reset=False, ensure_2d=True)
        else:
            X = check_array(X)
            self.n_features_in_ = X.shape[1]
        check_is_fitted(self, 'weights_')

        self._create_network_if_necessary(testing_mode=True)
        # Record what the last action has been,
        # in order to force re-creating the network
        # when switching actions.
        self.last_state_ = 'test'
        return self.run_the_simulation(
            X,
            y_train=None
        )


    def _create_network_if_necessary(self, testing_mode):
        if (
            # We are allowed not to re-create.
            self.warm_start
            # This is not the first call of fit().
            and hasattr(self, 'is_fitted_')
            # The previous call and the current one
            # have been the same:
            # either both have been fit(),
            # or both have been transform().
            # Otherwise, the synapses may have been created
            # with disabled plasticity.
            and (
                self.last_state_ == 'test' if testing_mode
                else self.last_state_ == 'train'
            )
            # The NEST object descriptors we have in network_objects
            # hold references to existing NEST objects
            # (which may not be the case if our estimator
            # has just been unpickled).
            and check_nest_objects(self.network_objects)
        ):
            return
        self._create_network(testing_mode)

    def __getstate__(self):
        """Modify the pickling behaviour to exclude nest object descriptors.

        When this Transformer instance is pickled,
        self.network_objects should be excluded.
        Not only in order to force recreating the network
        (_create_network_if_necessary will guess that anyway),
        but mainly because those descriptors are unpicklable.

        Source: https://docs.python.org/3/library/pickle.html#handling-stateful-objects
        """
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        
        unpicklable_properties = ['network_objects']
        for property_name in unpicklable_properties:
            if property_name in state:
                del state[property_name]

        return state
