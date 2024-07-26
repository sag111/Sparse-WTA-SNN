import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


def get_classes_rank_per_one_vector(spike_rates, set_of_classes, assignments):
    number_of_classes = len(set_of_classes)
    summed_rates = [0] * number_of_classes
    number_of_neurons_assigned_to_this_class = [0] * number_of_classes
    for class_number, current_class in enumerate(set_of_classes):
        number_of_neurons_assigned_to_this_class = len(np.where(assignments == current_class)[0])
        if number_of_neurons_assigned_to_this_class == 0:
            continue
        summed_rates[class_number] = np.sum(spike_rates[assignments == current_class]) / number_of_neurons_assigned_to_this_class
    return np.argsort(summed_rates)[::-1]

def get_assignments(rates, y):
    neurons_number = rates.shape[1]
    assignments = [-1] * neurons_number
    maximum_rates_for_all_neurons = [0] * neurons_number
    for current_class in set(y):
        number_of_vectors_in_the_current_class = len(np.where(y == current_class)[0])
        rates_for_this_class = np.sum(rates[y == current_class], axis=0) / number_of_vectors_in_the_current_class
        for i in range(neurons_number):
            if rates_for_this_class[i] > maximum_rates_for_all_neurons[i]:
                maximum_rates_for_all_neurons[i] = rates_for_this_class[i]
                assignments[i] = current_class
    return assignments


class DiehlDecoder(BaseEstimator, ClassifierMixin):

    def fit(self, X, y):
        if hasattr(self, '_validate_data'):
            # requires sklearn>=0.23
            X, y = self._validate_data(X, y, ensure_2d=True)
        else:
            X, y = check_X_y(X, y)
            self.n_features_in_ = X.shape[0]
        self.classes_ = unique_labels(y)

        self.assignments_ = get_assignments(X, y)

        self.is_fitted_ = True
        return self

    def predict(self, X):
        X = check_array(X)
        check_is_fitted(self, 'assignments_')
        rates = X

        number_of_classes = len(self.classes_)
        neurons_number = rates.shape[1]
        assignments = self.assignments_
        class_certainty_ranks = [
            get_classes_rank_per_one_vector(
                this_vector_rates, self.classes_, assignments
            )
            for this_vector_rates in rates
        ]
        y_predicted = np.array(class_certainty_ranks)[:,0]
        return y_predicted