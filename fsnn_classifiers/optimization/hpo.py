
from hyperopt import hp, fmin, tpe, space_eval, Trials
import pickle

def run_evals(run, space, n_evals, trial_name):
    trials = pickle.load(open(trial_name, "rb"))
    best = fmin(run, space, algo=tpe.suggest, trials=trials, max_evals=n_evals)
    pickle.dump(trials, open(trial_name, "wb"))
    return best


def adjust(run, space, trial_name, h_evals=10, max_evals=100, resume=False):
    if not resume:
        trials = Trials()
        pickle.dump(trials, open(trial_name, "wb"))
    else:
        trials = pickle.load(open(trial_name, "rb")) 

    n_evals = h_evals
    max_evals = max_evals
    while n_evals <= max_evals:
        best = run_evals(run, space, n_evals, trial_name)
        best_space = space_eval(space, best)
        n_evals += h_evals

    best_space = space_eval(space, best)
    print(best_space)

    return best_space