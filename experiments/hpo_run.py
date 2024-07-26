from fsnn_classifiers.optimization.hpo import space_from_cfg, experiment_run, adjust
import argparse
import os
import json
from functools import partial

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", type=str)
    parser.add_argument("--ho_bound", type=int, default=10)
    parser.add_argument("--resume", action='store_true')

    return parser.parse_args()

def main(args: argparse.Namespace):

    # load hyperopt config
    cfg_path = f"{os.path.dirname(__file__)}/configs/hpo/{args.config_name}.json"
    with open(cfg_path, 'r') as fp:
        cfg = json.load(fp)

    # parse config
    model_name = cfg["model_name"]
    plasticity = cfg["plasticity"]
    dataset = cfg["dataset"]

    exp_dir = f"{os.path.dirname(__file__)}/hpo/trials/{model_name}_{plasticity}_{dataset}/"
    os.makedirs(exp_dir, exist_ok=True)

    space = space_from_cfg(cfg["space"])

    run = partial(experiment_run, cfg=cfg)

    best_space = adjust(run, space, f"{exp_dir}/trial.pkl", h_evals=1, max_evals=args.ho_bound, resume=args.resume)
    print(best_space)

if __name__ == "__main__":
    args = parse_args()
    main(args)