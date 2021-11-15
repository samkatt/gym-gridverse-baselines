"""Entrypoint of baseline experiments on gym-gridverse

Functions as a gateway to the different baseline experiments. Accepts the
number of runs and a domain yaml file, then specifies the type of solution
method, followed by solution method specific cofigurations. For example, to run
state-based (mcts) online planning::

    python plan.py 1 runtime_info.pkl yaml/gv_empty.8x8.yaml mcts yaml/online_mcts.yaml

Note that most solution methods assume configurations are at some point passed
through a yaml file. For convenience we allow *overwriting* values in these
config files by appending any call with overwriting values, for example::

    python plan.py 5 yaml/gv_crossing.5x5.yaml po-uct yaml/online_pouct.yaml num_sims=128

Additionally (key-word) arguments are `-o/--out_file` and `-v/--verbose`.
"""

import argparse
import itertools
import logging
import pickle
import pandas as pd

from gym_gridverse.envs.yaml.factory import factory_env_from_yaml
from yaml.loader import SafeLoader

import yaml
from gym_gridverse_baselines.planning.belief import create_rejection_sampling
from gym_gridverse_baselines.planning.experiment import (
    run_mdp_episode,
    run_pomdp_episode,
)
from gym_gridverse_baselines.planning.planners import create_mcts, create_pouct


def main():
    """Main entry point of gym-gridverse-baselines planning"""

    global_parser = argparse.ArgumentParser()

    global_parser.add_argument("n", type=int, help="Number of episodes to run")
    global_parser.add_argument("domain_yaml")
    global_parser.add_argument("planner", choices=["mcts", "po-uct"])
    global_parser.add_argument("conf")
    global_parser.add_argument(
        "-o", "--out_file", help="Name of file to save to", default=None
    )
    global_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Whether to log verbose messages",
    )
    args, overwrites = global_parser.parse_known_args()

    # load domain
    env = factory_env_from_yaml(args.domain_yaml)

    with open(args.conf, "rb") as conf_file:
        conf = yaml.load(conf_file, Loader=SafeLoader)

    # overwrite `conf` with additional key=value parameters in `overwrites`
    for overwrite in overwrites:
        overwritten_key, overwritten_value = overwrite.split("=")
        conf[overwritten_key] = type(conf[overwritten_key])(overwritten_value)

    # logging: set the loggers and provide to tell planner whether to show progress bars
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    conf["show_progress_bar"] = args.verbose

    # run experiments
    if args.planner == "mcts":
        runtime_info = [
            run_mdp_episode(env, create_mcts(env, **conf)) for _ in range(args.n)
        ]
    elif args.planner == "po-uct":
        runtime_info = [
            run_pomdp_episode(
                env,
                create_pouct(env, **conf),
                create_rejection_sampling(
                    env, conf["num_particles"], conf["show_progress_bar"]
                ),
            )
            for _ in range(args.n)
        ]
    else:
        raise ValueError(f"Unsupported planning {args.planner}")

    # saving results
    for episode in range(args.n):
        for info in runtime_info[episode]:
            info["episode"] = episode

    if args.out_file:
        with open(args.out_file, "wb") as save_file:
            pickle.dump(
                {"meta": conf, "data": pd.DataFrame(itertools.chain(*runtime_info))}, save_file
            )


if __name__ == "__main__":
    main()
