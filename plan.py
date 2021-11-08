"""Entrypoint of baseline experiments on gym-gridverse

Functions as a gateway to the different baseline experiments. Accepts a domain
yaml file, then specifies the type of solution method, followed by solution
method specific cofigurations. For example, to run state-based (mcts) online
planning::

    python plan.py path/to/env/yaml mcts yaml/online_mcts.yaml

Note that most solution methods assume configurations are at some point passed
through a yaml file. For convenience we allow *overwriting* values in these
config files by appending any call with overwriting values, for example::

    python plan.py path/to/env/yaml po-uct yaml/online_pouct.yaml num_sims=128
"""

import argparse

from gym_gridverse.envs.yaml.factory import factory_env_from_yaml
from yaml.loader import SafeLoader

import yaml
from gym_gridverse_baselines.planning.belief import create_rejection_sampling
from gym_gridverse_baselines.planning.experiment import mdp_planning, pomdp_planning
from gym_gridverse_baselines.planning.planners import create_mcts, create_pouct


def main():
    """Main entry point of gym-gridverse-baselines planning"""

    global_parser = argparse.ArgumentParser()

    global_parser.add_argument("domain_yaml")
    global_parser.add_argument("observability", choices=["mcts", "po-uct"])
    global_parser.add_argument("conf")

    args, overwrites = global_parser.parse_known_args()

    # load domain
    env = factory_env_from_yaml(args.domain_yaml)

    with open(args.conf) as conf_file:
        conf = yaml.load(conf_file, Loader=SafeLoader)

    # overwrite `conf` with additional key=value parameters in `overwrites`
    for overwrite in overwrites:
        overwritten_key, overwritten_value = overwrite.split("=")
        conf[overwritten_key] = type(conf[overwritten_key])(overwritten_value)

    if args.observability == "mcts":
        mdp_planning(env, create_mcts(env, **conf))
    if args.observability == "po-uct":
        pomdp_planning(
            env,
            create_pouct(env, **conf),
            create_rejection_sampling(env, conf["num_particles"]),
        )


if __name__ == "__main__":
    main()
