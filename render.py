"""Entrypoint of baseline experiments visualizations on gym-gridverse

Functions as a gateway to visualizing (rendering) solutions. Accepts a domain
yaml file, then specifies the type of solution method, followed by solution
method specific cofigurations. For example, to run the random policy::

    python render.py yaml/gv_crossing.5x5.yaml random

For state-based (mcts) online planning, run for example::

    python render.py yaml/gv_empty.8x8.yaml planning mcts yaml/online_mcts.yaml

Note that most solution methods assume configurations are at some point passed
through a yaml file. For convenience we allow *overwriting* values in these
config files by appending any call with overwriting values, for example::

    python render.py path/to/env/yaml planning po-uct yaml/online_pouct.yaml num_sims=128
"""

import argparse
import random

from gym_gridverse.envs.yaml.factory import factory_env_from_yaml
from yaml.loader import SafeLoader

import yaml
from gym_gridverse_baselines.planning.belief import create_rejection_sampling
from gym_gridverse_baselines.planning.planners import create_mcts, create_pouct
from gym_gridverse_baselines.visualize import belief_planner_policy, render_policy


def main():
    """Main entry point of gym-gridverse-baselines planning"""

    global_parser = argparse.ArgumentParser()
    global_parser.add_argument("domain_yaml")

    cmd_parser = global_parser.add_subparsers(dest="cmd")

    cmd_parser.add_parser("random")

    planning_parser = cmd_parser.add_parser("planning")
    planning_parser.add_argument("planner", choices=["mcts", "po-uct"])
    planning_parser.add_argument("conf")

    args, overwrites = global_parser.parse_known_args()

    # load domain
    env = factory_env_from_yaml(args.domain_yaml)

    if args.cmd == "random":
        policy = lambda _, __: random.choice(env.action_space.actions)
    elif args.cmd == "planning":

        with open(args.conf, "rb") as conf_file:
            conf = yaml.load(conf_file, Loader=SafeLoader)

        # overwrite `conf` with additional key=value parameters in `overwrites`
        for overwrite in overwrites:
            overwritten_key, overwritten_value = overwrite.split("=")
            conf[overwritten_key] = type(conf[overwritten_key])(overwritten_value)
        conf["show_progress_bar"] = True

        if args.planner == "mcts":
            planner = create_mcts(env, **conf)
            # state-based policy ==> bleief == true state
            policy = lambda s, o: planner(lambda: s)[0]
        elif args.planner == "po-uct":
            policy = belief_planner_policy(
                create_pouct(env, **conf),
                create_rejection_sampling(
                    env, conf["num_particles"], conf["show_progress_bar"]
                ),
            )
        else:
            raise ValueError(f"Unexpected planner {args.planner}")

    else:
        raise ValueError(f"Unexpected command {args.cmd}")

    render_policy(policy, env)


if __name__ == "__main__":
    main()
