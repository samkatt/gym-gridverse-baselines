"""Runs a planning experiments

Currently provided:
    - ``mdp_planning``: runs a state-based (MDP) planner
    - ``pomdp_planning``: runs a belief/observation-based (POMDP) planner
"""
import itertools
import logging

from gym_gridverse.action import Action as GridverseAction
from gym_gridverse.envs.inner_env import InnerEnv
from online_pomdp_planning.types import Planner
from pomdp_belief_tracking.types import Belief


def run_mdp_episode(env: InnerEnv, planner: Planner):
    """Runs an episode in ``env`` using ``planner`` to pick actions

    NOTE: provides ``planner`` with the (true) state
    """

    # `runtime_info[t]` gives a dictionary of runtime info at timestep `t`
    runtime_info = []
    env.reset()

    logging.info("Starting episode in %s", env.state)

    for timestep in itertools.count(0, 1):

        # "sampling from belief" here just means directly using state
        action, planning_info = planner(lambda: env.state)
        assert isinstance(action, GridverseAction)

        reward, terminal = env.step(action)

        logging.info("%s => %s", action, env.state.agent)

        runtime_info.append(
            {
                "timestep": timestep,
                "reward": reward,
                "terminal": terminal,
                **planning_info,
            }
        )

        if terminal:
            break

    return runtime_info


def run_pomdp_episode(env: InnerEnv, planner: Planner, belief: Belief):
    """Runs an episode in ``env`` using ``planner`` to pick actions and ``belief`` for state estimation"""

    # `runtime_info[t]` gives a dictionary of runtime info at timestep `t`
    runtime_info = []
    env.reset()

    logging.info("Starting episode in %s", env.state.agent)

    for timestep in itertools.count(0, 1):

        action, planning_info = planner(belief.sample)
        assert isinstance(action, GridverseAction)

        reward, terminal = env.step(action)
        obs = env.observation

        logging.info("%s => %s", action, env.state.agent)

        belief_info = belief.update(action, obs)

        runtime_info.append(
            {
                "timestep": timestep,
                "reward": reward,
                "terminal": terminal,
                "planning_info": planning_info,
                "belief_info": belief_info,
            }
        )

    return runtime_info
