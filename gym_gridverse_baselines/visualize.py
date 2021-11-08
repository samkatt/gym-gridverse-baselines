"""Visualization code

Provides::
    - code to render a policy's trajectory :func:`render_policy`

"""

import time
from typing import Callable

import online_pomdp_planning.types as planning_types
import pomdp_belief_tracking.types as belief_types
from gym_gridverse.action import Action
from gym_gridverse.envs.inner_env import InnerEnv
from gym_gridverse.observation import Observation
from gym_gridverse.recording import HUD_Info
from gym_gridverse.rendering import GridVerseViewer
from gym_gridverse.state import State

Policy = Callable[[State, Observation], Action]


def render_policy(policy: Policy, env: InnerEnv):
    """Renders (visualizes) a trajectory in ``env`` picking actions with ``pol``


    :param policy: Anything that returns an action given an observation
    :param env: The environment to interact with
    """
    spf = 1.5

    state_viewer = GridVerseViewer(env.state_space.grid_shape, caption="State")
    observation_viewer = GridVerseViewer(
        env.observation_space.grid_shape, caption="Observation"
    )

    hud_info: HUD_Info = {
        "action": None,
        "reward": None,
        "ret": None,
        "done": None,
    }

    done = True
    ret = 0.0

    env.reset()

    state_viewer.render(env.state, return_rgb_array=True, **hud_info)
    observation_viewer.render(env.observation, return_rgb_array=True, **hud_info)

    done = False
    while not done:

        t = time.time()
        action = policy(env.state, env.observation)

        # make sure the viewer has at least ``spf`` time to look at state
        time.sleep(max(0, spf - (time.time() - t)))

        reward, done = env.step(action)

        ret += reward
        hud_info = {
            "action": action,
            "reward": reward,
            "ret": ret,
            "done": done,
        }

        state_viewer.render(env.state, return_rgb_array=True, **hud_info)
        observation_viewer.render(env.observation, return_rgb_array=True, **hud_info)


def belief_planner_policy(
    planner: planning_types.Planner, belief: belief_types.Belief
) -> Policy:
    """Creates a :class:`Policy` out of ``planner`` and ``belief``"""

    action = None

    def policy(state: State, observation: Observation) -> Action:
        """The policy (return), makes use of and updates ``action``

        :param state: ignored, since we are doing *belief* (observation) based planning
        :param observation: used to update the belief together with last ``action``
        """
        nonlocal action

        if action is not None:
            belief.update(action, observation)

        action, _ = planner(belief.sample)
        return action

    return policy
