"""Runs random policy experiments"""

import random

from gym_gridverse.envs.inner_env import InnerEnv


def random_episode(env: InnerEnv):
    """Runs an episode in ``env`` picking random actions"""

    rewards = []

    terminal = False

    env.reset()

    while not terminal:

        action = random.choice(env.action_space.actions)
        reward, terminal = env.step(action)

        print(f"{action} => {env.state.agent}")

        rewards.append(reward)

    return rewards
