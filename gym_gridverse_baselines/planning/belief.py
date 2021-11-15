"""Functionality of tracking belief in gym-gridverse

Mostly provided through ``pomdp_belief_tracking``::

    - rejection sampling

"""

from operator import eq

import pomdp_belief_tracking.pf.rejection_sampling as RS
import pomdp_belief_tracking.types as belief_types
from gym_gridverse.envs.inner_env import InnerEnv


def create_rejection_sampling(
    env: InnerEnv, n: int, show_progress_bar: bool
) -> belief_types.Belief:
    """Creates rejection sampling update

    :param n: number of samples to track
    :param show_progress_bar: whether or not to print a progress bar
    """

    def sim(s, a):
        next_state, _, _ = env.functional_step(s, a)
        obs = env.functional_observation(next_state)

        return next_state, obs

    process_acpt: RS.ProcessAccepted = (
        RS.AcceptionProgressBar(n) if show_progress_bar else RS.accept_noop
    )

    return belief_types.Belief(
        env.functional_reset,
        RS.create_rejection_sampling(sim, n, eq, process_acpt=process_acpt),
    )
